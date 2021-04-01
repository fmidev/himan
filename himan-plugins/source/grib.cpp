#include "grib.h"
#include "NFmiGrib.h"
#include "file_accessor.h"
#include "grid.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include "plugin_factory.h"
#include "producer.h"
#include "reduced_gaussian_grid.h"
#include "s3.h"
#include "stereographic_grid.h"
#include "timer.h"
#include "transverse_mercator_grid.h"
#include "util.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <ogr_spatialref.h>

using namespace std;
using namespace himan;
using namespace himan::plugin;

#include "radon.h"

#include "cuda_helper.h"
#include "packed_data.h"

#define BitMask1(i) (1u << i)
#define BitTest(n, i) !!((n)&BitMask1(i))

std::string GetParamNameFromGribShortName(const std::string& paramFileName, const std::string& shortName);
void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len, size_t unpackedLen);

static mutex singleGribMessageCounterMutex;
static map<string, std::mutex> singleGribMessageCounterMap;

long DetermineProductDefinitionTemplateNumber(long agg, long proc, long ftype)
{
	// Determine which code table to use to represent our data
	// We have four options:
	//
	//  aggregation | processing
	//  ------------+-----------
	//       0      |     0
	//       0      |     1
	//       1      |     0
	//       1      |     1

	using namespace himan;

	long templateNumber = 0;  // default: No aggregation, no processing

	if (agg == kUnknownAggregationType && proc == kUnknownProcessingType)
	{
		// Case 1: No aggregation, no processing, for example T-K

		templateNumber = 0;  // Analysis or forecast at a horizontal level or in a horizontal layer at a point in time

		if (ftype == kEpsPerturbation || ftype == kEpsControl)
		{
			templateNumber = 1;  // Individual ensemble forecast, control and perturbed, at a horizontal level or in
			                     // a horizontal layer at a point in time
		}
	}
	else if (agg == kUnknownAggregationType && proc != kUnknownProcessingType)
	{
		// Case 2: not aggregated, but processed, for example F50-T-K

		switch (proc)
		{
			default:
				templateNumber = 2;  // Derived forecasts based on all ensemble members at a horizontal level or in a
				                     // horizontal layer at a point in time.
				break;
			case kProbabilityGreaterThan:
			case kProbabilityLessThan:
			case kProbabilityGreaterThanOrEqual:
			case kProbabilityLessThanOrEqual:
			case kProbabilityBetween:
			case kProbabilityEquals:
			case kProbabilityNotEquals:
			case kProbabilityEqualsIn:
				// probabilities
				templateNumber =
				    5;  // Probability forecasts at a horizontal level or in a horizontal layer at a point in time
				break;

			case kFractile:
				templateNumber =
				    6;  //  Percentile forecasts at a horizontal level or in a horizontal layer at a point in time
				break;
		}
	}
	else if (agg != kUnknownAggregationType && proc == kUnknownProcessingType)
	{
		// Case 3: aggregated, but not processed, for example RR-1-H

		templateNumber = 8;  // Average, accumulation, extreme values or other statistically processed values
		                     // at a horizontal level or in a horizontal layer in a continuous or
		                     // non-continuous time interval

		if (ftype == kEpsPerturbation || ftype == kEpsControl)
		{
			templateNumber = 11;  // Individual ensemble forecast, control and perturbed, at a horizontal level or
			                      // in a horizontal layer, in a continuous or non-continuous time interval.
		}
	}
	else if (agg != kUnknownAggregationType && proc != kUnknownProcessingType)
	{
		// Case 4: aggregated and processed, for example PROB-RR-1

		switch (proc)
		{
			default:
				templateNumber = 12;  // Derived forecasts based on all ensemble members at a horizontal level or in a
				                      // horizontal layer, in a continuous or non-continuous interval
				break;
			case kProbabilityGreaterThan:
			case kProbabilityLessThan:
			case kProbabilityGreaterThanOrEqual:
			case kProbabilityLessThanOrEqual:
			case kProbabilityBetween:
			case kProbabilityEquals:
			case kProbabilityNotEquals:
			case kProbabilityEqualsIn:
				// probabilities
				templateNumber = 9;  //  Probability forecasts at a horizontal level or in a horizontal layer in a
				                     //  continuous or non-continuous time interval
				break;
			case kFractile:
				templateNumber = 10;  // Percentile forecasts at a horizontal level or in a horizontal layer in a
				                      // continuous or non-continuous time interval
				break;
		}
	}

	return templateNumber;
}

template <typename T>
long DetermineBitsPerValue(const vector<T>& values, int precision)
{
	/*
	 * Calculate the minimum amount of bits required to represent the data in the precision specified.
	 * We calculate the number manually here because if we let grib_api do it it leads to the data
	 * being packed twice:
	 * - first time with 24 bits when we set the data array (grib_set_double_array(...))
	 * - second time when we request precision change (grib_set_long(changeDecimalPrecision, ...))
	 */

	himan::logger log("grib");

	int bitsPerValue = 24;  // default if no precision is set

	if (precision == himan::kHPMissingInt)
	{
		return bitsPerValue;
	}

	// https://www.wmo.int/pages/prog/www/WDM/Guides/Guide-binary-2.html

	// define manual minmax search as std::minmax_element uses std::less
	// for comparison which does not work well with nan
	T min = himan::MissingValue<T>(), max = himan::MissingValue<T>();

	for (const auto& v : values)
	{
		// man fmin:
		// "If one argument is a NaN, the other argument is returned."
		min = fmin(min, v);
		max = fmax(max, v);
	}

	// Required scale value to reach wanted precision
	const T D = static_cast<T>(std::pow(10, precision));

	// Range of scaled data, ie the largest value we must be able to write
	const int range = static_cast<int>(ceil(D * max - D * min));

	if (himan::IsMissing(min) || himan::IsMissing(max) || range == 0)
	{
		// static grid (max == min)
		bitsPerValue = 0;
	}
	else
	{
		// Number of bits required to represent the largest value
		// Range is incremented with one because we have to able to encode
		// value zero also. For example if range=4 and precision=0, possible values
		// are 0,1,2,3,4 --> 3 bits are required.

		bitsPerValue = static_cast<int>(ceil(log2(range + 1)));
	}

	// Fallback if the calculation above fails
	if (bitsPerValue < 0 || bitsPerValue > 24)
	{
		log.Error(fmt::format("Bits per value calculation failed for precision={}, defaulting to 24", precision));
		log.Trace("D=" + to_string(static_cast<int>(D)) + " min=" + to_string(min) + " max=" + to_string(max) +
		          " range=" + to_string(range));
		bitsPerValue = 24;
	}

	return bitsPerValue;
}

template <typename T>
void EncodePrecipitationFormToGrib2(vector<T>& arr)
{
	for (auto& val : arr)
	{
		if (himan::IsMissing(val))
			continue;

		switch (static_cast<int>(val))
		{
			// rain
			case 1:
				break;
			// drizzle
			case 0:
				val = 11;
				break;
			// sleet
			case 2:
				val = 7;
				break;
			// snow
			case 3:
				val = 5;
				break;
			// freezing drizzle
			case 4:
				val = 12;
				break;
			// freezing rain
			case 5:
				val = 3;
				break;
			// graupel
			case 6:
				val = 9;
				break;
			// snow pellet
			case 7:
				val = 13;
				break;
			// ice pellet
			case 8:
				break;
			default:
				throw runtime_error("Unknown precipitation form: " + to_string(val));
		}
	}
}

template <typename T>
void DecodePrecipitationFormFromGrib2(vector<T>& arr)
{
	for (auto& val : arr)
	{
		if (himan::IsMissing(val))
			continue;

		switch (static_cast<int>(val))
		{
			// rain
			case 1:
				break;
			// drizzle
			case 11:
				val = 0.;
				break;
			// sleet
			case 7:
				val = 2.;
				break;
			// snow
			case 5:
				val = 3.;
				break;
			// freezing drizzle
			case 12:
				val = 4.;
				break;
			// freezing rain
			case 3:
				val = 5.;
				break;
			// graupel
			case 9:
				val = 6;
				break;
			// snow pellet
			case 13:
				val = 7;
				break;
			// ice pellet
			case 8:
				break;
			default:
				throw runtime_error("Unknown precipitation form: " + to_string(val));
		}
	}
}

himan::time_duration DurationFromTimeRange(long unitOfTimeRange)
{
	using namespace himan;

	switch (unitOfTimeRange)
	{
		case 1:
			return ONE_HOUR;
		case 10:
			return THREE_HOURS;
		case 11:
			return SIX_HOURS;
		case 12:
			return TWELVE_HOURS;
		case 0:
			return time_duration("00:01:00");
		case 254:
		case 13:
			return FIFTEEN_MINUTES;
		case 14:
			return time_duration("00:30:00");
		case -999:
			return time_duration();
		default:
			throw invalid_argument("Unsupported unit of time range: " + to_string(unitOfTimeRange));
	}
}

grib::grib()
{
	itsLogger = logger("grib");
}

void WriteAreaAndGrid(NFmiGribMessage& message, const shared_ptr<himan::grid>& grid, const producer& prod)
{
	const long edition = message.Edition();
	HPScanningMode scmode = kUnknownScanningMode;

	auto firstGridPoint = grid->FirstPoint();

	if (edition == 2)
	{
		while (firstGridPoint.X() < 0)
			firstGridPoint.X(firstGridPoint.X() + 360.);
	}

	logger logr("grib");
	// UVRelativeToGrid is set in ToFile()

	switch (grid->Type())
	{
		case kLatitudeLongitude:
		{
			auto rg = dynamic_pointer_cast<latitude_longitude_grid>(grid);

			long gridType = 0;  // Grib 1

			auto lastGridPoint = grid->LastPoint();

			if (edition == 2)
			{
				while (lastGridPoint.X() < 0)
					lastGridPoint.X(lastGridPoint.X() + 360.);
				gridType = message.GridTypeToAnotherEdition(gridType, 2);
			}

			message.GridType(gridType);

			message.X0(firstGridPoint.X());
			message.X1(lastGridPoint.X());
			message.Y0(firstGridPoint.Y());
			message.Y1(lastGridPoint.Y());

			message.iDirectionIncrement(rg->Di());
			message.jDirectionIncrement(rg->Dj());

			message.SizeX(static_cast<long>(rg->Ni()));
			message.SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			break;
		}

		case kRotatedLatitudeLongitude:
		{
			auto rg = dynamic_pointer_cast<rotated_latitude_longitude_grid>(grid);

			long gridType = 10;  // Grib 1

			// In grib we put rotated coordinates as first and last point
			firstGridPoint = rg->Rotate(grid->FirstPoint());
			auto lastGridPoint = rg->Rotate(rg->LastPoint());

			if (edition == 2)
			{
				while (firstGridPoint.X() < 0)
					firstGridPoint.X(firstGridPoint.X() + 360.);
				while (lastGridPoint.X() < 0)
					lastGridPoint.X(lastGridPoint.X() + 360.);
				gridType = message.GridTypeToAnotherEdition(gridType, 2);
			}

			message.GridType(gridType);

			message.X0(firstGridPoint.X());
			message.Y0(firstGridPoint.Y());
			message.X1(lastGridPoint.X());
			message.Y1(lastGridPoint.Y());

			message.SouthPoleX(rg->SouthPole().X());
			message.SouthPoleY(rg->SouthPole().Y());

			message.iDirectionIncrement(rg->Di());
			message.jDirectionIncrement(rg->Dj());

			message.GridType(gridType);

			message.SizeX(static_cast<long>(rg->Ni()));
			message.SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			break;
		}

		case kStereographic:
		{
			auto rg = dynamic_pointer_cast<stereographic_grid>(grid);

			long gridType = 5;  // Grib 1

			if (edition == 2)
			{
				gridType = message.GridTypeToAnotherEdition(gridType, 2);
			}

			message.GridType(gridType);

			message.X0(firstGridPoint.X());
			message.Y0(firstGridPoint.Y());

			message.GridOrientation(rg->Orientation());

			message.XLengthInMeters(rg->Di());
			message.YLengthInMeters(rg->Dj());

			message.SizeX(static_cast<long>(rg->Ni()));
			message.SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			if (edition == 2)
			{
				message.SetLongKey("LaDInDegrees", 60);
			}

			break;
		}

		case kReducedGaussian:
		{
			auto gg = dynamic_pointer_cast<reduced_gaussian_grid>(grid);

			long gridType = 4;  // Grib 1

			if (edition == 2)
			{
				gridType = message.GridTypeToAnotherEdition(gridType, 2);
			}

			message.GridType(gridType);

			const double lonMin = firstGridPoint.X();
			const double lonMax = gg->LatLon(gg->NumberOfPointsAlongParallels()[gg->N()], gg->N()).X();
			const double latMin = gg->Latitudes().back();
			const double latMax = gg->Latitudes().front();

			message.X0(lonMin);
			message.Y0(latMax);
			message.X1(lonMax);
			message.Y1(latMin);

			message.SetLongKey("iDirectionIncrement", 65535);
			message.SetLongKey("numberOfPointsAlongAParallel", 65535);

			message.SetLongKey("N", static_cast<long>(gg->N()));

			message.PL(gg->NumberOfPointsAlongParallels());

			scmode = kTopLeft;

			break;
		}

		case kLambertConformalConic:
		{
			auto lccg = dynamic_pointer_cast<lambert_conformal_grid>(grid);

			long gridType = 3;  // Grib 1

			if (edition == 2)
			{
				gridType = message.GridTypeToAnotherEdition(gridType, 2);
			}

			message.GridType(gridType);

			message.X0(firstGridPoint.X());
			message.Y0(firstGridPoint.Y());

			message.GridOrientation(lccg->Orientation());

			message.XLengthInMeters(lccg->Di());
			message.YLengthInMeters(lccg->Dj());

			message.SizeX(static_cast<long>(lccg->Ni()));
			message.SizeY(static_cast<long>(lccg->Nj()));

			message.SetDoubleKey("Latin1InDegrees", lccg->StandardParallel1());

			if (!IsKHPMissingValue(lccg->StandardParallel2()))
			{
				message.SetDoubleKey("Latin2InDegrees", lccg->StandardParallel2());
			}

			scmode = lccg->ScanningMode();

			if (edition == 2)
			{
				message.SetDoubleKey("LaDInDegrees", lccg->StandardParallel1());
			}

			break;
		}

		case kTransverseMercator:
		{
			auto tmg = dynamic_pointer_cast<transverse_mercator_grid>(grid);

			if (edition == 1)
			{
				logr.Fatal("transverse mercator only supported with grib2");
				himan::Abort();
			}

			message.GridType(12);

			message.SizeX(static_cast<long>(tmg->Ni()));
			message.SizeY(static_cast<long>(tmg->Nj()));

			message.SetLongKey("longitudeOfReferencePoint", static_cast<long>(tmg->Orientation() * 1000000));
			message.SetLongKey("latitudeOfReferencePoint", static_cast<long>(tmg->StandardParallel() * 1000000));

			message.SetLongKey("XR", 100 * static_cast<long>(0));  // TODO
			message.SetLongKey("YR", 100 * static_cast<long>(0));
			message.SetLongKey("scaleFactorAtReferencePoint", static_cast<long>(tmg->Scale()));
			message.SetLongKey("X1", 0);  // TODO
			message.SetLongKey("X2", 0);
			message.SetLongKey("Y1", 0);
			message.SetLongKey("Y2", 0);
			message.SetLongKey("Di", static_cast<long>(tmg->Di() * 100));
			message.SetLongKey("Dj", static_cast<long>(tmg->Dj() * 100));

			scmode = tmg->ScanningMode();

			break;
		}

		default:
			logr.Fatal("Invalid projection while writing grib: " + to_string(grid->Type()));
			himan::Abort();
	}

#if 0
	// Earth shape is not set yet, as it will change many of the test results (metadata changes)
	// and we don't want to do that until we have set the *correct* radius for those producers
	// that we have it for. Remember that at this point we force all producers to use radius
	// found from newbase.

	// Set earth shape

	const double a = grid->EarthShape().A(), b = grid->EarthShape().B();

	if (a == b)
	{
		// sphere
		if (edition == 1)
		{
			message.SetLongKey("earthIsOblate", 0);

			long flag = message.ResolutionAndComponentFlags();

			flag &= ~(1UL << 6);

			message.ResolutionAndComponentFlags(flag);
		}
		else
		{
			if (a == 6367470)
			{
				message.SetLongKey("shapeOfTheEarth", 0);
			}
			else
			{
				message.SetLongKey("shapeOfTheEarth", 1);
				message.SetLongKey("scaleFactorOfRadiusOfSphericalEarth", 1);
				message.SetLongKey("scaledValueOfRadiusOfSphericalEarth", a);
			}
		}
	}
	else
	{
		itsLogger.Fatal("A spheroid, really?");
		himan::Abort();
	}
#endif
	message.Centre(prod.Centre() == kHPMissingInt ? 86 : prod.Centre());
	message.Process(prod.Process() == kHPMissingInt ? 255 : prod.Process());

	bool iNegative, jPositive;

	switch (scmode)
	{
		case kTopLeft:
			iNegative = false;
			jPositive = false;
			break;

		case kTopRight:
			iNegative = true;
			jPositive = false;
			break;

		case kBottomLeft:
			iNegative = false;
			jPositive = true;
			break;

		case kBottomRight:
			iNegative = true;
			jPositive = true;
			break;

		default:
			logr.Fatal("Unknown scanning mode when writing grib");
			himan::Abort();
	}

	message.IScansNegatively(iNegative);
	message.JScansPositively(jPositive);
}

void WriteTime(NFmiGribMessage& message, const forecast_time& ftime, const producer& prod, const param& par)
{
	message.DataDate(stol(ftime.OriginDateTime().ToDate()));
	message.DataTime(stol(ftime.OriginDateTime().String("%H%M")));

	logger logr("grib");
	if (message.Edition() == 1)
	{
		time_duration stepUnit = ONE_HOUR;

		if (ftime.Step().Minutes() % 60 != 0)
		{
			logr.Fatal("Sub-hour output ony in grib2");
			himan::Abort();
		}
		else if (ftime.Step().Hours() > 255)  // Forecast with stepvalues that don't fit in one byte
		{
			const long hours = ftime.Step().Hours();

			if (hours % 3 == 0 && hours / 3 < 255)
			{
				message.UnitOfTimeRange(10);  // 3 hours
				stepUnit = THREE_HOURS;
			}
			else if (hours % 6 == 0 && hours / 6 < 255)
			{
				message.UnitOfTimeRange(11);  // 6 hours
				stepUnit = SIX_HOURS;
			}
			else if (hours % 12 == 0 && hours / 12 < 255)
			{
				message.UnitOfTimeRange(12);  // 12 hours
				stepUnit = TWELVE_HOURS;
			}
			else
			{
				logr.Fatal("Step too large, unable to continue");
				himan::Abort();
			}
		}

		// These are used if parameter is aggregated
		long p1 = ((ftime.Step() + par.Aggregation().TimeOffset()) / static_cast<int>(stepUnit.Hours())).Hours();
		long p2 = p1 + par.Aggregation().TimeDuration().Hours() / static_cast<int>(stepUnit.Hours());

		switch (par.Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				// Forecast product valid for reference time + P1 (P1 > 0)
				message.TimeRangeIndicator(0);
				message.P1((ftime.Step() / static_cast<int>(stepUnit.Hours())).Hours());
				break;
			case kMaximum:
			case kMinimum:
				// Product with a valid time ranging between reference time + P1 and reference time + P2
				message.TimeRangeIndicator(2);

				if (p1 < 0)
				{
					logr.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				message.P1(p1);
				message.P2(p2);
				break;
			case kAverage:
				// Average (reference time + P1 to reference time + P2)
				message.TimeRangeIndicator(3);

				if (p1 < 0)
				{
					logr.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				message.P1(p1);
				message.P2(p2);
				break;
			case kAccumulation:
				// Accumulation (reference time + P1 to reference time + P2) product considered valid at reference time
				// + P2
				message.TimeRangeIndicator(4);

				if (p1 < 0)
				{
					logr.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}
				message.P1(p1);
				message.P2(p2);
				break;
			case kDifference:
				// Difference (reference time + P2 minus reference time + P1) product considered valid at reference time
				// + P2
				message.TimeRangeIndicator(5);

				if (p1 < 0)
				{
					logr.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				message.P1(p1);
				message.P2(p2);
				break;
		}

		ASSERT(message.TimeRangeIndicator() != 10);
	}
	else
	{
		// Set in WriteParam()
		const long templateNumber = message.ProductDefinitionTemplateNumber();

		// leadtime for this prognosis
		long unitOfTimeRange = 1;  // hours
		long stepValue = ftime.Step().Hours();

		if (ftime.Step().Minutes() % 60 != 0)
		{
			unitOfTimeRange = 0;  // minutes
			stepValue = ftime.Step().Minutes();
		}

		message.UnitOfTimeRange(unitOfTimeRange);
		// Statistical processing is set in WriteParameter()
		switch (par.Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				message.ForecastTime(stepValue);
				break;
			case kAverage:
			case kAccumulation:
			case kDifference:
			case kMinimum:
			case kMaximum:
				// duration of the aggregation period, if any

				if (par.Aggregation().TimeDuration().Empty() == false)
				{
					long unitForTimeRange = 1;  // hours
					long lengthOfTimeRange = par.Aggregation().TimeDuration().Hours();

					long timeOffset = par.Aggregation().TimeOffset().Hours();

					if (par.Aggregation().TimeDuration().Minutes() % 60 != 0)
					{
						unitForTimeRange = 0;  // minutes
						lengthOfTimeRange = par.Aggregation().TimeDuration().Minutes();
					}

					if (unitOfTimeRange == unitForTimeRange)
					{
						stepValue += timeOffset;
					}
					else if (unitOfTimeRange == 1 && unitForTimeRange == 0)
					{
						stepValue += timeOffset / 60;
					}
					else if (unitOfTimeRange == 0 && unitForTimeRange == 1)
					{
						stepValue += timeOffset * 60;
					}

					message.SetLongKey("indicatorOfUnitForTimeRange", unitForTimeRange);
					message.LengthOfTimeRange(lengthOfTimeRange);
				}

				message.ForecastTime(stepValue);  // start step

				// for productDefinitionTemplateNumber 9,10,11,12,13,14,34,43,47,61,73,73
				// grib2 has extra keys for "end of overall time interval"
				if (templateNumber >= 9 && templateNumber <= 14)
				{
					raw_time endOfInterval = raw_time(ftime.ValidDateTime());

					if (par.Aggregation().TimeDuration().Empty() == false)
					{
						// If parameter is time-aggregated, adjust the end date. Otherwise
						// end date is valid time.

						endOfInterval += par.Aggregation().TimeOffset() + par.Aggregation().TimeDuration();
					}

					message.SetLongKey("yearOfEndOfOverallTimeInterval", stol(endOfInterval.String("%Y")));
					message.SetLongKey("monthOfEndOfOverallTimeInterval", stol(endOfInterval.String("%m")));
					message.SetLongKey("dayOfEndOfOverallTimeInterval", stol(endOfInterval.String("%d")));
					message.SetLongKey("hourOfEndOfOverallTimeInterval", stol(endOfInterval.String("%H")));
					message.SetLongKey("minuteOfEndOfOverallTimeInterval", stol(endOfInterval.String("%M")));
					message.SetLongKey("secondOfEndOfOverallTimeInterval", 0);
				}
				break;
		}
	}
}

void WriteParameter(NFmiGribMessage& message, const param& par, const producer& prod, const forecast_type& ftype)
{
	logger logr("grib");
	if (message.Edition() == 1)
	{
		if (par.GribTableVersion() != kHPMissingInt && par.GribIndicatorOfParameter() != kHPMissingInt)
		{
			// In radon table version is a parameter property, not a
			// producer property

			message.Table2Version(par.GribTableVersion());
			message.ParameterNumber(par.GribIndicatorOfParameter());
		}
		else if (prod.Id() != kHPMissingInt)  // no-database example has 999999 as producer
		{
			logr.Warning("Parameter " + par.Name() + " does not have mapping for producer " + to_string(prod.Id()) +
			             " in radon, setting table2version to 203");
			message.ParameterNumber(0);
			message.Table2Version(203);
		}
	}
	else if (message.Edition() == 2)
	{
		if (par.GribParameter() == kHPMissingInt)
		{
			logr.Warning("Parameter information not found from radon for producer " + to_string(prod.Id()) + ", name " +
			             par.Name());
		}
		else
		{
			message.ParameterNumber(par.GribParameter());
			message.ParameterCategory(par.GribCategory());
			message.ParameterDiscipline(par.GribDiscipline());
		}

		const auto aggType = par.Aggregation().Type();
		const auto procType = par.ProcessingType().Type();
		const long templateNumber = DetermineProductDefinitionTemplateNumber(aggType, procType, ftype.Type());

		message.ProductDefinitionTemplateNumber(templateNumber);

		switch (aggType)
		{
			case kAverage:
				message.TypeOfStatisticalProcessing(0);
				break;
			case kAccumulation:
				message.TypeOfStatisticalProcessing(1);
				break;
			case kMaximum:
				message.TypeOfStatisticalProcessing(2);
				break;
			case kMinimum:
				message.TypeOfStatisticalProcessing(3);
				break;
			default:
			case kUnknownAggregationType:
				break;
		}

		long num = static_cast<long>(par.ProcessingType().NumberOfEnsembleMembers());

		if (num == static_cast<long>(kHPMissingInt))
		{
			num = 0;
		}

		switch (procType)
		{
			default:
				break;
			case kProbabilityGreaterThan:  // Probability of event above upper limit
			case kProbabilityGreaterThanOrEqual:
				message.SetLongKey("probabilityType", 1);
				message.SetLongKey("scaledValueOfUpperLimit", static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityLessThan:  // Probability of event below lower limit
			case kProbabilityLessThanOrEqual:
				message.SetLongKey("probabilityType", 0);
				message.SetLongKey("scaledValueOfLowerLimit", static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityBetween:
				message.SetLongKey("probabilityType", 192);
				message.SetLongKey("scaledValueOfLowerLimit", static_cast<long>(par.ProcessingType().Value()));
				message.SetLongKey("scaledValueOfUpperLimit", static_cast<long>(par.ProcessingType().Value2()));
				break;
			case kProbabilityEquals:
				message.SetLongKey("probabilityType", 193);
				message.SetLongKey("scaledValueOfLowerLimit", static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityNotEquals:
				message.SetLongKey("probabilityType", 193);
				message.SetLongKey("scaledValueOfLowerLimit", static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityEqualsIn:
				message.SetLongKey("probabilityType", 194);
				break;
			case kFractile:
				message.SetLongKey("percentileValue", static_cast<long>(par.ProcessingType().Value()));
				break;
			case kEnsembleMean:
				message.SetLongKey("derivedForecast", 0);
				message.SetLongKey("numberOfForecastsInEnsemble", num);
				break;
			case kSpread:
				message.SetLongKey("derivedForecast", 4);
				message.SetLongKey("numberOfForecastsInEnsemble", num);
				break;
			case kStandardDeviation:
				message.SetLongKey("derivedForecast", 2);
				message.SetLongKey("numberOfForecastsInEnsemble", num);
				break;
			case kEFI:
				message.SetLongKey("derivedForecast", 199);
				message.SetLongKey("numberOfForecastsInEnsemble", num);
				break;
		}
	}
}

void WriteLevel(NFmiGribMessage& message, const level& lev)
{
	const long edition = message.Edition();

	// Himan levels equal to grib 1

	if (edition == 1)
	{
		message.LevelType(lev.Type());
	}
	else if (edition == 2)
	{
		if (lev.Type() == kHeightLayer)
		{
			message.LevelType(103);
			message.SetLongKey("typeOfSecondFixedSurface", 103);
		}
		else
		{
			message.LevelType(message.LevelTypeToAnotherEdition(lev.Type(), 2));
		}
	}

	switch (lev.Type())
	{
		case kHeightLayer:
		{
			message.LevelValue(static_cast<long>(0.01 * lev.Value()), 100);    // top
			message.LevelValue2(static_cast<long>(0.01 * lev.Value2()), 100);  // bottom
			break;
		}
		case kPressure:
		{
			// pressure in grib2 is pascals
			double scale = 1;
			if (edition == 2)
			{
				scale = 100;
			}

			message.LevelValue(static_cast<long>(lev.Value() * scale));
			break;
		}
		default:
			message.LevelValue(static_cast<long>(lev.Value()));
			break;
	}
}

void WriteForecastType(NFmiGribMessage& message, const forecast_type& forecastType, const producer& prod)
{
	// For grib1 this is simple: we only support analysis and deterministic forecasts
	// (when writing). Those do not need any extra metadata.

	if (message.Edition() == 1)
	{
		return;
	}

	// For grib2 there are several keys:
	// - type of data (https://apps.ecmwf.int/codes/grib/format/grib2/ctables/1/4)
	// - type of generating process (https://apps.ecmwf.int/codes/grib/format/grib2/ctables/4/3)
	// - product definition template number (https://apps.ecmwf.int/codes/grib/format/grib2/ctables/4/0)
	//
	// The key 'productDefinitionTemplateNumber' is set at WriteParameter()

	logger logr("grib");

	switch (forecastType.Type())
	{
		case kAnalysis:
			message.TypeOfGeneratingProcess(0);
			message.SetLongKey("typeOfProcessedData", 0);
			break;
		case kDeterministic:
			message.TypeOfGeneratingProcess(2);
			message.SetLongKey("typeOfProcessedData", 2);
			break;
		case kEpsControl:
		case kEpsPerturbation:
		{
			const long gribTOPD = (forecastType.Type() == kEpsControl ? 3 : 4);
			message.SetLongKey("typeOfProcessedData", gribTOPD);
			message.TypeOfGeneratingProcess(4);
			message.PerturbationNumber(static_cast<long>(forecastType.Value()));
			auto r = GET_PLUGIN(radon);

			try
			{
				const long ensembleSize = stol(r->RadonDB().GetProducerMetaData(prod.Id(), "ensemble size"));
				message.SetLongKey("numberOfForecastsInEnsemble", ensembleSize);
			}
			catch (const invalid_argument& e)
			{
				logr.Warning("Unable to get valid ensemble size information from radon for producer " +
				             to_string(prod.Id()));
			}
		}
		break;

		case kStatisticalProcessing:
			// "Post-processed forecast", one could consider everything produced by
			// Himan to be of this category but we only use this to represent statistical
			// post processing
			message.TypeOfGeneratingProcess(13);
			// Use locally reserved number because standard tables do not have a suitable
			// option
			message.SetLongKey("typeOfProcessedData", 192);
			break;

		default:
			logr.Warning("Unrecognized forecast type: " + static_cast<string>(forecastType));
			break;
	}
}

template <typename T>
void WriteDataValues(const vector<T>&, NFmiGribMessage&, double);

template <>
void WriteDataValues(const vector<double>& values, NFmiGribMessage& msg, double missingValue)
{
	msg.Values(values.data(), static_cast<long>(values.size()), missingValue);
}

template <>
void WriteDataValues(const vector<float>& values, NFmiGribMessage& msg, double missingValue)
{
	double* arr = new double[values.size()];
	replace_copy_if(values.begin(), values.end(), arr, [](const float& val) { return himan::IsMissing(val); },
	                missingValue);

	msg.Values(arr, static_cast<long>(values.size()), missingValue);

	delete[] arr;
}

std::pair<himan::HPWriteStatus, himan::file_information> grib::ToFile(info<double>& anInfo)
{
	return ToFile<double>(anInfo);
}

int DetermineCorrectGribEdition(int edition, const himan::forecast_type& ftype, const himan::forecast_time& ftime,
                                const himan::level& lvl, const himan::param& par)
{
	if (edition == 2)
	{
		// never switch from 2 to 1
		return 2;
	}

	// Check levelvalue, forecast type and param processing type since those might force us to change to grib2!

	using namespace himan;

	const bool lvlCondition = lvl.AB().size() > 255;
	const bool ftypeCondition = (ftype.Type() != kAnalysis && ftype.Type() != kDeterministic);
	const bool parCondition = par.ProcessingType().Type() != kUnknownProcessingType;
	const bool timeCondition = ftime.Step().Minutes() % 60 != 0;

	if (lvlCondition || ftypeCondition || parCondition || timeCondition)
	{
		himan::logger lgr("grib");
		lgr.Trace("File type forced to GRIB2 (level value: " + to_string(lvl.Value()) +
		          ", forecast type: " + HPForecastTypeToString.at(ftype.Type()) +
		          ", processing type: " + HPProcessingTypeToString.at(par.ProcessingType().Type()) +
		          " step: " + static_cast<string>(ftime.Step()) + ")");
		return 2;
	}

	return edition;
}

himan::forecast_type DetermineCorrectForecastType(const himan::forecast_type& ftype, const himan::param& par)
{
	// A kind of a workaround to make sure metadata in grib2 is correct when writing
	// statistically processed fields.

	using namespace himan;

	if (par.ProcessingType().Type() != kUnknownProcessingType && ftype.Type() != kStatisticalProcessing)
	{
		logger lgr("grib");
		lgr.Debug("Changing forecast type from " + static_cast<string>(ftype) + " to statistical processing");
		return forecast_type(kStatisticalProcessing);
	}

	return ftype;
}

template <typename T>
void WriteData(NFmiGribMessage& message, info<T>& anInfo, bool useBitmap, int precision)
{
	// set to missing value to a large value to prevent it from mixing up with valid
	// values in the data. eccodes does not support nan as missing value.

	const double gribMissing = 1e38;
	message.MissingValue(static_cast<T>(gribMissing));

	if (useBitmap && anInfo.Data().MissingCount() > 0)
	{
		message.Bitmap(true);
	}

	/*
	 * Possible precipitation form value encoding must be done before determining
	 * bits per value, as the range of values is changed.
	 */

	const auto paramName = anInfo.Param().Name();
	const int myprecision = (precision == kHPMissingInt) ? anInfo.Param().Precision() : precision;

	long bitsPerValue;

	if (message.Edition() == 2 && (paramName == "PRECFORM-N" || paramName == "PRECFORM2-N"))
	{
		// We take a copy of the data, because the values at cache should not change
		auto values = anInfo.Data().Values();

		EncodePrecipitationFormToGrib2(values);
		bitsPerValue = DetermineBitsPerValue(values, myprecision);

		// Change missing value 'nan' to a real fp value
		replace_if(values.begin(), values.end(), [](const T& v) { return IsMissing(v); }, static_cast<T>(gribMissing));

		message.BitsPerValue(bitsPerValue);
		WriteDataValues<T>(values, message, gribMissing);
	}
	else
	{
		// In this branch no copy is made
		const auto& values = anInfo.Data().Values();
		bitsPerValue = DetermineBitsPerValue(values, myprecision);

		// Change missing value 'nan' to a real fp value
		anInfo.Data().MissingValue(static_cast<T>(gribMissing));

		message.BitsPerValue(bitsPerValue);

		WriteDataValues<T>(values, message, gribMissing);
	}

	logger logr("grib");
	logr.Trace("Using " + (myprecision == kHPMissingInt ? "maximum precision" : to_string(myprecision) + " decimals") +
	           " (" + to_string(bitsPerValue) + " bits) to store " + paramName);

	// Return missing value to nan if info is recycled (luatool)
	anInfo.Data().MissingValue(MissingValue<T>());
}

template <typename T>
pair<himan::file_information, NFmiGribMessage> grib::CreateGribMessage(info<T>& anInfo)
{
	// Write only that data which is currently set at descriptors

	file_information finfo;
	finfo.file_location = util::MakeFileName(anInfo, *itsWriteOptions.configuration);
	finfo.storage_type = itsWriteOptions.configuration->WriteStorageType();

	long edition = DetermineCorrectGribEdition(static_cast<int>(itsWriteOptions.configuration->OutputFileType()),
	                                           anInfo.ForecastType(), anInfo.Time(), anInfo.Level(), anInfo.Param());

	finfo.file_type = static_cast<HPFileType>(edition);

	if (edition == 2 && itsWriteOptions.configuration->OutputFileType() == kGRIB1)
	{
		// backwards compatibility: previously grib version number was
		// not appended to filename when 'file_write' : 'single'
		if ((itsWriteOptions.configuration->LegacyWriteMode() == false ||
		     itsWriteOptions.configuration->WriteMode() != kAllGridsToAFile))
		{
			finfo.file_location += "2";
		}
	}

	if (itsWriteOptions.configuration->FileCompression() == kGZIP)
	{
		finfo.file_location += ".gz";
	}
	else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
	{
		finfo.file_location += ".bz2";
	}

	NFmiGribMessage message;
	message.Edition(edition);

	if (anInfo.Producer().Centre() == kHPMissingInt)
	{
		message.Centre(86);
		message.Process(255);
	}
	else
	{
		message.Centre(anInfo.Producer().Centre());
		message.Process(anInfo.Producer().Process());
	}

	// Parameter

	WriteParameter(message, anInfo.Param(), anInfo.Producer(), anInfo.ForecastType());

	// Forecast type

	WriteForecastType(message, DetermineCorrectForecastType(anInfo.ForecastType(), anInfo.Param()), anInfo.Producer());

	// Area and Grid

	WriteAreaAndGrid(message, anInfo.Grid(), anInfo.Producer());

	// Time information

	WriteTime(message, anInfo.Time(), anInfo.Producer(), anInfo.Param());

	// Level

	WriteLevel(message, anInfo.Level());

	// Set data to grib with correct precision and missing value

	WriteData<T>(message, anInfo, itsWriteOptions.use_bitmap, itsWriteOptions.precision);

	if (edition == 2)
	{
		switch (itsWriteOptions.configuration->PackingType())
		{
			case kJpegPacking:
				message.PackingType("grid_jpeg");
				break;
			case kCcsdsPacking:
				message.PackingType("grid_ccsds");
				break;
			default:
				break;
		}
	}
	/*
	 *  GRIB 1
	 *
	 * 	BIT	VALUE	MEANING
	 *	1	0		Direction increments not given
	 *	1	1		Direction increments given
	 *	2	0		Earth assumed spherical with radius = 6367.47 km
	 *	2	1		Earth assumed oblate spheroid with size as determined by IAU in 1965: 6378.160 km, 6356.775 km,
	 *f = 1/297.0 3-4	0		reserved (set to 0) 5	0		u- and v-components of vector quantities resolved
	 *relative to easterly and northerly directions 5	1		u and v components of vector quantities resolved
	 *relative to the defined grid in the direction of increasing x and y (or i and j) coordinates respectively
	 *	6-8	0		reserved (set to 0)
	 *
	 *	GRIB2
	 *
	 *	Bit No. 	Value 	Meaning
	 *	1-2			Reserved
	 *	3		0	i direction increments not given
	 *	3		1	i direction increments given
	 *	4		0	j direction increments not given
	 *	4		1	j direction increments given
	 *	5		0	Resolved u and v components of vector quantities relative to easterly and northerly directions
	 *	5		1	Resolved u and v components of vector quantities relative to the defined grid in the direction
	 *				of increasing x and y (or i and j) coordinates, respectively.
	 *	6-8			Reserved - set to zero
	 *
	 */

	if (anInfo.Grid()->Type() == kReducedGaussian)
	{
		message.ResolutionAndComponentFlags(0);
	}
	else
	{
		if (edition == 1)
		{
			message.ResolutionAndComponentFlags(128);  // 10000000
		}
		else
		{
			message.ResolutionAndComponentFlags(48);  // 00110000
		}

		if (anInfo.Grid()->UVRelativeToGrid())
		{
			message.UVRelativeToGrid(true);
		}
	}

	vector<double> AB = anInfo.Level().AB();

	if (!AB.empty())
	{
		message.NV(static_cast<long>(AB.size()));
		message.PV(AB, AB.size());
	}

	return make_pair(finfo, message);
}

template pair<himan::file_information, NFmiGribMessage> grib::CreateGribMessage<double>(info<double>&);
template pair<himan::file_information, NFmiGribMessage> grib::CreateGribMessage<float>(info<float>&);

void DetermineMessageNumber(NFmiGribMessage& message, file_information& finfo, HPWriteMode writeMode)
{
	// message length can only be received from eccodes since it includes
	// all grib headers etc
	finfo.length = message.GetLongKey("totalLength");

	// appending to a file is a serial operation -- two threads cannot
	// do it to a single file simultaneously. therefore offset is just
	// the size of the file so far.
	// because we don't want to actually fetch the file size every time
	// (too slow), store the size in a variable

	// handling message number is a bit tricker. fmigrib library cannot really
	// be used for tracking message number of written messages, because neither it
	// nor eccodes has any visibility to any possibly existing messages in a file
	// that is being appended to. therefore here we are tracking the message count
	// per file

	static std::map<std::string, unsigned long> offsets, messages;

	try
	{
		messages.at(finfo.file_location) = messages.at(finfo.file_location) + 1;
	}
	catch (const out_of_range& e)
	{
		if (boost::filesystem::exists(finfo.file_location) == false)
		{
			// file does not exist yet --> start counting from msg 0
			offsets[finfo.file_location] = 0;
			messages[finfo.file_location] = 0;
		}
		else
		{
			// file existed before Himan started --> count the messages from
			// the existing files and start numbering from there
			NFmiGrib rdr;
			rdr.Open(finfo.file_location);
			const int msgCount = rdr.MessageCount();

			if (msgCount == INVALID_INT_VALUE)
			{
				// HIMAN-319
				// Existing grib file is not finished correctly, it has 'GRIB'
				// but not '7777'
				//
				// What can we do now? We cannot append to this file because that will
				// cause problems to readers.
				//
				// 1. Abort and log this information
				//  + Not destructive
				//  - Needs manual intervention
				//  - Log line is easily lost as a lot of logging is produced
				//
				// 2. Rename invalid file to something else and continue
				//    with new file
				//  + No data is lost
				//  + Processing can continue without intervention
				//  - Old file is left as an 'orphan', it will not be cleaned
				//
				// 3. Remove invalid file
				//  + Clean solution
				//  + Processing can continue without intervention
				//  - Data is lost, could be very bad (if accidentally the filename is
				//    the same as some very important data file)
				//  - Possible existing memory mapping of file will cause signals
				//    to reading programs
				//
				// 4. Truncate file so that invalid message is removed
				//   + Clean solution
				//   + Processing can continue without intervention
				//   + Older messages are left intact and information is radon is accurate
				//   - Slow operation
				//
				// Choose option 4.

				logger logr("grib");
				logr.Warning(fmt::format("Found incomplete grib file '{}', truncating to last complete message",
				                         finfo.file_location));

				ifstream fp(finfo.file_location.c_str(), ios::in | ios::binary | ios::ate);
				ASSERT(fp);

				const long long origlen = fp.tellg();
				long long len = origlen;
				char buffer[8];

				for (long long i = 8; i <= origlen; i++)
				{
					fp.seekg(-i, fp.end);
					fp.read(buffer, 8);
					if (strncmp(buffer, "7777GRIB", 8) == 0)
					{
						break;
					}
					len = fp.tellg();
				}

				if (len <= 8)
				{
					logr.Error(fmt::format("Unable to truncate file '{}', remove it manually", finfo.file_location));
					himan::Abort();
				}

				len -= 4;
				boost::filesystem::resize_file(finfo.file_location, len);

				logr.Debug(fmt::format("Truncated file '{}' from {} to {} bytes", finfo.file_location, origlen, len));

				return DetermineMessageNumber(message, finfo, writeMode);
			}
			else
			{
				messages[finfo.file_location] = msgCount;
				offsets[finfo.file_location] = boost::filesystem::file_size(finfo.file_location);
			}
		}
	}

	finfo.offset = offsets.at(finfo.file_location);
	finfo.message_no = messages.at(finfo.file_location);

	offsets.at(finfo.file_location) = offsets.at(finfo.file_location) + finfo.length.get();
}

HPWriteStatus WriteMessageToFile(NFmiGribMessage& message, const file_information& finfo, const write_options& wopts)
{
	timer aTimer(true);
	logger logr("grib");

	bool appendToFile =
	    (wopts.configuration->WriteMode() == kAllGridsToAFile || wopts.configuration->WriteMode() == kFewGridsToAFile);

	if (appendToFile &&
	    (wopts.configuration->FileCompression() == kGZIP || wopts.configuration->FileCompression() == kBZIP2))
	{
		logr.Warning("Unable to write multiple grids to a packed file");
		appendToFile = false;
	}

	namespace fs = boost::filesystem;
	fs::path pathname(finfo.file_location);

	if (!pathname.parent_path().empty() && !fs::is_directory(pathname.parent_path()))
	{
		fs::create_directories(pathname.parent_path());
	}

	message.Write(finfo.file_location, appendToFile);

	aTimer.Stop();

	const float duration = static_cast<float>(aTimer.GetTime());
	const float bytes = static_cast<float>(finfo.length.get());  // TODO: does not work correctly if file is packed
	const float speed = (bytes / 1024.f / 1024.f) / (duration / 1000.f);

	stringstream ss;

	ss.precision((speed < 1) ? 1 : 0);

	string verb = (appendToFile ? "Appended to " : "Wrote ");

	ss << verb << "file '" << finfo.file_location << "' (" << fixed << speed << " MB/s)";
	logr.Info(ss.str());
	return HPWriteStatus::kFinished;
}

template <typename T>
std::pair<himan::HPWriteStatus, himan::file_information> grib::ToFile(info<T>& anInfo)
{
	if (anInfo.Grid()->Class() == kIrregularGrid && anInfo.Grid()->Type() != kReducedGaussian)
	{
		itsLogger.Error("Unable to write irregular grid of type " + HPGridTypeToString.at(anInfo.Grid()->Type()) +
		                " to grib");
		throw kInvalidWriteOptions;
	}

	if (itsWriteOptions.configuration->WriteStorageType() == kS3ObjectStorageSystem)
	{
		return make_pair(HPWriteStatus::kPending, file_information());
	}

	auto ret = CreateGribMessage<T>(anInfo);
	auto finfo = ret.first;
	auto msg = ret.second;

	HPWriteStatus status;

	if (itsWriteOptions.configuration->WriteMode() != kSingleGridToAFile)
	{
		pair<map<string, std::mutex>::iterator, bool> muret;

		// Acquire mutex to (possibly) modify map containing "file name":"mutex" pairs
		unique_lock<mutex> sflock(singleGribMessageCounterMutex);

		// create or refer to a mutex for this specific file name
		muret = singleGribMessageCounterMap.emplace(piecewise_construct, forward_as_tuple(finfo.file_location),
		                                            forward_as_tuple());
		// allow other threads to modify the map so as not to block threads writing
		// to other files
		sflock.unlock();
		// lock the mutex for this file name
		lock_guard<mutex> uniqueLock(muret.first->second);

		DetermineMessageNumber(msg, finfo, itsWriteOptions.configuration->WriteMode());
		status = WriteMessageToFile(msg, finfo, itsWriteOptions);
	}
	else
	{
		finfo.offset = 0;
		finfo.message_no = 0;
		finfo.length = msg.GetLongKey("totalLength");
		status = WriteMessageToFile(msg, finfo, itsWriteOptions);
	}

	return make_pair(status, finfo);
}

template std::pair<himan::HPWriteStatus, himan::file_information> grib::ToFile<double>(info<double>&);
template std::pair<himan::HPWriteStatus, himan::file_information> grib::ToFile<float>(info<float>&);

// ---------------------------------------------------------------------------

himan::earth_shape<double> ReadEarthShape(const NFmiGribMessage& msg)
{
	double a = himan::MissingDouble(), b = himan::MissingDouble();
	if (msg.Edition() == 1)
	{
		const long flag = msg.ResolutionAndComponentFlags();

		if (flag & (1 << 6))
		{
			// Earth assumed oblate spheroid with size as determined by IAU in 1965:
			// 6378.160 km, 6356.775 km, f = 1/297.0
			a = 6378160;
			b = 6356775;
		}
		else
		{
			// Earth assumed spherical with radius = 6367.47 km
			a = 6367470;
			b = 6367470;
		}
	}
	else
	{
		const long flag = msg.GetLongKey("shapeOfTheEarth");

		switch (flag)
		{
			// http://apps.ecmwf.int/codes/grib/format/grib2/ctables/3/2
			case 0:
				// Earth assumed spherical with radius = 6,367,470.0 m
				a = b = 6367470;
				break;
			case 1:
			{
				// Earth assumed spherical with radius specified (in m) by data producer
				const long scale = msg.GetLongKey("scaleFactorOfRadiusOfSphericalEarth");
				const long r = msg.GetLongKey("scaledValueOfRadiusOfSphericalEarth");
				a = b = static_cast<double>(scale * r);
				break;
			}
			case 2:
				// Earth assumed oblate spheroid with size as determined by IAU in 1965 (major axis = 6,378,160.0 m,
				// minor axis = 6,356,775.0 m, f = 1/297.0)
				a = 6378160;
				b = 6356775;
				break;
			case 3:
			{
				// Earth assumed oblate spheroid with major and minor axes specified (in km) by data producer
				long scale = msg.GetLongKey("scaleFactorOfEarthMajorAxis");
				long val = msg.GetLongKey("scaledValueOfEarthMajorAxis");
				a = static_cast<double>(1000 * scale * val);

				scale = msg.GetLongKey("scaleFactorOfEarthMinorAxis");
				val = msg.GetLongKey("scaledValueOfEarthMinorAxis");
				b = static_cast<double>(1000 * scale * val);
				break;
			}
			case 4:
				// Earth assumed oblate spheroid as defined in IAG-GRS80 model (major axis = 6,378,137.0 m, minor
				// axis = 6,356,752.314 m, f = 1/298.257222101)
				a = 6378137;
				b = 6356752.314;
				break;
			case 5:
				// Earth assumed represented by WGS84 (as used by ICAO since 1998)
				a = 6378137;
				b = 6356752.314245;
				break;
			case 6:
				// Earth assumed spherical with radius of 6,371,229.0 m
				a = b = 6371229;
				break;
			case 7:
			{
				// Earth assumed oblate spheroid with major and minor axes specified (in m) by data producer
				long scale = msg.GetLongKey("scaleFactorOfEarthMajorAxis");
				long val = msg.GetLongKey("scaledValueOfEarthMajorAxis");
				a = static_cast<double>(scale * val);

				scale = msg.GetLongKey("scaleFactorOfEarthMinorAxis");
				val = msg.GetLongKey("scaledValueOfEarthMinorAxis");
				b = static_cast<double>(scale * val);
				break;
			}
			case 8:
				// Earth model assumed spherical with radius 6371200 m, but the horizontal datum of the resulting
				// latitude/longitude field is the WGS84 reference frame
				a = b = 6371200;
				break;
			case 9:
				//  Earth represented by the Ordnance Survey Great Britain 1936 Datum, using the Airy 1830 Spheroid,
				//  the Greenwich meridian as 0 longitude, and the Newlyn datum as mean sea level, 0 height
				a = 6377563.396;
				b = 6356256.909;
				break;
			default:
			{
				himan::logger log("grib");
				log.Fatal("Unknown shape of earth in grib: " + to_string(flag));
				himan::Abort();
			}
		}
	}

	// Same hard coding here as in json_parser: first replace newbase area classes with gdal *and*
	// maintaing backwards compatibility, ie use the same values for earth radius as before.
	// The next phase is then to use the correct values and decide what to do with differing interpolation
	// results (newbase vs himan native).

	if (msg.NormalizedGridType() == 3)
	{
		a = b = 6367470.;
	}
	else
	{
		a = b = 6371220.;
	}

	return himan::earth_shape<double>(a, b);
}

unique_ptr<himan::grid> ReadAreaAndGrid(const NFmiGribMessage& message)
{
	bool iNegative = message.IScansNegatively();
	bool jPositive = message.JScansPositively();

	HPScanningMode m = kUnknownScanningMode;

	if (!iNegative && !jPositive)
	{
		m = kTopLeft;
	}
	else if (iNegative && !jPositive)
	{
		m = kTopRight;
	}
	else if (iNegative && jPositive)
	{
		m = kBottomRight;
	}
	else if (!iNegative && jPositive)
	{
		m = kBottomLeft;
	}
	else
	{
		throw runtime_error("WHAT?");
	}

	double X0 = message.X0();
	const double Y0 = message.Y0();

	// GRIB2 has longitude 0 .. 360, but in Himan we internally normalize it to -180 .. 180
	//
	// Make conversion to GRIB1 style coordinates, but in the long run we should figure out how to
	// handle grib 1 & grib 2 longitude values in a smart way. (a single geometry
	// can have coordinates in both ways!)

	if (X0 > 180)
		X0 -= 360.;
	const himan::point firstPoint(X0, Y0);

	unique_ptr<grid> newGrid;

	switch (message.NormalizedGridType())
	{
		case 0:
		{
			// clang-format off
			newGrid = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(
			    m,
			    firstPoint,
			    static_cast<size_t>(message.SizeX()),
			    static_cast<size_t>(message.SizeY()),
			    message.iDirectionIncrement(),
			    message.jDirectionIncrement(),
			    ReadEarthShape(message)
			));
			// clang-format on
			break;
		}
		case 3:
		{
			// clang-format off
			newGrid = unique_ptr<lambert_conformal_grid>(new lambert_conformal_grid(
			    m,
			    firstPoint,
			    message.SizeX(),
			    message.SizeY(),
			    message.XLengthInMeters(),
			    message.YLengthInMeters(),
			    message.GridOrientation(),
			    static_cast<double>(message.GetDoubleKey("Latin1InDegrees")),
			    static_cast<double>(message.GetDoubleKey("Latin2InDegrees")),
			    ReadEarthShape(message),
			    false
			));
			// clang-format off
			break;
		}

		case 4:
		{
			if (m == kTopLeft || m == kUnknownScanningMode)
			{
				newGrid = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
				reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(newGrid.get());

				gg->N(static_cast<int>(message.GetLongKey("N")));
				gg->NumberOfPointsAlongParallels(message.PL());
				gg->EarthShape(ReadEarthShape(message));

				break;
			}
			break;
		}

		case 5:
		{
			// clang-format off
			newGrid = unique_ptr<stereographic_grid>(new stereographic_grid(
			    m,
			    firstPoint,
			    message.SizeX(),
			    message.SizeY(),
			    message.XLengthInMeters(),
			    message.YLengthInMeters(),
			    message.GridOrientation(),
			    ReadEarthShape(message),
			    false
			));
			// clang-format off
			break;
		}

		case 10:
		{
			// clang-format off
			newGrid = unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid(
			    m,
			    firstPoint,
			    static_cast<size_t>(message.SizeX()),
			    static_cast<size_t>(message.SizeY()),
			    message.iDirectionIncrement(),
			    message.jDirectionIncrement(),
			    ReadEarthShape(message),
			    point(message.SouthPoleX(), message.SouthPoleY())
			));
			// clang-format on
			break;
		}
		default:
			logger logr("grib");
			logr.Fatal("Unsupported grid type: " + to_string(message.NormalizedGridType()));
			himan::Abort();
	}

	newGrid->UVRelativeToGrid(message.UVRelativeToGrid());

	return newGrid;
}

himan::param ReadParam(const search_options& options, const producer& prod, const NFmiGribMessage& message)
{
	param p;

	long number = message.ParameterNumber();

	shared_ptr<radon> r;

	auto dbtype = options.configuration->DatabaseType();
	logger logr("grib");

	if (message.Edition() == 1)
	{
		long no_vers = message.Table2Version();

		long timeRangeIndicator = message.TimeRangeIndicator();

		string parmName = "";
		int parmId = 0;

		if (dbtype == kRadon)
		{
			r = GET_PLUGIN(radon);

			auto parminfo = r->RadonDB().GetParameterFromGrib1(prod.Id(), no_vers, number, timeRangeIndicator,
			                                                   message.NormalizedLevelType(),
			                                                   static_cast<double>(message.LevelValue()));

			if (!parminfo.empty())
			{
				parmName = parminfo["name"];
				parmId = std::stoi(parminfo["id"]);
			}
		}

		if (parmName.empty() && dbtype == kNoDatabase)
		{
			parmName =
			    GetParamNameFromGribShortName(options.configuration->ParamFile(), message.GetStringKey("shortName"));
		}

		if (parmName.empty())
		{
			logr.Warning("Parameter name not found from " + HPDatabaseTypeToString.at(dbtype) +
			             " for producer: " + to_string(prod.Id()) + " no_vers: " + to_string(no_vers) +
			             ", number: " + to_string(number) + ", timeRangeIndicator: " + to_string(timeRangeIndicator));
			throw kFileMetaDataNotFound;
		}
		else
		{
			p.Name(parmName);
			p.Id(parmId);
		}

		p.GribParameter(number);
		p.GribTableVersion(no_vers);

		// Determine aggregation

		aggregation a;

		switch (timeRangeIndicator)
		{
			case 0:   // forecast
			case 1:   // analysis
			case 10:  // forecast with timestep > 255
				if (prod.Id() == 131)
				{
					// yeah, sometimes timeRangeIndicator=0 even if shortName=tp,
					// what can we do :shrug:
					a = util::GetAggregationFromParamName(p.Name(), forecast_time());

					if (a.Type() != kUnknownAggregationType)
					{
						// only P1 used in grib if tri=0/1
						a.TimeDuration(DurationFromTimeRange(message.UnitOfTimeRange()) *
						               static_cast<int>(message.P1()));
					}
				}
				break;

			case 2:  // typically max / min
			         // but which one?
			{
				if (parmName == "FFG-MS" || parmName == "FFG3H-MS" || parmName == "WGU-MS" || parmName == "WGV-MS" ||
				    parmName == "TMAX-K" || parmName == "TMAX12H-K")
				{
					a.Type(kMaximum);
					a.TimeDuration(DurationFromTimeRange(message.UnitOfTimeRange()) *
					               static_cast<int>(message.P2() - message.P1()));
				}
				else if (parmName == "TMIN-K" || parmName == "TMIN12H-K")
				{
					a.Type(kMinimum);
					a.TimeDuration(DurationFromTimeRange(message.UnitOfTimeRange()) *
					               static_cast<int>(message.P2() - message.P1()));
				}
			}
			break;

			case 3:  // average
				a.Type(kAverage);
				a.TimeDuration(DurationFromTimeRange(message.UnitOfTimeRange()) *
				               static_cast<int>(message.P2() - message.P1()));
				break;
			case 4:  // accumulation
				a.Type(kAccumulation);
				a.TimeDuration(DurationFromTimeRange(message.UnitOfTimeRange()) *
				               static_cast<int>(message.P2() - message.P1()));
				break;
		}

		if (a.Type() != kUnknownAggregationType)
		{
			p.Aggregation(a);
		}
	}
	else
	{
		aggregation a;
		processing_type pt;

		const long unitForTimeRange = message.GetLongKey("indicatorOfUnitForTimeRange");
		const long category = message.ParameterCategory();
		const long discipline = message.ParameterDiscipline();
		const long tosp = (message.TypeOfStatisticalProcessing() == -999) ? -1 : message.TypeOfStatisticalProcessing();

		// If there is no time aggregation, set td values to "not_a_time_duration" to indicate
		// that there is no time aggregation (aggregation is done in some other dimension)
		const auto td = (message.LengthOfTimeRange() == 0)
		                    ? himan::time_duration()
		                    : DurationFromTimeRange(unitForTimeRange) * static_cast<int>(message.LengthOfTimeRange());

		// Type of statistical processign key is defined only for
		// parameters "in a continuous or non-continuous time interval"
		switch (tosp)
		{
			case 0:  // Average
				a.Type(kAverage);
				a.TimeDuration(td);
				break;

			case 1:  // Accumulation
				a.Type(kAccumulation);
				a.TimeDuration(td);
				break;

			case 2:  // Maximum
				a.Type(kMaximum);
				a.TimeDuration(td);
				break;

			case 3:  // Minimum
				a.Type(kMinimum);
				a.TimeDuration(td);
				break;
			case 6:  // Standard deviation
				pt.Type(kStandardDeviation);
				break;
		}

		// "derivedForecasts" is defined also for
		// "at a point in time"
		const long df = message.GetLongKey("derivedForecast");

		switch (df)
		{
			case 0:  // Unweighted Mean of All Members
			case 1:  // Weighted Mean of All Members
				pt.Type(kEnsembleMean);
				break;
			case 2:  // Standard Deviation with respect to Cluster Mean
				pt.Type(kStandardDeviation);
				break;
			case 4:  // Spread of All Members
				pt.Type(kSpread);
				break;
			case 199:  // Extreme Forecast Index
				pt.Type(kEFI);
				break;
		}

		string parmName = "";
		int parmId = 0;

		if (dbtype == kRadon)
		{
			r = GET_PLUGIN(radon);

			// Because database table param_grib2 has only column 'type_of_statistical_processing',
			// we have to use grib2 key 'derivedForecast' to fake the value. When grib2 has
			// productDefinitionTemplateNumber, the statistical processing is not given with
			// 'typeOfStatisticalProcessing' but with 'derivedForecast'.

			long effective_tosp = tosp;

			if (tosp == -1)
			{
				switch (pt.Type())
				{
					case kEnsembleMean:
						effective_tosp = 0;
						break;
					case kStandardDeviation:
						effective_tosp = 6;
						break;
					default:
						break;
				}
			}

			auto parminfo =
			    r->RadonDB().GetParameterFromGrib2(prod.Id(), discipline, category, number, message.LevelType(),
			                                       static_cast<double>(message.LevelValue()), effective_tosp);

			if (parminfo.size())
			{
				parmName = parminfo["name"];
				parmId = std::stoi(parminfo["id"]);
			}
		}

		if (parmName.empty() && dbtype == kNoDatabase)
		{
			parmName =
			    GetParamNameFromGribShortName(options.configuration->ParamFile(), message.GetStringKey("shortName"));
		}

		if (parmName.empty())
		{
			logr.Warning("Parameter name not found from database for producer: " + to_string(prod.Id()) +
			             " discipline: " + to_string(discipline) + ", category: " + to_string(category) +
			             ", number: " + to_string(number) + ", statistical processing: " + to_string(tosp));
			throw kFileMetaDataNotFound;
		}
		else
		{
			p.Name(parmName);
			p.Id(parmId);
		}

		p.GribParameter(number);
		p.GribDiscipline(discipline);
		p.GribCategory(category);

		if (a.TimeDuration().Empty() == false)
		{
			p.Aggregation(a);
		}
		if (pt.Type() != kUnknownProcessingType)
		{
			const int numMemb = static_cast<int>(message.GetLongKey("numberOfForecastsInEnsemble"));
			pt.NumberOfEnsembleMembers(numMemb == INVALID_INT_VALUE ? 255 : numMemb);
			p.ProcessingType(pt);
		}
	}

	string unit = message.ParameterUnit();

	if (unit == "K")
	{
		p.Unit(kK);
	}
	else if (unit == "Pa s-1" || unit == "Pa s**-1")
	{
		p.Unit(kPas);
	}
	else if (unit == "%")
	{
		p.Unit(kPrcnt);
	}
	else if (unit == "m s**-1" || unit == "m s-1")
	{
		p.Unit(kMs);
	}
	else if (unit == "m" || unit == "m of water equivalent")
	{
		p.Unit(kM);
	}
	else if (unit == "mm")
	{
		p.Unit(kMm);
	}
	else if (unit == "Pa")
	{
		p.Unit(kPa);
	}
	else if (unit == "m**2 s**-2")
	{
		p.Unit(kGph);
	}
	else if (unit == "kg kg**-1")
	{
		p.Unit(kKgkg);
	}
	else if (unit == "J m**-2")
	{
		p.Unit(kJm2);
	}
	else if (unit == "kg m**-2")
	{
		p.Unit(kKgm2);
	}
	else if (unit == "hPa")
	{
		p.Unit(kHPa);
	}
	else
	{
		logr.Trace("Unable to determine himan parameter unit for grib unit " + message.ParameterUnit());
	}

	p.InterpolationMethod(options.param.InterpolationMethod());

	return p;
}

himan::forecast_time ReadTime(const NFmiGribMessage& message)
{
	string dataDate = to_string(message.DataDate());

	/*
	 * dataTime is HH24MM in long datatype.
	 * So, for example analysistime 00 is 0, and 06 is 600.
	 */

	long dt = message.DataTime();
	char fmt[5];
	snprintf(fmt, 5, "%04ld", dt);

	long step = message.NormalizedStep(true, true);

	string originDateTimeStr = dataDate + string(fmt);
	raw_time originDateTime(originDateTimeStr, "%Y%m%d%H%M");

	forecast_time t(originDateTime, originDateTime);

	long unitOfTimeRange = message.NormalizedUnitOfTimeRange();

	HPTimeResolution timeResolution = kUnknownTimeResolution;

	switch (unitOfTimeRange)
	{
		case 1:
		case 10:
		case 11:
		case 12:
			timeResolution = kHourResolution;
			break;

		case 0:
		case 13:
		case 14:
		case 254:
			timeResolution = kMinuteResolution;
			break;

		default:
			logger logr("grib");
			logr.Warning("Unsupported unit of time range: " + to_string(unitOfTimeRange));
			break;
	}

	t.ValidDateTime().Adjust(timeResolution, static_cast<int>(step));

	return t;
}

himan::level ReadLevel(const search_options& opts, const producer& prod, const NFmiGribMessage& message)
{
	himan::HPLevelType levelType = kUnknownLevel;
	logger logr("grib");

	if (opts.configuration->DatabaseType() == kNoDatabase)
	{
		// Minimal set of levels for those who might try to run himan
		// without a database connection
		const long gribLevel = message.NormalizedLevelType();

		switch (gribLevel)
		{
			case 1:
				levelType = himan::kGround;
				break;
			case 100:
				levelType = himan::kPressure;
				break;
			case 105:
				levelType = himan::kHeight;
				break;
			case 109:
				levelType = himan::kHybrid;
				break;
			default:
				logr.Fatal("Unsupported level type for no database mode: " + to_string(gribLevel));
				himan::Abort();
		}
	}
	else
	{
		long gribLevelType = message.LevelType();
		const long edition = message.Edition();

		if (edition == 2)
		{
			// Special cases checked *before* checking database, because we possibly change
			// level numbers here.

			// 1. In grib2 also typeOfSecondFixedSurface is set. Radon database does
			// not support this currently, these files seem to be very rare.

			const long gribLevelType2 = message.GetLongKey("typeOfSecondFixedSurface");

			if (gribLevelType == 1 && gribLevelType2 == 8)
			{
				// In this particular case levels GROUND and TOP are defined
				// Change the level to ENTATM. Note: metadata is only changed
				// in database, file metadata is left the same. This means
				// that if reading program validates metadata, it will notice
				// the difference.
				// This behavior was inherited from grid_to_radon program.

				gribLevelType = 10;
			}
		}

		auto r = GET_PLUGIN(radon);

		auto levelInfo = r->RadonDB().GetLevelFromGrib(prod.Id(), gribLevelType, message.Edition());

		if (levelInfo.empty())
		{
			logr.Error("Unsupported level type for producer " + to_string(prod.Id()) + ": " + to_string(gribLevelType) +
			           ", grib edition " + to_string(message.Edition()));
			throw kFileMetaDataNotFound;
		}

		string levelName = levelInfo["name"];
		boost::algorithm::to_lower(levelName);

		if (edition == 2)
		{
			// Special cases checked *after* checking database.

			// 1. Check if we have a height_layer, which in grib2 is first and second leveltype 103.

			const long gribLevelType2 = message.GetLongKey("typeOfSecondFixedSurface");

			if (gribLevelType == 103 && gribLevelType2 == 103)
			{
				const long levelValue2 = message.LevelValue2();

				if (levelValue2 != -999 && levelValue2 != 214748364700)
				{
					levelName = "height_layer";
				}
			}
		}

		levelType = HPStringToLevelType.at(levelName);
	}

	himan::level l;

	switch (levelType)
	{
		case himan::kHeightLayer:
			l = level(levelType, 100 * static_cast<double>(message.LevelValue()),
			          100 * static_cast<double>(message.LevelValue2()));
			break;

		case himan::kGroundDepth:
		case himan::kPressureDelta:
		{
			long gribLevelValue2 = message.LevelValue2();
			// Missing in grib is all bits set
			if (gribLevelValue2 == 2147483647)
			{
				gribLevelValue2 = -1;
			}

			l = level(levelType, static_cast<float>(message.LevelValue()), static_cast<float>(gribLevelValue2));
		}
		break;

		default:
			l = level(levelType, static_cast<float>(message.LevelValue()));
			break;
	}

	return l;
}

himan::producer ReadProducer(const search_options& options, const NFmiGribMessage& message)
{
	long centre = message.Centre();
	long process = message.Process();

	producer prod(centre, process);

	logger logr("grib");

	if (options.configuration->DatabaseType() == kRadon)
	{
		// Do a double check and fetch the fmi producer id from database.

		auto r = GET_PLUGIN(radon);

		if (centre == 98)
		{
			// legacy: for ECMWF must still separate between different produces
			// based on forecast type
			// future goal: forecast type is not a producer property
			long typeId = 1;  // deterministic forecast, default
			long msgType = message.ForecastType();

			if (msgType == 2)
			{
				typeId = 2;  // ANALYSIS
			}
			else if (msgType == 3 || msgType == 4)
			{
				typeId = 3;  // ENSEMBLE
			}

			auto prodInfo = r->RadonDB().GetProducerFromGrib(centre, process, typeId);

			if (!prodInfo.empty())
			{
				prod.Id(stoi(prodInfo["id"]));
				return prod;
			}

			if (process <= 151 && process >= 142)
			{
				if (message.ForecastType() <= 2)
				{
					prod.Id(131);
				}
				else if (message.ForecastType() >= 3)
				{
					prod.Id(134);
				}
			}
			else
			{
				logr.Warning("Producer information not found from database for centre " + to_string(centre) +
				             ", process " + to_string(process));
			}
		}
		else if (centre == 251 && process == 40)
		{
			// support old (40) and new (0) number for MEPS
			prod.Id(4);
		}
		else
		{
			auto prodInfo = r->RadonDB().GetProducerFromGrib(centre, process);
			if (prodInfo.empty())
			{
				logr.Warning("Producer information not found from database for centre " + to_string(centre) +
				             ", process " + to_string(process));
			}
			else if (prodInfo.size() >= 1)
			{
				prod.Id(stoi(prodInfo[0]["id"]));
				if (prodInfo.size() > 1)
				{
					logr.Warning("More than producer definition found from radon for centre " + to_string(centre) +
					             ", process " + to_string(process) + ", selecting first one=" + prodInfo[0]["Id"]);
				}
			}
		}
	}

	return prod;
}

template <typename T>
void ReadDataValues(vector<T>&, const NFmiGribMessage& msg);

template <>
void ReadDataValues(vector<double>& values, const NFmiGribMessage& msg)
{
	size_t len = msg.ValuesLength();
	msg.GetValues(values.data(), &len, himan::MissingDouble());
}

template <>
void ReadDataValues(vector<float>& values, const NFmiGribMessage& msg)
{
	double* arr = new double[values.size()];
	size_t len = msg.ValuesLength();
	msg.GetValues(arr, &len, himan::MissingDouble());

	replace_copy_if(arr, arr + values.size(), values.begin(), [](const double& val) { return himan::IsMissing(val); },
	                himan::MissingFloat());

	delete[] arr;
}

template <typename T>
void ReadData(shared_ptr<info<T>> newInfo, bool readPackedData, const NFmiGribMessage& message)
{
	auto& dm = newInfo->Data();

	bool decodePrecipitationForm = false;

	logger logr("grib");

#if defined GRIB_READ_PACKED_DATA && defined HAVE_CUDA

	const auto paramName = newInfo->Param().Name();
	long producerId = newInfo->Producer().Id();

	if (message.Edition() == 2 && (paramName == "PRECFORM-N" || paramName == "PRECFORM2-N") &&
	    (producerId == 230 || producerId == 240 || producerId == 243 || producerId == 250 || producerId == 260 ||
	     producerId == 265 || producerId == 270))
	{
		decodePrecipitationForm = true;
	}

	if (readPackedData && decodePrecipitationForm == false && message.PackingType() == "grid_simple")
	{
		// Get coefficient information

		double bsf = static_cast<double>(message.BinaryScaleFactor());
		double dsf = static_cast<double>(message.DecimalScaleFactor());
		double rv = message.ReferenceValue();
		int bpv = static_cast<int>(message.BitsPerValue());

		auto packed = make_shared<simple_packed>(bpv, util::ToPower(bsf, 2), util::ToPower(-dsf, 10), rv);

		// Get packed values from grib

		size_t len = message.PackedValuesLength();
		int* unpackedBitmap = 0;

		packed->unpackedLength = message.SizeX() * message.SizeY();

		if (len > 0)
		{
			ASSERT(packed->data == 0);
			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&packed->data), len * sizeof(unsigned char)));

			message.PackedValues(packed->data);
			packed->packedLength = len;

			logr.Trace("Retrieved " + to_string(len) + " bytes of packed data from grib");
		}
		else
		{
			logr.Trace("Grid is constant or empty");
		}

		if (message.Bitmap())
		{
			size_t bitmap_len = message.BytesLength("bitmap");
			size_t bitmap_size = static_cast<size_t>(ceil(static_cast<double>(bitmap_len) / 8));

			logr.Trace("Grib has bitmap, length " + to_string(bitmap_len) + " size " + to_string(bitmap_size) +
			           " bytes");

			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&unpackedBitmap), bitmap_len * sizeof(int)));

			unsigned char* bitmap = new unsigned char[bitmap_size];

			message.Bytes("bitmap", bitmap);

			UnpackBitmap(bitmap, unpackedBitmap, bitmap_size, bitmap_len);

			packed->bitmap = unpackedBitmap;
			packed->bitmapLength = bitmap_len;

			delete[] bitmap;
		}
		auto b = newInfo->Base();
		b->pdata = move(packed);
	}
	else
#endif
	{
		ReadDataValues<T>(dm.Values(), message);

		if (decodePrecipitationForm)
		{
			logr.Trace("Decoding precipitation form grib2 values");
			DecodePrecipitationFormFromGrib2(dm.Values());
		}

		logr.Trace("Retrieved " + std::to_string(dm.Size() * sizeof(T)) + " bytes of unpacked data from grib");
	}
}

template <typename T>
bool grib::CreateInfoFromGrib(const search_options& options, bool readPackedData, bool readIfNotMatching,
                              shared_ptr<info<T>> newInfo, const NFmiGribMessage& message, bool readData) const
{
	bool dataIsValid = true;

	auto prod = ReadProducer(options, message);

	if (options.prod.Process() != prod.Process() || options.prod.Centre() != prod.Centre())
	{
		if (!readIfNotMatching)
		{
			itsLogger.Trace("centre/process do not match: " + to_string(options.prod.Process()) + " vs " +
			                to_string(prod.Process()));
			itsLogger.Trace("centre/process do not match: " + to_string(options.prod.Centre()) + " vs " +
			                to_string(prod.Centre()));
		}
	}

	auto p = ReadParam(options, prod, message);

	if (p != options.param)
	{
		if (readIfNotMatching)
		{
			dataIsValid = false;
		}
		else
		{
			itsLogger.Trace("Parameter does not match: " + options.param.Name() + " (requested) vs " + p.Name() +
			                " (found)");
			itsLogger.Trace("Aggregation: " + static_cast<string>(options.param.Aggregation()) + " (requested) vs " +
			                static_cast<string>(p.Aggregation()) + " (found)");
			itsLogger.Trace("Processing type: " + static_cast<string>(options.param.ProcessingType()) +
			                " (requested) vs " + static_cast<string>(p.ProcessingType()) + " (found)");
			return false;
		}
	}

	auto t = ReadTime(message);

	if (t != options.time)
	{
		if (readIfNotMatching)
		{
			dataIsValid = false;
		}
		else
		{
			forecast_time optsTime(options.time);

			itsLogger.Trace("Times do not match");

			if (optsTime.OriginDateTime() != t.OriginDateTime())
			{
				itsLogger.Trace("OriginDateTime: " + optsTime.OriginDateTime().String() + " (requested) vs " +
				                t.OriginDateTime().String() + " (found)");
			}

			if (optsTime.ValidDateTime() != t.ValidDateTime())
			{
				itsLogger.Trace("ValidDateTime: " + optsTime.ValidDateTime().String() + " (requested) vs " +
				                t.ValidDateTime().String() + " (found)");
			}

			return false;
		}
	}

	auto l = ReadLevel(options, prod, message);

	if (l != options.level)
	{
		if (readIfNotMatching)
		{
			dataIsValid = false;
		}
		else
		{
			itsLogger.Trace("Level does not match");
			itsLogger.Trace(static_cast<string>(options.level) + " vs " + static_cast<string>(l));

			return false;
		}
	}

	forecast_type ty(static_cast<HPForecastType>(message.ForecastType()),
	                 static_cast<double>(message.ForecastTypeValue()));

	if (options.ftype.Type() != ty.Type() || options.ftype.Value() != ty.Value())
	{
		if (readIfNotMatching)
		{
			dataIsValid = false;
		}
		else
		{
			itsLogger.Trace("Forecast type does not match");
			itsLogger.Trace(static_cast<string>(options.ftype) + " vs " + static_cast<string>(ty));

			return false;
		}
	}

	// END VALIDATION OF SEARCH PARAMETERS

	newInfo->Producer(prod);

	std::vector<double> ab;

	if (l.Type() == himan::kHybrid)
	{
		long nv = message.NV();

		if (nv > 0)
		{
			ab = message.PV();
		}
	}

	l.AB(ab);

	newInfo->template Set<param>({p});
	newInfo->template Set<forecast_time>({t});
	newInfo->template Set<level>({l});
	newInfo->template Set<forecast_type>({ty});

	unique_ptr<grid> newGrid = ReadAreaAndGrid(message);

	ASSERT(newGrid);

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(newGrid->Clone());

	newInfo->Create(b, readData);

	// Set descriptors

	newInfo->template Find<param>(p);
	newInfo->template Find<forecast_time>(t);
	newInfo->template Find<level>(l);
	newInfo->template Find<forecast_type>(ty);

	if (readData == false)
	{
		return true;
	}

	/*
	 * Read data from grib. If interpolation is required, it's better to do the unpacking
	 * at host to avoid unnecessary copying between CPU and GPU.
	 */

	ReadData(newInfo, readPackedData && (*options.configuration->BaseGrid() == *newInfo->Grid()), message);

	if (!dataIsValid)
	{
		return false;
	}

	newInfo->First();
	return true;
}

template bool grib::CreateInfoFromGrib<double>(const search_options&, bool, bool, shared_ptr<info<double>>,
                                               const NFmiGribMessage&, bool) const;

vector<shared_ptr<himan::info<double>>> grib::FromFile(const file_information& theInputFile,
                                                       const search_options& options, bool readPackedData,
                                                       bool readIfNotMatching) const
{
	return FromFile<double>(theInputFile, options, readPackedData, readIfNotMatching);
}

template <typename T>
vector<shared_ptr<himan::info<T>>> grib::FromFile(const file_information& theInputFile, const search_options& options,
                                                  bool readPackedData, bool readIfNotMatching) const
{
	vector<shared_ptr<himan::info<T>>> infos;

	if (options.prod.Centre() == kHPMissingInt && options.configuration->DatabaseType() != kNoDatabase)
	{
		itsLogger.Error("Process and centre information for producer " + to_string(options.prod.Id()) +
		                " are undefined");
		return infos;
	}

	timer aTimer(true);
	NFmiGrib reader;

	if (readIfNotMatching || !theInputFile.offset)
	{
		// read all messages from local 'auxiliary' file
		if (!reader.Open(theInputFile.file_location))
		{
			itsLogger.Error("Opening file '" + theInputFile.file_location + "' failed");
			return infos;
		}

		while (reader.NextMessage())
		{
			auto newInfo = make_shared<info<T>>();
			if (CreateInfoFromGrib(options, readPackedData, readIfNotMatching, newInfo, reader.Message()) ||
			    readIfNotMatching)
			{
				infos.push_back(newInfo);
				newInfo->First();
			}
		}
	}
	else
	{
		file_accessor fa;
		const buffer buf = fa.Read(theInputFile);

		if (!reader.ReadMessage(buf.data, buf.length))
		{
			itsLogger.Error("Creating GRIB message from memory failed");
			return infos;
		}

		auto newInfo = make_shared<info<T>>();

		if (CreateInfoFromGrib(options, readPackedData, false, newInfo, reader.Message()))
		{
			infos.push_back(newInfo);
			newInfo->First();
		}
	}

	aTimer.Stop();

	const long duration = aTimer.GetTime();
	const auto bytes =
	    (theInputFile.length) ? theInputFile.length.get() : boost::filesystem::file_size(theInputFile.file_location);
	const float speed = (static_cast<float>(bytes) / 1024.f / 1024.f) / (static_cast<float>(duration) / 1000.f);

	stringstream ss;
	ss.precision((speed < 1.) ? 1 : 0);

	ss << "Read from file '" << theInputFile.file_location << "' ";

	if (theInputFile.offset)
	{
		ss << "position " << theInputFile.offset.get() << ":" << bytes << " msg# " << theInputFile.message_no.get();
	}

	ss << " (" << fixed << speed << " MB/s)";

	itsLogger.Debug(ss.str());

	return infos;
}

template vector<shared_ptr<himan::info<double>>> grib::FromFile<double>(const file_information&, const search_options&,
                                                                        bool, bool) const;
template vector<shared_ptr<himan::info<float>>> grib::FromFile<float>(const file_information&, const search_options&,
                                                                      bool, bool) const;
/**
 * @brief UnpackBitmap
 *
 * Transform regular bitmap (unsigned char) to a int-based bitmap where each array key represents
 * an actual data value. If bitmap is zero for that key, zero is also put to the int array. If bitmap
 * is set for that key, the value is one.
 *
 * TODO: Change int-array to unpacked unsigned char array (reducing size 75%) or even not unpack bitmap beforehand
 * but do it
 * while computing stuff with the data array.
 *
 * @param bitmap Original bitmap read from grib
 * @param unpacked Unpacked bitmap where number of keys is the same as in the data array
 * @param len Length of original bitmap
 */

void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len, size_t unpackedLen)
{
	size_t i, idx = 0;
	int v = 1;

	for (i = 0; i < len; i++)
	{
		for (short j = 7; j >= 0; j--)
		{
			if (BitTest(bitmap[i], j))
			{
				unpacked[idx] = v++;
			}
			else
			{
				unpacked[idx] = 0;
			}

			if (++idx >= unpackedLen)
			{
				// packed data might not be aligned nicely along byte boundaries --
				// need to break from loop after final element has been processed
				break;
			}
		}
	}
}

std::string GetParamNameFromGribShortName(const std::string& paramFileName, const std::string& shortName)
{
	ifstream paramFile;
	paramFile.open(paramFileName, ios::in);

	if (!paramFile.is_open())
	{
		throw runtime_error("Unable to open file '" + paramFileName + "'");
	}

	string line, ret;

	while (getline(paramFile, line))
	{
		auto elems = himan::util::Split(line, ",");

		if (elems.size() == 2)
		{
			if (elems[0] == shortName)
			{
				ret = elems[1];
				break;
			}
		}
#ifdef DEBUG
		else
		{
			cout << "paramFile invalid line: '" << line << "'\n";
		}
#endif
	}

	paramFile.close();

	return ret;
}
