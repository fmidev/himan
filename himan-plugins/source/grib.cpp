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
#include "stereographic_grid.h"
#include "timer.h"
#include "util.h"
#include <algorithm>
#include <boost/filesystem.hpp>

using namespace std;
using namespace himan::plugin;

#include "radon.h"

#include "cuda_helper.h"
#include "packed_data.h"

#define BitMask1(i) (1u << i)
#define BitTest(n, i) !!((n)&BitMask1(i))

std::string GetParamNameFromGribShortName(const std::string& paramFileName, const std::string& shortName);

const double gribMissing = 32700.;

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
				templateNumber =
				    4;  // Derived forecasts based on a cluster of ensemble members over a circular area at a
				        // horizontal level or in a horizontal layer at a point in time.
				break;
			case kProbabilityGreaterThan:
			case kProbabilityLessThan:
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
long DetermineBitsPerValue(const vector<T>& values, double precision)
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
		log.Error("bits per value calculation failed, defaulting to 24");
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
			throw runtime_error("Unsupported unit of time range: " + to_string(unitOfTimeRange));
	}
}

grib::grib()
{
	itsLogger = logger("grib");

	itsGrib = make_shared<NFmiGrib>();
}

shared_ptr<NFmiGrib> grib::Reader()
{
	return itsGrib;
}

void grib::WriteAreaAndGrid(const shared_ptr<himan::grid>& grid, const producer& prod)
{
	const long edition = itsGrib->Message().Edition();
	HPScanningMode scmode = kUnknownScanningMode;

	auto firstGridPoint = grid->FirstPoint();

	if (edition == 2)
	{
		if (firstGridPoint.X() < 0)
		{
			firstGridPoint.X(firstGridPoint.X() + 360.);
		}
	}

	// UVRelativeToGrid is set in ToFile()

	switch (grid->Type())
	{
		case kLatitudeLongitude:
		{
			auto rg = dynamic_pointer_cast<latitude_longitude_grid>(grid);

			himan::point lastGridPoint = rg->LastPoint();

			long gridType = 0;  // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(firstGridPoint.X());
			itsGrib->Message().X1(lastGridPoint.X());
			itsGrib->Message().Y0(firstGridPoint.Y());
			itsGrib->Message().Y1(lastGridPoint.Y());

			itsGrib->Message().iDirectionIncrement(rg->Di());
			itsGrib->Message().jDirectionIncrement(rg->Dj());

			itsGrib->Message().SizeX(static_cast<long>(rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			break;
		}

		case kRotatedLatitudeLongitude:
		{
			auto rg = dynamic_pointer_cast<rotated_latitude_longitude_grid>(grid);

			himan::point lastGridPoint = rg->LastPoint();

			long gridType = 10;  // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(firstGridPoint.X());
			itsGrib->Message().Y0(firstGridPoint.Y());
			itsGrib->Message().X1(lastGridPoint.X());
			itsGrib->Message().Y1(lastGridPoint.Y());

			itsGrib->Message().SouthPoleX(rg->SouthPole().X());
			itsGrib->Message().SouthPoleY(rg->SouthPole().Y());

			itsGrib->Message().iDirectionIncrement(rg->Di());
			itsGrib->Message().jDirectionIncrement(rg->Dj());

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().SizeX(static_cast<long>(rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			break;
		}

		case kStereographic:
		{
			auto rg = dynamic_pointer_cast<stereographic_grid>(grid);

			long gridType = 5;  // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(firstGridPoint.X());
			itsGrib->Message().Y0(firstGridPoint.Y());

			itsGrib->Message().GridOrientation(rg->Orientation());

			itsGrib->Message().XLengthInMeters(rg->Di());
			itsGrib->Message().YLengthInMeters(rg->Dj());

			itsGrib->Message().SizeX(static_cast<long>(rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			if (edition == 2)
			{
				itsGrib->Message().SetLongKey("LaDInDegrees", 60);
			}

			break;
		}

		case kReducedGaussian:
		{
			auto gg = dynamic_pointer_cast<reduced_gaussian_grid>(grid);

			long gridType = 4;  // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			const double lonMin = firstGridPoint.X();
			const double lonMax = gg->LatLon(gg->NumberOfPointsAlongParallels()[gg->N()], gg->N()).X();
			const double latMin = gg->Latitudes().back();
			const double latMax = gg->Latitudes().front();

			itsGrib->Message().X0(lonMin);
			itsGrib->Message().Y0(latMax);
			itsGrib->Message().X1(lonMax);
			itsGrib->Message().Y1(latMin);

			itsGrib->Message().SetLongKey("iDirectionIncrement", 65535);
			itsGrib->Message().SetLongKey("numberOfPointsAlongAParallel", 65535);

			itsGrib->Message().SetLongKey("N", static_cast<long>(gg->N()));

			itsGrib->Message().PL(gg->NumberOfPointsAlongParallels());

			scmode = kTopLeft;

			break;
		}

		case kLambertConformalConic:
		{
			auto lccg = dynamic_pointer_cast<lambert_conformal_grid>(grid);

			long gridType = 3;  // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(firstGridPoint.X());
			itsGrib->Message().Y0(firstGridPoint.Y());

			itsGrib->Message().GridOrientation(lccg->Orientation());

			itsGrib->Message().XLengthInMeters(lccg->Di());
			itsGrib->Message().YLengthInMeters(lccg->Dj());

			itsGrib->Message().SizeX(static_cast<long>(lccg->Ni()));
			itsGrib->Message().SizeY(static_cast<long>(lccg->Nj()));

			itsGrib->Message().SetLongKey("Latin1InDegrees", static_cast<long>(lccg->StandardParallel1()));

			if (!IsKHPMissingValue(lccg->StandardParallel2()))
			{
				itsGrib->Message().SetLongKey("Latin2InDegrees", static_cast<long>(lccg->StandardParallel2()));
			}

			scmode = lccg->ScanningMode();

			if (edition == 2)
			{
				itsGrib->Message().SetLongKey("LaDInDegrees", 60);
			}

			break;
		}

		default:
			itsLogger.Fatal("Invalid projection while writing grib: " + to_string(grid->Type()));
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
			itsGrib->Message().SetLongKey("earthIsOblate", 0);

			long flag = itsGrib->Message().ResolutionAndComponentFlags();

			flag &= ~(1UL << 6);

			itsGrib->Message().ResolutionAndComponentFlags(flag);
		}
		else
		{
			if (a == 6367470)
			{
				itsGrib->Message().SetLongKey("shapeOfTheEarth", 0);
			}
			else
			{
				itsGrib->Message().SetLongKey("shapeOfTheEarth", 1);
				itsGrib->Message().SetLongKey("scaleFactorOfRadiusOfSphericalEarth", 1);
				itsGrib->Message().SetLongKey("scaledValueOfRadiusOfSphericalEarth", a);
			}
		}
	}
	else
	{
		itsLogger.Fatal("A spheroid, really?");
		himan::Abort();
	}
#endif
	itsGrib->Message().Centre(prod.Centre() == kHPMissingInt ? 86 : prod.Centre());
	itsGrib->Message().Process(prod.Process() == kHPMissingInt ? 255 : prod.Process());

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
			throw runtime_error(ClassName() + ": Unknown scanning mode when writing grib");
			break;
	}

	itsGrib->Message().IScansNegatively(iNegative);
	itsGrib->Message().JScansPositively(jPositive);
}

void grib::WriteTime(const forecast_time& ftime, const producer& prod, const param& par)
{
	itsGrib->Message().DataDate(stol(ftime.OriginDateTime().String("%Y%m%d")));
	itsGrib->Message().DataTime(stol(ftime.OriginDateTime().String("%H%M")));

	if (itsGrib->Message().Edition() == 1)
	{
		time_duration stepUnit = ONE_HOUR;

		if (ftime.Step().Minutes() % 60 != 0)
		{
			itsLogger.Fatal("Sub-hour output ony in grib2");
			himan::Abort();
		}
		else if (ftime.Step().Hours() > 255)  // Forecast with stepvalues that don't fit in one byte
		{
			const long hours = ftime.Step().Hours();

			if (hours % 3 == 0 && hours / 3 < 255)
			{
				itsGrib->Message().UnitOfTimeRange(10);  // 3 hours
				stepUnit = THREE_HOURS;
			}
			else if (hours % 6 == 0 && hours / 6 < 255)
			{
				itsGrib->Message().UnitOfTimeRange(11);  // 6 hours
				stepUnit = SIX_HOURS;
			}
			else if (hours % 12 == 0 && hours / 12 < 255)
			{
				itsGrib->Message().UnitOfTimeRange(12);  // 12 hours
				stepUnit = TWELVE_HOURS;
			}
			else
			{
				itsLogger.Fatal("Step too large, unable to continue");
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
				itsGrib->Message().TimeRangeIndicator(0);
				itsGrib->Message().P1((ftime.Step() / static_cast<int>(stepUnit.Hours())).Hours());
				break;
			case kAverage:
				// Average (reference time + P1 to reference time + P2)
				itsGrib->Message().TimeRangeIndicator(3);

				if (p1 < 0)
				{
					itsLogger.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				itsGrib->Message().P1(p1);
				itsGrib->Message().P2(p2);
				break;
			case kAccumulation:
				// Accumulation (reference time + P1 to reference time + P2) product considered valid at reference time
				// + P2
				itsGrib->Message().TimeRangeIndicator(4);

				if (p1 < 0)
				{
					itsLogger.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				itsGrib->Message().P1(p1);
				itsGrib->Message().P2(p2);
				break;
			case kDifference:
				// Difference (reference time + P2 minus reference time + P1) product considered valid at reference time
				// + P2
				itsGrib->Message().TimeRangeIndicator(5);

				if (p1 < 0)
				{
					itsLogger.Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				itsGrib->Message().P1(p1);
				itsGrib->Message().P2(p2);
				break;
		}

		ASSERT(itsGrib->Message().TimeRangeIndicator() != 10);
	}
	else
	{
		const long templateNumber = itsGrib->Message().ProductDefinitionTemplateNumber();

		// leadtime for this prognosis
		long unitOfTimeRange = 1;  // hours
		long stepValue = ftime.Step().Hours();

		if (ftime.Step().Minutes() % 60 != 0)
		{
			unitOfTimeRange = 0;  // minutes
			stepValue = ftime.Step().Minutes();
		}

		itsGrib->Message().UnitOfTimeRange(unitOfTimeRange);
		// Statistical processing is set in WriteParameter()
		switch (par.Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				itsGrib->Message().ForecastTime(stepValue);
				break;
			case kAverage:
			case kAccumulation:
			case kDifference:
			case kMinimum:
			case kMaximum:
				// duration of the aggregation period
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

				itsGrib->Message().SetLongKey("indicatorOfUnitForTimeRange", unitForTimeRange);
				itsGrib->Message().ForecastTime(stepValue);  // start step
				itsGrib->Message().LengthOfTimeRange(lengthOfTimeRange);

				// for productDefinitionTemplateNumber 9,10,11,12,13,14,34,43,47,61,73,73
				// grib2 has extra keys for "end of overall time interval"
				if (templateNumber >= 9 && templateNumber <= 14)
				{
					raw_time endOfInterval = raw_time(ftime.ValidDateTime());
					endOfInterval += par.Aggregation().TimeOffset() + par.Aggregation().TimeDuration();

					itsGrib->Message().SetLongKey("yearOfEndOfOverallTimeInterval", stol(endOfInterval.String("%Y")));
					itsGrib->Message().SetLongKey("monthOfEndOfOverallTimeInterval", stol(endOfInterval.String("%m")));
					itsGrib->Message().SetLongKey("dayOfEndOfOverallTimeInterval", stol(endOfInterval.String("%d")));
					itsGrib->Message().SetLongKey("hourOfEndOfOverallTimeInterval", stol(endOfInterval.String("%H")));
					itsGrib->Message().SetLongKey("minuteOfEndOfOverallTimeInterval", stol(endOfInterval.String("%M")));
					itsGrib->Message().SetLongKey("secondOfEndOfOverallTimeInterval", 0);
				}
				break;
		}
	}
}

void grib::WriteParameter(const param& par, const producer& prod, const forecast_type& ftype)
{
	if (itsGrib->Message().Edition() == 1)
	{
		if (par.GribTableVersion() != kHPMissingInt && par.GribIndicatorOfParameter() != kHPMissingInt)
		{
			// In radon table version is a parameter property, not a
			// producer property

			itsGrib->Message().Table2Version(par.GribTableVersion());
			itsGrib->Message().ParameterNumber(par.GribIndicatorOfParameter());
		}
		else if (prod.Id() != kHPMissingInt)  // no-database example has 999999 as producer
		{
			itsLogger.Warning("Parameter " + par.Name() + " does not have mapping for producer " +
			                  to_string(prod.Id()) + " in radon, setting table2version to 203");
			itsGrib->Message().ParameterNumber(0);
			itsGrib->Message().Table2Version(203);
		}
	}
	else if (itsGrib->Message().Edition() == 2)
	{
		if (par.GribParameter() == kHPMissingInt)
		{
			itsLogger.Warning("Parameter information not found from radon for producer " + to_string(prod.Id()) +
			                  ", name " + par.Name());
		}
		else
		{
			itsGrib->Message().ParameterNumber(par.GribParameter());
			itsGrib->Message().ParameterCategory(par.GribCategory());
			itsGrib->Message().ParameterDiscipline(par.GribDiscipline());
		}

		const auto aggType = par.Aggregation().Type();
		const auto procType = par.ProcessingType().Type();
		const long templateNumber = DetermineProductDefinitionTemplateNumber(aggType, procType, ftype.Type());

		itsGrib->Message().ProductDefinitionTemplateNumber(templateNumber);

		switch (aggType)
		{
			case kAverage:
				itsGrib->Message().TypeOfStatisticalProcessing(0);
				break;
			case kAccumulation:
				itsGrib->Message().TypeOfStatisticalProcessing(1);
				break;
			case kMaximum:
				itsGrib->Message().TypeOfStatisticalProcessing(2);
				break;
			case kMinimum:
				itsGrib->Message().TypeOfStatisticalProcessing(3);
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
				itsGrib->Message().SetLongKey("probabilityType", 1);
				itsGrib->Message().SetLongKey("scaledValueOfUpperLimit",
				                              static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityLessThan:  // Probability of event below lower limit
				itsGrib->Message().SetLongKey("probabilityType", 0);
				itsGrib->Message().SetLongKey("scaledValueOfLowerLimit",
				                              static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityBetween:
				itsGrib->Message().SetLongKey("probabilityType", 192);
				itsGrib->Message().SetLongKey("scaledValueOfLowerLimit",
				                              static_cast<long>(par.ProcessingType().Value()));
				itsGrib->Message().SetLongKey("scaledValueOfUpperLimit",
				                              static_cast<long>(par.ProcessingType().Value2()));
				break;
			case kProbabilityEquals:
				itsGrib->Message().SetLongKey("probabilityType", 193);
				itsGrib->Message().SetLongKey("scaledValueOfLowerLimit",
				                              static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityNotEquals:
				itsGrib->Message().SetLongKey("probabilityType", 193);
				itsGrib->Message().SetLongKey("scaledValueOfLowerLimit",
				                              static_cast<long>(par.ProcessingType().Value()));
				break;
			case kProbabilityEqualsIn:
				itsGrib->Message().SetLongKey("probabilityType", 194);
				break;
			case kFractile:
				itsGrib->Message().SetLongKey("percentileValue", static_cast<long>(par.ProcessingType().Value()));
				break;
			case kEnsembleMean:
				itsGrib->Message().SetLongKey("derivedForecast", 0);
				itsGrib->Message().SetLongKey("numberOfForecastsInEnsemble", num);
				break;
			case kSpread:
				itsGrib->Message().SetLongKey("derivedForecast", 4);
				itsGrib->Message().SetLongKey("numberOfForecastsInEnsemble", num);
				break;
			case kStandardDeviation:
				itsGrib->Message().SetLongKey("derivedForecast", 2);
				itsGrib->Message().SetLongKey("numberOfForecastsInEnsemble", num);
				break;
			case kEFI:
				itsGrib->Message().SetLongKey("derivedForecast", 199);
				itsGrib->Message().SetLongKey("numberOfForecastsInEnsemble", num);
				break;
		}
	}
}

void grib::WriteLevel(const level& lev)
{
	const long edition = itsGrib->Message().Edition();

	// Himan levels equal to grib 1

	if (edition == 1)
	{
		itsGrib->Message().LevelType(lev.Type());
	}
	else if (edition == 2)
	{
		if (lev.Type() == kHeightLayer)
		{
			itsGrib->Message().LevelType(103);
			itsGrib->Message().SetLongKey("typeOfSecondFixedSurface", 103);
		}
		else
		{
			itsGrib->Message().LevelType(itsGrib->Message().LevelTypeToAnotherEdition(lev.Type(), 2));
		}
	}

	switch (lev.Type())
	{
		case kHeightLayer:
		{
			itsGrib->Message().LevelValue(static_cast<long>(0.01 * lev.Value()), 100);    // top
			itsGrib->Message().LevelValue2(static_cast<long>(0.01 * lev.Value2()), 100);  // bottom
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

			itsGrib->Message().LevelValue(static_cast<long>(lev.Value() * scale));
			break;
		}
		default:
			itsGrib->Message().LevelValue(static_cast<long>(lev.Value()));
			break;
	}
}

template <typename T>
void WriteDataValues(const vector<T>&, NFmiGribMessage&);

template <>
void WriteDataValues(const vector<double>& values, NFmiGribMessage& msg)
{
	msg.Values(values.data(), static_cast<long>(values.size()));
}

template <>
void WriteDataValues(const vector<float>& values, NFmiGribMessage& msg)
{
	double* arr = new double[values.size()];
	replace_copy_if(values.begin(), values.end(), arr, [](const float& val) { return himan::IsMissing(val); },
	                himan::MissingDouble());

	msg.Values(arr, static_cast<long>(values.size()));

	delete[] arr;
}

himan::file_information grib::ToFile(info<double>& anInfo)
{
	return ToFile<double>(anInfo);
}

template <typename T>
himan::file_information grib::ToFile(info<T>& anInfo)
{
	// Write only that data which is currently set at descriptors

	timer aTimer;
	aTimer.Start();

	if (anInfo.Grid()->Class() == kIrregularGrid && anInfo.Grid()->Type() != kReducedGaussian)
	{
		itsLogger.Error("Unable to write irregular grid of type " + HPGridTypeToString.at(anInfo.Grid()->Type()) +
		                " to grib");
		throw kInvalidWriteOptions;
	}

	bool appendToFile = (itsWriteOptions.configuration->WriteMode() == kAllGridsToAFile ||
	                     itsWriteOptions.configuration->WriteMode() == kFewGridsToAFile);

	if (appendToFile && (itsWriteOptions.configuration->FileCompression() == kGZIP ||
	                     itsWriteOptions.configuration->FileCompression() == kBZIP2))
	{
		itsLogger.Warning("Unable to write multiple grids to a packed file");
		appendToFile = false;
	}

	file_information finfo;
	finfo.file_location = util::MakeFileName(anInfo, *itsWriteOptions.configuration) + ".grib";
	finfo.file_type = kGRIB1;
	finfo.storage_type = kLocalFileSystem;

	if (itsWriteOptions.configuration->OutputFileType() == kGRIB2)
	{
		finfo.file_location += "2";
		finfo.file_type = kGRIB2;
	}

	if (itsWriteOptions.configuration->FileCompression() == kGZIP)
	{
		finfo.file_location += ".gz";
	}
	else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
	{
		finfo.file_location += ".bz2";
	}

	namespace fs = boost::filesystem;

	fs::path pathname(finfo.file_location);

	if (!pathname.parent_path().empty() && !fs::is_directory(pathname.parent_path()))
	{
		fs::create_directories(pathname.parent_path());
	}

	long edition = static_cast<long>(itsWriteOptions.configuration->OutputFileType());

	// Check levelvalue, forecast type and param processing type since those might force us to change to grib2!

	HPForecastType forecastType = anInfo.ForecastType().Type();

	if (edition == 1 &&
	    (anInfo.Level().AB().size() > 255 || (forecastType == kEpsControl || forecastType == kEpsPerturbation) ||
	     anInfo.Param().ProcessingType().Type() != kUnknownProcessingType || anInfo.Time().Step().Minutes() % 60 != 0))
	{
		itsLogger.Trace("File type forced to GRIB2 (level value: " + to_string(anInfo.Level().Value()) +
		                ", forecast type: " + HPForecastTypeToString.at(forecastType) +
		                ", processing type: " + HPProcessingTypeToString.at(anInfo.Param().ProcessingType().Type()) +
		                " step: " + static_cast<string>(anInfo.Time().Step()) + ")");
		edition = 2;
		finfo.file_type = kGRIB2;

		if (itsWriteOptions.configuration->FileCompression() == kNoCompression &&
		    itsWriteOptions.configuration->WriteMode() != kAllGridsToAFile)
		{
			finfo.file_location += "2";
		}

		if (itsWriteOptions.configuration->FileCompression() == kGZIP)
		{
			finfo.file_location.insert(finfo.file_location.end() - 3, '2');
		}
		else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
		{
			finfo.file_location.insert(finfo.file_location.end() - 4, '2');
		}
		else if (itsWriteOptions.configuration->FileCompression() != kNoCompression)
		{
			itsLogger.Error("Unable to write to compressed grib. Unknown file compression: " +
			                HPFileCompressionToString.at(itsWriteOptions.configuration->FileCompression()));
			throw kInvalidWriteOptions;
		}
	}

	itsGrib->Message().Edition(edition);

	if (anInfo.Producer().Centre() == kHPMissingInt)
	{
		itsGrib->Message().Centre(86);
		itsGrib->Message().Process(255);
	}
	else
	{
		itsGrib->Message().Centre(anInfo.Producer().Centre());
		itsGrib->Message().Process(anInfo.Producer().Process());
	}

	// Forecast type

	// Note: forecast type is also checked in WriteParameter(), because
	// it might affect productDefinitionTemplateNumber (grib2)

	itsGrib->Message().ForecastType(anInfo.ForecastType().Type());

	if (static_cast<int>(anInfo.ForecastType().Type()) > 2)
	{
		itsGrib->Message().ForecastTypeValue(static_cast<long>(anInfo.ForecastType().Value()));
		auto r = GET_PLUGIN(radon);

		try
		{
			const long ensembleSize = stol(r->RadonDB().GetProducerMetaData(anInfo.Producer().Id(), "ensemble size"));
			itsGrib->Message().SetLongKey("numberOfForecastsInEnsemble", ensembleSize);
		}
		catch (const invalid_argument& e)
		{
			itsLogger.Warning("Unable to get valid ensemble size information from radon for producer " +
			                  to_string(anInfo.Producer().Id()));
		}
	}

	// Parameter

	WriteParameter(anInfo.Param(), anInfo.Producer(), anInfo.ForecastType());

	// Area and Grid

	WriteAreaAndGrid(anInfo.Grid(), anInfo.Producer());

	// Time information

	WriteTime(anInfo.Time(), anInfo.Producer(), anInfo.Param());

	// Level

	WriteLevel(anInfo.Level());

	// set to missing value to a large value to prevent it from mixing up with valid
	// values in the data

	itsGrib->Message().MissingValue(gribMissing);

	if (itsWriteOptions.use_bitmap && anInfo.Data().MissingCount() > 0)
	{
		itsGrib->Message().Bitmap(true);
	}

	/*
	 * Possible precipitation form value encoding must be done before determining
	 * bits per value, as the range of values is changed.
	 */

	const auto paramName = anInfo.Param().Name();
	const int precision =
	    (itsWriteOptions.precision == kHPMissingInt) ? anInfo.Param().Precision() : itsWriteOptions.precision;

	long bitsPerValue;

	if (edition == 2 && (paramName == "PRECFORM-N" || paramName == "PRECFORM2-N"))
	{
		// We take a copy of the data, because the values at cache should not change
		auto values = anInfo.Data().Values();

		EncodePrecipitationFormToGrib2(values);
		bitsPerValue = DetermineBitsPerValue(values, precision);

		// Change missing value 'nan' to a real fp value
		replace_if(values.begin(), values.end(), [](const T& v) { return IsMissing(v); }, gribMissing);

		itsGrib->Message().BitsPerValue(bitsPerValue);
		WriteDataValues<T>(values, itsGrib->Message());
	}
	else
	{
		// In this branch no copy is made
		const auto& values = anInfo.Data().Values();
		bitsPerValue = DetermineBitsPerValue(values, precision);

		// Change missing value 'nan' to a real fp value
		anInfo.Data().MissingValue(gribMissing);

		itsGrib->Message().BitsPerValue(bitsPerValue);

		WriteDataValues<T>(values, itsGrib->Message());
	}

	itsLogger.Trace("Using " + (precision == kHPMissingInt ? "maximum precision" : to_string(precision) + " decimals") +
	                " (" + to_string(bitsPerValue) + " bits) to store " + paramName);

	// Return missing value to nan if info is recycled (luatool)
	anInfo.Data().MissingValue(MissingValue<T>());

	if (edition == 2 && itsWriteOptions.packing_type == kJpegPacking)
	{
		itsGrib->Message().PackingType("grid_jpeg");
	}

	/*
	 *  GRIB 1
	 *
	 * 	BIT	VALUE	MEANING
	 *	1	0		Direction increments not given
	 *	1	1		Direction increments given
	 *	2	0		Earth assumed spherical with radius = 6367.47 km
	 *	2	1		Earth assumed oblate spheroid with size as determined by IAU in 1965: 6378.160 km, 6356.775 km, f =
	 *1/297.0
	 *	3-4	0		reserved (set to 0)
	 *	5	0		u- and v-components of vector quantities resolved relative to easterly and northerly directions
	 * 	5	1		u and v components of vector quantities resolved relative to the defined grid in the direction of
	 *increasing x and y (or i and j) coordinates respectively
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
		itsGrib->Message().ResolutionAndComponentFlags(0);
	}
	else
	{
		if (edition == 1)
		{
			itsGrib->Message().ResolutionAndComponentFlags(128);  // 10000000
		}
		else
		{
			itsGrib->Message().ResolutionAndComponentFlags(48);  // 00110000
		}

		if (anInfo.Grid()->UVRelativeToGrid())
		{
			itsGrib->Message().UVRelativeToGrid(true);
		}
	}

	vector<double> AB = anInfo.Level().AB();

	if (!AB.empty())
	{
		itsGrib->Message().NV(static_cast<long>(AB.size()));
		itsGrib->Message().PV(AB, AB.size());
	}

	// message length can only be received from eccodes since it includes
	// all grib headers etc
	finfo.length = itsGrib->Message().GetLongKey("totalLength");

	if (itsWriteOptions.configuration->WriteMode() != kSingleGridToAFile)
	{
		// appending to a file is a serial operation -- two threads cannot
		// append to a single file simultaneously. therefore offset is just
		// the size of the file so far.

		try
		{
			finfo.offset = boost::filesystem::file_size(finfo.file_location);
		}
		catch (const boost::filesystem::filesystem_error& e)
		{
			finfo.offset = 0;
		}

		// handling message number is a bit tricker. fmigrib library cannot really
		// be used for tracking message no of written messages, because neither it
		// nor eccodes has any visibility to any ossibly existing messages in a file
		// that is appended to. therefore here we are tracking the message count per
		// file, but this is assuming that when himan starts, the file appended to
		// does not exist!

		static std::map<std::string, unsigned long> messages;

		try
		{
			messages.at(finfo.file_location) = messages.at(finfo.file_location) + 1;
		}
		catch (const out_of_range& e)
		{
			messages[finfo.file_location] = 0;
		}

		finfo.message_no = messages.at(finfo.file_location);
	}
	else
	{
		finfo.offset = 0;
		finfo.message_no = 0;
	}

	itsGrib->Message().Write(finfo.file_location, appendToFile);

	aTimer.Stop();
	const float duration = static_cast<float>(aTimer.GetTime());
	const float bytes = static_cast<float>(finfo.length.get());  // TODO: does not work correctly if file is packed
	const float speed = (bytes / 1024.f / 1024.f) / (duration / 1000.f);

	stringstream ss;

	ss.precision((speed < 1) ? 1 : 0);

	string verb = (appendToFile ? "Appended to " : "Wrote ");

	ss << verb << "file '" << finfo.file_location << "' (" << fixed << speed << " MB/s)";
	itsLogger.Info(ss.str());

	return finfo;
}

template himan::file_information grib::ToFile<double>(info<double>&);
template himan::file_information grib::ToFile<float>(info<float>&);

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
				// Earth assumed oblate spheroid as defined in IAG-GRS80 model (major axis = 6,378,137.0 m, minor axis =
				// 6,356,752.314 m, f = 1/298.257222101)
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
				//  Earth represented by the Ordnance Survey Great Britain 1936 Datum, using the Airy 1830 Spheroid, the
				//  Greenwich meridian as 0 longitude, and the Newlyn datum as mean sea level, 0 height
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

unique_ptr<himan::grid> grib::ReadAreaAndGrid() const
{
	bool iNegative = itsGrib->Message().IScansNegatively();
	bool jPositive = itsGrib->Message().JScansPositively();

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

	double X0 = itsGrib->Message().X0();
	double Y0 = itsGrib->Message().Y0();

	// GRIB2 has longitude 0 .. 360, but in neons we have it -180 .. 180
	// NB! ONLY FOR EC and FMI! GFS and GEM geometries are in grib2 format
	//
	// Make conversion to GRIB1 style coordinates, but in the long run we should figure out how to
	// handle grib 1 & grib 2 longitude values in a smart way. (a single geometry
	// can have coordinates in both ways!)

	long centre = itsGrib->Message().Centre();

	if (itsGrib->Message().Edition() == 2 && (centre == 98 || centre == 86) && X0 != 0)
	{
		X0 -= 360;
		if (X0 < -180)
			X0 += 360;
	}

	himan::point firstPoint(X0, Y0);

	if (centre == 98 && firstPoint.X() == 180)
	{
		/**
		 * Global EC data area is defined as
		 *
		 * latitudeOfFirstGridPointInDegrees = 90;
		 * longitudeOfFirstGridPointInDegrees = 180;
		 * latitudeOfLastGridPointInDegrees = 0;
		 * longitudeOfLastGridPointInDegrees = 180;
		 *
		 * Normalize the first value to -180.
		 */

		ASSERT(m == kBottomLeft || m == kTopLeft);  // Want to make sure we always read from left to right

		firstPoint.X(-180.);
	}

	unique_ptr<grid> newGrid;

	switch (itsGrib->Message().NormalizedGridType())
	{
		case 0:
		{
			newGrid = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid);
			latitude_longitude_grid* const rg = dynamic_cast<latitude_longitude_grid*>(newGrid.get());

			size_t ni = static_cast<size_t>(itsGrib->Message().SizeX());
			size_t nj = static_cast<size_t>(itsGrib->Message().SizeY());

			rg->Ni(ni);
			rg->Nj(nj);

			rg->Di(itsGrib->Message().iDirectionIncrement());
			rg->Dj(itsGrib->Message().jDirectionIncrement());

			rg->ScanningMode(m);

			rg->FirstPoint(firstPoint);

			double X1 = itsGrib->Message().X1();
			double Y1 = itsGrib->Message().Y1();

			rg->LastPoint(point(X1, Y1));

			break;
		}
		case 3:
		{
			newGrid = unique_ptr<lambert_conformal_grid>(
			    new lambert_conformal_grid(m, point(itsGrib->Message().X0(), itsGrib->Message().Y0())));
			lambert_conformal_grid* const lccg = dynamic_cast<lambert_conformal_grid*>(newGrid.get());

			lccg->Ni(static_cast<size_t>(itsGrib->Message().SizeX()));
			lccg->Nj(static_cast<size_t>(itsGrib->Message().SizeY()));

			lccg->ScanningMode(m);
			lccg->Orientation(itsGrib->Message().GridOrientation());
			lccg->Di(itsGrib->Message().XLengthInMeters());
			lccg->Dj(itsGrib->Message().YLengthInMeters());

			lccg->StandardParallel1(static_cast<double>(itsGrib->Message().GetLongKey("Latin1InDegrees")));
			lccg->StandardParallel2(static_cast<double>(itsGrib->Message().GetLongKey("Latin2InDegrees")));
			lccg->UVRelativeToGrid(itsGrib->Message().UVRelativeToGrid());

			break;
		}

		case 4:
		{
			if (m == kTopLeft || m == kUnknownScanningMode)
			{
				newGrid = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
				reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(newGrid.get());

				gg->N(static_cast<int>(itsGrib->Message().GetLongKey("N")));
				gg->NumberOfPointsAlongParallels(itsGrib->Message().PL());

				break;
			}
		}

		case 5:
		{
			newGrid = unique_ptr<stereographic_grid>(new stereographic_grid);
			stereographic_grid* const rg = dynamic_cast<stereographic_grid*>(newGrid.get());

			size_t ni = static_cast<size_t>(itsGrib->Message().SizeX());
			size_t nj = static_cast<size_t>(itsGrib->Message().SizeY());

			rg->Ni(ni);
			rg->Nj(nj);

			rg->Orientation(itsGrib->Message().GridOrientation());
			rg->Di(itsGrib->Message().XLengthInMeters());
			rg->Dj(itsGrib->Message().YLengthInMeters());

			rg->ScanningMode(m);
			rg->UVRelativeToGrid(itsGrib->Message().UVRelativeToGrid());

			rg->FirstPoint(firstPoint);

			break;
		}

		case 10:
		{
			newGrid = unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid);
			rotated_latitude_longitude_grid* const rg = dynamic_cast<rotated_latitude_longitude_grid*>(newGrid.get());

			size_t ni = static_cast<size_t>(itsGrib->Message().SizeX());
			size_t nj = static_cast<size_t>(itsGrib->Message().SizeY());

			rg->Ni(ni);
			rg->Nj(nj);

			rg->Di(itsGrib->Message().iDirectionIncrement());
			rg->Dj(itsGrib->Message().jDirectionIncrement());

			rg->SouthPole(himan::point(itsGrib->Message().SouthPoleX(), itsGrib->Message().SouthPoleY()));
			rg->UVRelativeToGrid(itsGrib->Message().UVRelativeToGrid());

			rg->ScanningMode(m);

			rg->FirstPoint(firstPoint);

			double X1 = itsGrib->Message().X1();
			double Y1 = itsGrib->Message().Y1();

			rg->LastPoint(point(X1, Y1));

			break;
		}
		default:
			throw runtime_error(ClassName() +
			                    ": Unsupported grid type: " + to_string(itsGrib->Message().NormalizedGridType()));
			break;
	}

	const auto shape = ReadEarthShape(itsGrib->Message());
	newGrid->EarthShape(shape);

	return newGrid;
}

himan::param grib::ReadParam(const search_options& options, const producer& prod) const
{
	param p;

	long number = itsGrib->Message().ParameterNumber();

	shared_ptr<radon> r;

	auto dbtype = options.configuration->DatabaseType();

	if (itsGrib->Message().Edition() == 1)
	{
		long no_vers = itsGrib->Message().Table2Version();

		long timeRangeIndicator = itsGrib->Message().TimeRangeIndicator();

		string parmName = "";

		if (dbtype == kRadon)
		{
			r = GET_PLUGIN(radon);

			auto parminfo = r->RadonDB().GetParameterFromGrib1(prod.Id(), no_vers, number, timeRangeIndicator,
			                                                   itsGrib->Message().NormalizedLevelType(),
			                                                   static_cast<double>(itsGrib->Message().LevelValue()));

			if (!parminfo.empty())
			{
				parmName = parminfo["name"];
			}
		}

		if (parmName.empty() && dbtype == kNoDatabase)
		{
			parmName = GetParamNameFromGribShortName(options.configuration->ParamFile(),
			                                         itsGrib->Message().GetStringKey("shortName"));
		}

		if (parmName.empty())
		{
			itsLogger.Warning("Parameter name not found from " + HPDatabaseTypeToString.at(dbtype) +
			                  " for no_vers: " + to_string(no_vers) + ", number: " + to_string(number) +
			                  ", timeRangeIndicator: " + to_string(timeRangeIndicator));
		}
		else
		{
			p.Name(parmName);
		}

		p.GribParameter(number);
		p.GribTableVersion(no_vers);

		// Determine aggregation

		aggregation a;

		switch (timeRangeIndicator)
		{
			case 0:  // forecast
			case 1:  // analysis
				break;

			case 3:  // average
				a.Type(kAverage);
				a.TimeDuration(DurationFromTimeRange(itsGrib->Message().UnitOfTimeRange()) *
				               static_cast<int>(itsGrib->Message().P2() - itsGrib->Message().P1()));
				break;
			case 4:  // accumulation
				a.Type(kAccumulation);
				a.TimeDuration(DurationFromTimeRange(itsGrib->Message().UnitOfTimeRange()) *
				               static_cast<int>(itsGrib->Message().P2() - itsGrib->Message().P1()));
				break;
		}

		if (a.Type() != kUnknownAggregationType)
		{
			p.Aggregation(a);
		}
	}
	else
	{
		long category = itsGrib->Message().ParameterCategory();
		long discipline = itsGrib->Message().ParameterDiscipline();

		string parmName = "";

		const long tosp = (itsGrib->Message().TypeOfStatisticalProcessing() == -999)
		                      ? -1
		                      : itsGrib->Message().TypeOfStatisticalProcessing();

		if (dbtype == kRadon)
		{
			r = GET_PLUGIN(radon);

			auto parminfo = r->RadonDB().GetParameterFromGrib2(
			    prod.Id(), discipline, category, number, itsGrib->Message().LevelType(),
			    static_cast<double>(itsGrib->Message().LevelValue()), tosp);

			if (parminfo.size())
			{
				parmName = parminfo["name"];
			}
		}

		if (parmName.empty() && dbtype == kNoDatabase)
		{
			parmName = GetParamNameFromGribShortName(options.configuration->ParamFile(),
			                                         itsGrib->Message().GetStringKey("shortName"));
		}

		if (parmName.empty())
		{
			itsLogger.Warning("Parameter name not found from database for discipline: " + to_string(discipline) +
			                  ", category: " + to_string(category) + ", number: " + to_string(number) +
			                  ", statistical processing: " + to_string(tosp));
		}
		else
		{
			p.Name(parmName);
		}

		p.GribParameter(number);
		p.GribDiscipline(discipline);
		p.GribCategory(category);

		aggregation a;

		const long unitForTimeRange = itsGrib->Message().GetLongKey("indicatorOfUnitForTimeRange");
		const auto td =
		    DurationFromTimeRange(unitForTimeRange) * static_cast<int>(itsGrib->Message().LengthOfTimeRange());
		switch (itsGrib->Message().TypeOfStatisticalProcessing())
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
		}

		if (a.TimeDuration().Empty() == false)
		{
			p.Aggregation(a);
		}
	}

	string unit = itsGrib->Message().ParameterUnit();

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
		itsLogger.Trace("Unable to determine himan parameter unit for grib unit " + itsGrib->Message().ParameterUnit());
	}

	return p;
}

himan::forecast_time grib::ReadTime() const
{
	string dataDate = to_string(itsGrib->Message().DataDate());

	/*
	 * dataTime is HH24MM in long datatype.
	 * So, for example analysistime 00 is 0, and 06 is 600.
	 */

	long dt = itsGrib->Message().DataTime();
	char fmt[5];
	snprintf(fmt, 5, "%04ld", dt);

	long step = itsGrib->Message().NormalizedStep(true, true);

	string originDateTimeStr = dataDate + string(fmt);
	raw_time originDateTime(originDateTimeStr, "%Y%m%d%H%M");

	forecast_time t(originDateTime, originDateTime);

	long unitOfTimeRange = itsGrib->Message().NormalizedUnitOfTimeRange();

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
			itsLogger.Warning("Unsupported unit of time range: " + to_string(unitOfTimeRange));
			break;
	}

	t.ValidDateTime().Adjust(timeResolution, static_cast<int>(step));

	return t;
}

himan::level grib::ReadLevel(const search_options& opts, const producer& prod) const
{
	himan::HPLevelType levelType = kUnknownLevel;

	if (opts.configuration->DatabaseType() == kNoDatabase)
	{
		// Minimal set of levels for those who might try to run himan
		// without a database connection
		const long gribLevel = itsGrib->Message().NormalizedLevelType();

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
				itsLogger.Fatal("Unsupported level type for no database mode: " + to_string(gribLevel));
				himan::Abort();
		}
	}
	else
	{
		const long gribLevel = itsGrib->Message().LevelType();

		auto r = GET_PLUGIN(radon);

		auto levelInfo = r->RadonDB().GetLevelFromGrib(prod.Id(), gribLevel, itsGrib->Message().Edition());

		if (levelInfo.empty())
		{
			itsLogger.Fatal("Unsupported level type for producer " + to_string(prod.Id()) + ": " +
			                to_string(gribLevel) + ", grib edition " + to_string(itsGrib->Message().Edition()));
			himan::Abort();
		}

		string levelName = levelInfo["name"];
		boost::algorithm::to_lower(levelName);

		// Special cases:

		// 1. Check if we have a height_layer, which in grib2 is first and second leveltype 103

		if (levelName == "height" && itsGrib->Message().Edition() == 2)
		{
			const long levelType2 = itsGrib->Message().GetLongKey("typeOfSecondFixedSurface");
			const long levelValue2 = itsGrib->Message().LevelValue2();

			if (levelType2 == 103 && levelValue2 != -999 && levelValue2 != 214748364700)
			{
				levelName = "height_layer";
			}
		}

		levelType = HPStringToLevelType.at(levelName);
	}

	himan::level l;

	switch (levelType)
	{
		case himan::kHeightLayer:
			l = level(levelType, 100 * static_cast<double>(itsGrib->Message().LevelValue()),
			          100 * static_cast<double>(itsGrib->Message().LevelValue2()));
			break;

		case himan::kGroundDepth:
		case himan::kPressureDelta:
		{
			long gribLevelValue2 = itsGrib->Message().LevelValue2();
			// Missing in grib is all bits set
			if (gribLevelValue2 == 2147483647)
			{
				gribLevelValue2 = -1;
			}

			l = level(levelType, static_cast<float>(itsGrib->Message().LevelValue()),
			          static_cast<float>(gribLevelValue2));
		}
		break;

		default:
			l = level(levelType, static_cast<float>(itsGrib->Message().LevelValue()));
			break;
	}

	return l;
}

himan::producer grib::ReadProducer(const search_options& options) const
{
	long centre = itsGrib->Message().Centre();
	long process = itsGrib->Message().Process();

	producer prod(centre, process);

	if (options.configuration->DatabaseType() == kRadon)
	{
		// Do a double check and fetch the fmi producer id from database.

		long typeId = 1;  // deterministic forecast, default
		long msgType = itsGrib->Message().ForecastType();

		if (msgType == 2)
		{
			typeId = 2;  // ANALYSIS
		}
		else if (msgType == 3 || msgType == 4)
		{
			typeId = 3;  // ENSEMBLE
		}

		auto r = GET_PLUGIN(radon);

		auto prodInfo = r->RadonDB().GetProducerFromGrib(centre, process, typeId);

		if (!prodInfo.empty())
		{
			prod.Id(stoi(prodInfo["id"]));
		}
		else
		{
			if (centre == 98 && (process <= 149 && process >= 142))
			{
				if (typeId == 1 || typeId == 2)
				{
					prod.Id(131);
				}
				else if (typeId == 3)
				{
					prod.Id(134);
				}

				return prod;
			}

			itsLogger.Warning("Producer information not found from database for centre " + to_string(centre) +
			                  ", process " + to_string(process) + " type " + to_string(typeId));
		}
	}

	return prod;
}

template <typename T>
void ReadDataValues(vector<T>&, NFmiGribMessage& msg);

template <>
void ReadDataValues(vector<double>& values, NFmiGribMessage& msg)
{
	size_t len = msg.ValuesLength();
	msg.GetValues(values.data(), &len);
}

template <>
void ReadDataValues(vector<float>& values, NFmiGribMessage& msg)
{
	double* arr = new double[values.size()];
	size_t len = msg.ValuesLength();
	msg.GetValues(arr, &len);

	replace_copy_if(arr, arr + values.size(), values.begin(), [](const double& val) { return himan::IsMissing(val); },
	                himan::MissingFloat());

	delete[] arr;
}

template <typename T>
void grib::ReadData(shared_ptr<info<T>> newInfo, bool readPackedData) const
{
	auto& dm = newInfo->Data();

	bool decodePrecipitationForm = false;

#if defined GRIB_READ_PACKED_DATA && defined HAVE_CUDA

	const auto paramName = newInfo->Param().Name();
	long producerId = newInfo->Producer().Id();

	if (itsGrib->Message().Edition() == 2 && (paramName == "PRECFORM-N" || paramName == "PRECFORM2-N") &&
	    (producerId == 230 || producerId == 240 || producerId == 243 || producerId == 250 || producerId == 260 ||
	     producerId == 270))
	{
		decodePrecipitationForm = true;
	}

	if (readPackedData && decodePrecipitationForm == false && itsGrib->Message().PackingType() == "grid_simple")
	{
		// Get coefficient information

		double bsf = static_cast<double>(itsGrib->Message().BinaryScaleFactor());
		double dsf = static_cast<double>(itsGrib->Message().DecimalScaleFactor());
		double rv = itsGrib->Message().ReferenceValue();
		int bpv = static_cast<int>(itsGrib->Message().BitsPerValue());

		auto packed = make_shared<simple_packed>(bpv, util::ToPower(bsf, 2), util::ToPower(-dsf, 10), rv);

		// Get packed values from grib

		size_t len = itsGrib->Message().PackedValuesLength();
		int* unpackedBitmap = 0;

		packed->unpackedLength = itsGrib->Message().SizeX() * itsGrib->Message().SizeY();

		if (len > 0)
		{
			ASSERT(packed->data == 0);
			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&packed->data), len * sizeof(unsigned char)));

			itsGrib->Message().PackedValues(packed->data);
			packed->packedLength = len;

			itsLogger.Trace("Retrieved " + to_string(len) + " bytes of packed data from grib");
		}
		else
		{
			itsLogger.Trace("Grid is constant or empty");
		}

		if (itsGrib->Message().Bitmap())
		{
			size_t bitmap_len = itsGrib->Message().BytesLength("bitmap");
			size_t bitmap_size = static_cast<size_t>(ceil(static_cast<double>(bitmap_len) / 8));

			itsLogger.Trace("Grib has bitmap, length " + to_string(bitmap_len) + " size " + to_string(bitmap_size) +
			                " bytes");

			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&unpackedBitmap), bitmap_len * sizeof(int)));

			unsigned char* bitmap = new unsigned char[bitmap_size];

			itsGrib->Message().Bytes("bitmap", bitmap);

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
		dm.MissingValue(gribMissing);
		ReadDataValues<T>(dm.Values(), itsGrib->Message());

		dm.MissingValue(MissingValue<T>());

		if (decodePrecipitationForm)
		{
			itsLogger.Trace("Decoding precipitation form grib2 values");
			DecodePrecipitationFormFromGrib2(dm.Values());
		}

		itsLogger.Trace("Retrieved " + std::to_string(dm.Size() * sizeof(T)) + " bytes of unpacked data from grib");
	}
}

template <typename T>
bool grib::CreateInfoFromGrib(const search_options& options, bool readPackedData, bool readIfNotMatching,
                              shared_ptr<info<T>> newInfo) const
{
	shared_ptr<radon> r;

	if (options.configuration->DatabaseType() == kRadon)
	{
		r = GET_PLUGIN(radon);
	}

	bool dataIsValid = true;

	auto prod = ReadProducer(options);

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

	auto p = ReadParam(options, prod);

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

			return false;
		}
	}

	auto t = ReadTime();

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

	auto l = ReadLevel(options, prod);

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

	forecast_type ty(static_cast<HPForecastType>(itsGrib->Message().ForecastType()),
	                 static_cast<double>(itsGrib->Message().ForecastTypeValue()));

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
		long nv = itsGrib->Message().NV();

		if (nv > 0)
		{
			ab = itsGrib->Message().PV();
		}
	}

	l.AB(ab);

	newInfo->template Set<param>({p});
	newInfo->template Set<forecast_time>({t});
	newInfo->template Set<level>({l});
	newInfo->template Set<forecast_type>({ty});

	unique_ptr<grid> newGrid = ReadAreaAndGrid();

	ASSERT(newGrid);

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(newGrid->Clone());

	newInfo->Create(b, true);

	// Set descriptors

	newInfo->template Find<param>(p);
	newInfo->template Find<forecast_time>(t);
	newInfo->template Find<level>(l);
	newInfo->template Find<forecast_type>(ty);

	/*
	 * Read data from grib. If interpolation is required, it's better to do the unpacking
	 * at host to avoid unnecessary copying between CPU and GPU.
	 */

	ReadData(newInfo, readPackedData && (*options.configuration->BaseGrid() == *newInfo->Grid()));

	if (!dataIsValid)
	{
		return false;
	}

	newInfo->First();
	return true;
}

template bool grib::CreateInfoFromGrib<double>(const search_options&, bool, bool, shared_ptr<info<double>>) const;

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

	if (readIfNotMatching || !theInputFile.offset)
	{
		// read all messages from local 'auxiliary' file
		if (!itsGrib->Open(theInputFile.file_location))
		{
			itsLogger.Error("Opening file '" + theInputFile.file_location + "' failed");
			return infos;
		}

		while (itsGrib->NextMessage())
		{
			auto newInfo = make_shared<info<T>>();
			if (CreateInfoFromGrib(options, readPackedData, readIfNotMatching, newInfo) || readIfNotMatching)
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

		if (!itsGrib->ReadMessage(buf.data, buf.length))
		{
			itsLogger.Error("Creating GRIB message from memory failed");
			return infos;
		}

		auto newInfo = make_shared<info<T>>();

		if (CreateInfoFromGrib(options, readPackedData, false, newInfo))
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
		ss << "position " << theInputFile.offset.get() << ":" << bytes;
	}

	ss << " (" << fixed << speed << " MB/s)";

	itsLogger.Debug(ss.str());

	return infos;
}

template vector<shared_ptr<himan::info<double>>> grib::FromFile<double>(const file_information&, const search_options&,
                                                                        bool, bool) const;
template vector<shared_ptr<himan::info<float>>> grib::FromFile<float>(const file_information&, const search_options&,
                                                                      bool, bool) const;
void grib::UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len,
                        size_t unpackedLen) const
{
	size_t i, idx = 0;
	int v = 1;

	short j = 0;

	for (i = 0; i < len; i++)
	{
		for (j = 7; j >= 0; j--)
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
		auto elems = himan::util::Split(line, ",", false);

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
