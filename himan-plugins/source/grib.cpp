/**
 * @file grib.cpp
 *
 */

#include "grib.h"
#include "grid.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include "plugin_factory.h"
#include "producer.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include "util.h"
#include <algorithm>
#include <boost/filesystem.hpp>

using namespace std;
using namespace himan::plugin;

#include "radon.h"

#include "cuda_helper.h"
#include "simple_packed.h"

#define BitMask1(i) (1u << i)
#define BitTest(n, i) !!((n)&BitMask1(i))

std::string GetParamNameFromGribShortName(const std::string& paramFileName, const std::string& shortName);
void EncodePrecipitationFormToGrib2(vector<double>& arr);
void DecodePrecipitationFormFromGrib2(vector<double>& arr);

const double gribMissing = 32700.;

long DetermineBitsPerValue(const vector<double>& values, double precision)
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
	double min = himan::MissingDouble(), max = himan::MissingDouble();

	for (const auto& v : values)
	{
		// man fmin:
		// "If one argument is a NaN, the other argument is returned."
		min = fmin(min, v);
		max = fmax(max, v);
	}

	// Required scale value to reach wanted precision
	const double D = pow(10, precision);

	// Range of scaled data, ie the largest value we must be able to write
	const int range = static_cast<int>(ceil(D * max - D * min));

	if (himan::IsMissingDouble(min) || himan::IsMissingDouble(max) || range == 0)
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

grib::grib()
{
	itsLogger = logger("grib");

	itsGrib = make_shared<NFmiGrib>();
}

shared_ptr<NFmiGrib> grib::Reader()
{
	return itsGrib;
}
void grib::WriteAreaAndGrid(info& anInfo)
{
	const long edition = itsGrib->Message().Edition();
	HPScanningMode scmode = kUnknownScanningMode;

	auto firstGridPoint = anInfo.Grid()->FirstPoint();

	if (edition == 2)
	{
		if (firstGridPoint.X() < 0)
		{
			firstGridPoint.X(firstGridPoint.X() + 360.);
		}
	}

	switch (anInfo.Grid()->Type())
	{
		case kLatitudeLongitude:
		{
			latitude_longitude_grid* const rg = dynamic_cast<latitude_longitude_grid*>(anInfo.Grid());

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
			rotated_latitude_longitude_grid* const rg = dynamic_cast<rotated_latitude_longitude_grid*>(anInfo.Grid());

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

			itsGrib->Message().UVRelativeToGrid(rg->UVRelativeToGrid());

			itsGrib->Message().SizeX(static_cast<long>(rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long>(rg->Nj()));

			scmode = rg->ScanningMode();

			break;
		}

		case kStereographic:
		{
			stereographic_grid* const rg = dynamic_cast<stereographic_grid*>(anInfo.Grid());

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

			break;
		}

		case kReducedGaussian:
		{
			reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(anInfo.Grid());

			long gridType = 4;  // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			ASSERT(gg->ScanningMode() == kTopLeft);

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(gg->BottomLeft().X());
			itsGrib->Message().Y0(gg->BottomLeft().Y());
			itsGrib->Message().X1(gg->TopRight().X());
			itsGrib->Message().Y1(gg->TopRight().Y());

			itsGrib->Message().SetLongKey("iDirectionIncrement", 65535);
			itsGrib->Message().SetLongKey("numberOfPointsAlongAParallel", 65535);

			itsGrib->Message().SetLongKey("N", static_cast<long>(gg->N()));

			itsGrib->Message().PL(gg->NumberOfPointsAlongParallels());

			scmode = gg->ScanningMode();

			break;
		}
		case kLambertConformalConic:
		{
			lambert_conformal_grid* const lccg = dynamic_cast<lambert_conformal_grid*>(anInfo.Grid());

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

			break;
		}

		default:
			itsLogger.Fatal("Invalid projection while writing grib: " +
			                boost::lexical_cast<string>(anInfo.Grid()->Type()));
			himan::Abort();
	}

	// Earth shape is not set yet, as it will change many of the test results (metadata changes)
	// and we don't want to do that until we have set the *correct* radius for those producers
	// that we have it for. Remember that at this point we force all producers to use radius
	// found from newbase.

#if 0
	// Set earth shape

	const double a = anInfo.Grid()->EarthShape().A(), b = anInfo.Grid()->EarthShape().B();

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
	itsGrib->Message().Centre(anInfo.Producer().Centre() == kHPMissingInt ? 86 : anInfo.Producer().Centre());
	itsGrib->Message().Process(anInfo.Producer().Process() == kHPMissingInt ? 255 : anInfo.Producer().Process());

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

void grib::WriteTime(info& anInfo)
{
	itsGrib->Message().DataDate(boost::lexical_cast<long>(anInfo.Time().OriginDateTime().String("%Y%m%d")));
	itsGrib->Message().DataTime(boost::lexical_cast<long>(anInfo.Time().OriginDateTime().String("%H%M")));

	double divisor = 1;
	long unitOfTimeRange = 1;

	if (anInfo.Producer().Id() == 210)
	{
		// Harmonie
		unitOfTimeRange = 13;  // 15 minutes
		divisor = 15;
	}
	else if (anInfo.Time().Step() > 255)  // Forecast with stepvalues that don't fit in one byte
	{
		long step = anInfo.Time().Step();

		if (step % 3 == 0 && step / 3 < 255)
		{
			unitOfTimeRange = 10;  // 3 hours
			divisor = 3;
		}
		else if (step % 6 == 0 && step / 6 < 255)
		{
			unitOfTimeRange = 11;  // 6 hours
			divisor = 6;
		}
		else if (step % 12 == 0 && step / 12 < 255)
		{
			unitOfTimeRange = 12;  // 12 hours
			divisor = 12;
		}
		else
		{
			itsLogger.Fatal("Step too large, unable to continue");
			himan::Abort();
		}
	}

	long period = itsWriteOptions.configuration->ForecastStep();

	if (anInfo.Param().Aggregation().TimeResolution() != kUnknownTimeResolution)
	{
		period = anInfo.Param().Aggregation().TimeResolutionValue();

		// Time range and aggregation need to share a common time unit
		if (anInfo.Param().Aggregation().TimeResolution() == kHourResolution &&
		    (unitOfTimeRange == 0 || unitOfTimeRange == 13))
		{
			period *= 60;
		}
	}

	if (itsGrib->Message().Edition() == 1)
	{
		itsGrib->Message().UnitOfTimeRange(unitOfTimeRange);

		long p1;

		if (anInfo.Param().Aggregation().FirstTimeValue() != kHPMissingInt)
		{
			p1 = anInfo.Param().Aggregation().FirstTimeValue();
		}
		else
		{
			p1 = static_cast<long>(static_cast<double>(anInfo.Time().Step() - period) / divisor);
		}

		switch (anInfo.Param().Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				// Forecast product valid for reference time + P1 (P1 > 0)
				itsGrib->Message().TimeRangeIndicator(0);
				itsGrib->Message().P1(static_cast<int>(anInfo.Time().Step() / divisor));
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
				itsGrib->Message().P2(static_cast<long>(anInfo.Time().Step() / divisor));
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
				itsGrib->Message().P2(static_cast<long>(anInfo.Time().Step() / divisor));
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
				itsGrib->Message().P2(static_cast<long>(anInfo.Time().Step() / divisor));
				break;
		}

		ASSERT(itsGrib->Message().TimeRangeIndicator() != 10);
	}
	else
	{
		if (unitOfTimeRange == 13)
		{
			unitOfTimeRange = 254;
		}

		itsGrib->Message().UnitOfTimeRange(unitOfTimeRange);
		// Statistical processing is set in WriteParameter()
		switch (anInfo.Param().Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				itsGrib->Message().ForecastTime(static_cast<int>(static_cast<double>(anInfo.Time().Step()) / divisor));
				break;
			case kAverage:
			case kAccumulation:
			case kDifference:
				itsGrib->Message().SetLongKey("indicatorOfUnitForTimeRange", unitOfTimeRange);

				long firstTime = static_cast<long>(anInfo.Param().Aggregation().FirstTimeValue());

				if (firstTime == kHPMissingInt)
				{
					firstTime = static_cast<long>(static_cast<double>(anInfo.Time().Step() - period) / divisor);
				}

				itsGrib->Message().ForecastTime(firstTime);  // start step

#if 0
				if (anInfo.Param().Aggregation().TimeResolution() == kUnknownTimeResolution)
				{
					// Although parameter is an accumulation, the period is unknown
					// eg. RRR
					itsGrib->Message().LengthOfTimeRange(
					    static_cast<long>(itsWriteOptions.configuration->ForecastStep() / divisor));  // step length
				}
				else
#endif
				{
					// Accumulation period is known
					// eg. RR-1-MM
					itsGrib->Message().LengthOfTimeRange(
					    static_cast<long>(anInfo.Param().Aggregation().TimeResolutionValue()));
				}
				break;
		}
	}
}

void grib::WriteParameter(info& anInfo)
{
	if (itsGrib->Message().Edition() == 1)
	{
		if (anInfo.Param().GribTableVersion() != kHPMissingInt &&
		    anInfo.Param().GribIndicatorOfParameter() != kHPMissingInt)
		{
			// In radon table version is a parameter property, not a
			// producer property

			itsGrib->Message().Table2Version(anInfo.Param().GribTableVersion());
			itsGrib->Message().ParameterNumber(anInfo.Param().GribIndicatorOfParameter());
		}
		else if (anInfo.Producer().Id() != kHPMissingInt)  // no-database example has 999999 as producer
		{
			itsLogger.Warning("Parameter " + anInfo.Param().Name() + " does not have mapping for producer " +
			                  boost::lexical_cast<string>(anInfo.Producer().Id()) +
			                  " in radon, setting table2version to 203");
			itsGrib->Message().ParameterNumber(0);
			itsGrib->Message().Table2Version(203);
		}
	}
	else if (itsGrib->Message().Edition() == 2)
	{
		if (anInfo.Param().GribParameter() == kHPMissingInt)
		{
			itsLogger.Warning("Parameter information not found from radon for producer " +
			                  boost::lexical_cast<string>(anInfo.Producer().Id()) + ", name " + anInfo.Param().Name());
		}
		else
		{
			itsGrib->Message().ParameterNumber(anInfo.Param().GribParameter());
			itsGrib->Message().ParameterCategory(anInfo.Param().GribCategory());
			itsGrib->Message().ParameterDiscipline(anInfo.Param().GribDiscipline());
		}

		if (anInfo.Param().Aggregation().Type() != kUnknownAggregationType)
		{
			long templateNumber = 8;  // Average, accumulation, extreme values or other statistically processed values
			                          // at a horizontal level or in a horizontal layer in a continuous or
			                          // non-continuous time interval

			if (anInfo.ForecastType().Type() == kEpsPerturbation || anInfo.ForecastType().Type() == kEpsControl)
			{
				templateNumber = 11;  // Individual ensemble forecast, control and perturbed, at a horizontal level or
				                      // in a horizontal layer, in a continuous or non-continuous time interval.
			}

			itsGrib->Message().ProductDefinitionTemplateNumber(templateNumber);

			long type;

			switch (anInfo.Param().Aggregation().Type())
			{
				default:
				case kAverage:
					type = 0;
					break;
				case kAccumulation:
					type = 1;
					break;
				case kMaximum:
					type = 2;
					break;
				case kMinimum:
					type = 3;
					break;
			}

			itsGrib->Message().TypeOfStatisticalProcessing(type);
		}
	}
}

void grib::WriteLevel(info& anInfo)
{
	auto lev = anInfo.Level();

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

bool grib::ToFile(info& anInfo, string& outputFile, bool appendToFile)
{
	// Write only that data which is currently set at descriptors

	timer aTimer;
	aTimer.Start();

	if (anInfo.Grid()->Class() == kIrregularGrid && anInfo.Grid()->Type() != kReducedGaussian)
	{
		itsLogger.Error("Unable to write irregular grid of type " + HPGridTypeToString.at(anInfo.Grid()->Type()) +
		                " to grib");
		return false;
	}

	if (!itsWriteOptions.write_empty_grid)
	{
		if (anInfo.Data().MissingCount() == anInfo.Data().Size())
		{
			itsLogger.Debug("Not writing empty grid");
			return true;
		}
	}

	long edition = static_cast<long>(itsWriteOptions.configuration->OutputFileType());

	// Check levelvalue and forecast type since those might force us to change to grib2!

	HPForecastType forecastType = anInfo.ForecastType().Type();

	if (edition == 1 &&
	    (anInfo.Grid()->AB().size() > 255 || (forecastType == kEpsControl || forecastType == kEpsPerturbation)))
	{
		itsLogger.Trace("File type forced to GRIB2 (level value: " + to_string(anInfo.Level().Value()) +
		                ", forecast type: " + HPForecastTypeToString.at(forecastType) + ")");
		edition = 2;
		if (itsWriteOptions.configuration->FileCompression() == kNoCompression &&
		    itsWriteOptions.configuration->FileWriteOption() != kSingleFile)
		{
			outputFile += "2";
		}

		if (itsWriteOptions.configuration->FileCompression() == kGZIP)
		{
			outputFile.insert(outputFile.end() - 3, '2');
		}
		else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
		{
			outputFile.insert(outputFile.end() - 4, '2');
		}
		else if (itsWriteOptions.configuration->FileCompression() != kNoCompression)
		{
			itsLogger.Error("Unable to write to compressed grib. Unknown file compression: " +
			                HPFileCompressionToString.at(itsWriteOptions.configuration->FileCompression()));
			return false;
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

	WriteParameter(anInfo);

	// Area and Grid

	WriteAreaAndGrid(anInfo);

	// Time information

	WriteTime(anInfo);

	// Level

	WriteLevel(anInfo);

	// set to missing value to a large value to prevent it from mixing up with valid
	// values in the data

	itsGrib->Message().MissingValue(gribMissing);

	if (itsWriteOptions.use_bitmap && anInfo.Data().MissingCount() > 0)
	{
		itsGrib->Message().Bitmap(true);
	}

#if defined GRIB_WRITE_PACKED_DATA and defined HAVE_CUDA

	if (anInfo.Grid()->IsPackedData() && anInfo.Grid()->PackedData().ClassName() == "simple_packed")
	{
		itsLogger.Trace("Writing packed data");
		simple_packed* s = reinterpret_cast<simple_packed*>(&anInfo.Grid()->PackedData());

		itsGrib->Message().ReferenceValue(s->coefficients.referenceValue);
		itsGrib->Message().BinaryScaleFactor(static_cast<long>(s->coefficients.binaryScaleFactor));
		itsGrib->Message().DecimalScaleFactor(static_cast<long>(s->coefficients.decimalScaleFactor));
		itsGrib->Message().BitsPerValue(s->coefficients.bitsPerValue);

		itsLogger.Trace("bits per value: " + boost::lexical_cast<string>(itsGrib->Message().BitsPerValue()));
		itsLogger.Trace("decimal scale factor: " +
		                boost::lexical_cast<string>(itsGrib->Message().DecimalScaleFactor()));
		itsLogger.Trace("binary scale factor: " + boost::lexical_cast<string>(itsGrib->Message().BinaryScaleFactor()));
		itsLogger.Trace("reference value: " + boost::lexical_cast<string>(itsGrib->Message().ReferenceValue()));

		itsGrib->Message().PackedValues(s->data, anInfo.Data().Size(), 0, 0);
	}
	else
#endif
	{
		itsLogger.Trace("Writing unpacked data");

		/*
		 * Possible precipitation form value encoding must be done before determining
		 * bits per value, as the range of values is changed.
		 */

		const auto paramName = anInfo.Param().Name();
		const int precision = anInfo.Param().Precision();

		long bitsPerValue;

		if (edition == 2 && (paramName == "PRECFORM-N" || paramName == "PRECFORM2-N"))
		{
			// We take a copy of the data, because the values at cache should not change
			auto values = anInfo.Data().Values();

			EncodePrecipitationFormToGrib2(values);
			bitsPerValue = DetermineBitsPerValue(values, precision);

			// Change missing value 'nan' to a real fp value
			replace_if(values.begin(), values.end(), [](const double& v) { return IsMissingDouble(v); }, gribMissing);

			itsGrib->Message().BitsPerValue(bitsPerValue);
			itsGrib->Message().Values(values.data(), static_cast<long>(values.size()));
		}
		else
		{
			// In this branch no copy is made
			const auto& values = anInfo.Data().Values();
			bitsPerValue = DetermineBitsPerValue(values, precision);

			// Change missing value 'nan' to a real fp value
			anInfo.Data().MissingValue(gribMissing);

			itsGrib->Message().BitsPerValue(bitsPerValue);
			itsGrib->Message().Values(values.data(), static_cast<long>(values.size()));
		}

		itsLogger.Trace("Using " +
		                (precision == kHPMissingInt ? "maximum precision" : to_string(precision) + " decimals") + " (" +
		                to_string(bitsPerValue) + " bits) to store " + paramName);
	}

	// Return missing value to nan if info is recycled (luatool)
	anInfo.Data().MissingValue(MissingDouble());

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
		// TODO: check if parameter really is rotated uv
		if (edition == 1)
		{
			itsGrib->Message().ResolutionAndComponentFlags(128);  // 10000000
		}
		else
		{
			itsGrib->Message().ResolutionAndComponentFlags(48);  // 00110000
		}
	}

	vector<double> AB = anInfo.Grid()->AB();

	if (!AB.empty())
	{
		itsGrib->Message().NV(static_cast<long>(AB.size()));
		itsGrib->Message().PV(AB, AB.size());
	}

	if ((itsWriteOptions.configuration->FileCompression() == kGZIP ||
	     itsWriteOptions.configuration->FileCompression() == kBZIP2) &&
	    appendToFile)
	{
		itsLogger.Warning("Unable to append to a compressed file");
		appendToFile = false;
	}

	itsGrib->Message().Write(outputFile, appendToFile);

	aTimer.Stop();
	double duration = static_cast<double>(aTimer.GetTime());

	double bytes = static_cast<double>(boost::filesystem::file_size(outputFile));

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));

	string verb = (appendToFile ? "Appended to " : "Wrote ");
	itsLogger.Info(verb + "file '" + outputFile + "' (" + boost::lexical_cast<string>(speed) + " MB/s)");

	return true;
}

// ---------------------------------------------------------------------------

himan::earth_shape ReadEarthShape(const NFmiGribMessage& msg)
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

	return himan::earth_shape(a, b);
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
			newGrid = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
			reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(newGrid.get());

			gg->N(static_cast<int>(itsGrib->Message().GetLongKey("N")));
			gg->NumberOfPointsAlongParallels(itsGrib->Message().PL());
			gg->Nj(static_cast<size_t>(itsGrib->Message().SizeY()));
			gg->ScanningMode(m);

			ASSERT(m == kTopLeft);

			gg->TopLeft(firstPoint);

			point lastPoint(itsGrib->Message().X1(), itsGrib->Message().Y1());
			gg->BottomRight(lastPoint);

			break;
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
			rg->Di(round(100. * itsGrib->Message().XLengthInMeters()) / 100.);
			rg->Dj(round(100. * itsGrib->Message().YLengthInMeters()) / 100.);

			rg->ScanningMode(m);

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
			throw runtime_error(ClassName() + ": Unsupported grid type: " +
			                    boost::lexical_cast<string>(itsGrib->Message().NormalizedGridType()));
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
			itsLogger.Warning("Parameter name not found from " + HPDatabaseTypeToString.at(dbtype) + " for no_vers: " +
			                  boost::lexical_cast<string>(no_vers) + ", number: " +
			                  boost::lexical_cast<string>(number) + ", timeRangeIndicator: " +
			                  boost::lexical_cast<string>(timeRangeIndicator));
		}
		else
		{
			p.Name(parmName);
		}

		p.GribParameter(number);
		p.GribTableVersion(no_vers);

		// Determine aggregation

		aggregation a;
		a.TimeResolution(kHourResolution);

		switch (timeRangeIndicator)
		{
			case 0:  // forecast
			case 1:  // analysis
				break;

			case 3:  // average
				a.Type(kAverage);
				a.FirstTimeValue(static_cast<int>(itsGrib->Message().P1()));
				a.TimeResolutionValue(static_cast<int>(itsGrib->Message().P2() - itsGrib->Message().P1()));
				break;
			case 4:  // accumulation
				a.Type(kAccumulation);
				a.FirstTimeValue(static_cast<int>(itsGrib->Message().P1()));
				a.TimeResolutionValue(static_cast<int>(itsGrib->Message().P2() - itsGrib->Message().P1()));
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

		if (dbtype == kRadon)
		{
			auto r = GET_PLUGIN(radon);

			auto parminfo = r->RadonDB().GetParameterFromGrib2(prod.Id(), discipline, category, number,
			                                                   itsGrib->Message().NormalizedLevelType(),
			                                                   static_cast<double>(itsGrib->Message().LevelValue()));

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
			itsLogger.Warning("Parameter name not found from database for discipline: " +
			                  boost::lexical_cast<string>(discipline) + ", category: " +
			                  boost::lexical_cast<string>(category) + ", number: " +
			                  boost::lexical_cast<string>(number));
		}
		else
		{
			p.Name(parmName);
		}

		p.GribParameter(number);
		p.GribDiscipline(discipline);
		p.GribCategory(category);

		aggregation a;
		a.TimeResolution(kHourResolution);
		switch (itsGrib->Message().TypeOfStatisticalProcessing())
		{
			case 0:  // Average
				a.Type(kAverage);
				a.FirstTimeValue(static_cast<int>(itsGrib->Message().ForecastTime()));
				a.TimeResolutionValue(static_cast<int>(itsGrib->Message().LengthOfTimeRange()));
				break;

			case 1:  // Accumulation
				a.Type(kAccumulation);
				a.FirstTimeValue(static_cast<int>(itsGrib->Message().ForecastTime()));
				a.TimeResolutionValue(static_cast<int>(itsGrib->Message().LengthOfTimeRange()));
				break;

			case 2:  // Maximum
				a.Type(kMaximum);
				a.FirstTimeValue(static_cast<int>(itsGrib->Message().ForecastTime()));
				a.TimeResolutionValue(static_cast<int>(itsGrib->Message().LengthOfTimeRange()));
				break;

			case 3:  // Minimum
				a.Type(kMinimum);
				a.FirstTimeValue(static_cast<int>(itsGrib->Message().ForecastTime()));
				a.TimeResolutionValue(static_cast<int>(itsGrib->Message().LengthOfTimeRange()));
				break;
		}

		if (a.TimeResolutionValue() != kHPMissingInt)
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
	string dataDate = boost::lexical_cast<string>(itsGrib->Message().DataDate());

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
			timeResolution = kMinuteResolution;
			break;

		default:
			itsLogger.Warning("Unsupported unit of time range: " + boost::lexical_cast<string>(timeResolution));
			break;
	}

	t.StepResolution(timeResolution);

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
				itsLogger.Fatal("Unsupported level type for no database mode: " +
				                boost::lexical_cast<string>(gribLevel));
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
			prod.Id(boost::lexical_cast<int>(prodInfo["id"]));
		}
		else
		{
			if (centre == 98 &&
			    (process == 146 || process == 145 || process == 144 || process == 143 || process == 142))
			{
				// Older ECMWF forecast
				prod.Id(131);
			}
			else
			{
				itsLogger.Warning("Producer information not found from database for centre " + to_string(centre) +
				                  ", process " + to_string(process) + " type " + to_string(typeId));
			}
		}
	}

	return prod;
}

void grib::ReadData(info_t newInfo, bool readPackedData) const
{
	auto& dm = newInfo->Data();

	bool decodePrecipitationForm = false;

#if defined GRIB_READ_PACKED_DATA && defined HAVE_CUDA

	const auto paramName = newInfo->Param().Name();
	long producerId = newInfo->Producer().Id();

	if (itsGrib->Message().Edition() == 2 && (paramName == "PRECFORM-N" || paramName == "PRECFORM2-N") &&
	    (producerId == 230 || producerId == 240 || producerId == 243 || producerId == 250 || producerId == 260))
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

		auto packed =
		    unique_ptr<simple_packed>(new simple_packed(bpv, util::ToPower(bsf, 2), util::ToPower(-dsf, 10), rv));

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

			itsLogger.Trace("Retrieved " + boost::lexical_cast<string>(len) + " bytes of packed data from grib");
		}
		else
		{
			itsLogger.Trace("Grid is constant or empty");
		}

		if (itsGrib->Message().Bitmap())
		{
			size_t bitmap_len = itsGrib->Message().BytesLength("bitmap");
			size_t bitmap_size = static_cast<size_t>(ceil(static_cast<double>(bitmap_len) / 8));

			itsLogger.Trace("Grib has bitmap, length " + boost::lexical_cast<string>(bitmap_len) + " size " +
			                boost::lexical_cast<string>(bitmap_size) + " bytes");

			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&unpackedBitmap), bitmap_len * sizeof(int)));

			unsigned char* bitmap = new unsigned char[bitmap_size];

			itsGrib->Message().Bytes("bitmap", bitmap);

			UnpackBitmap(bitmap, unpackedBitmap, bitmap_size, bitmap_len);

			packed->Bitmap(unpackedBitmap, bitmap_len);

			delete[] bitmap;
		}

		newInfo->Grid()->PackedData(move(packed));
	}
	else
#endif
	{
		dm.MissingValue(gribMissing);
		size_t len = itsGrib->Message().ValuesLength();

		itsGrib->Message().GetValues(dm.ValuesAsPOD(), &len);
		dm.MissingValue(MissingDouble());

		if (decodePrecipitationForm)
		{
			DecodePrecipitationFormFromGrib2(dm.Values());
		}

		itsLogger.Trace("Retrieved " + std::to_string(len * sizeof(double)) + " bytes of unpacked data from grib");
	}
}

bool grib::CreateInfoFromGrib(const search_options& options, bool readPackedData, bool readIfNotMatching,
                              shared_ptr<info> newInfo) const
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

			if (optsTime.StepResolution() != t.StepResolution())
			{
				itsLogger.Trace("Step resolution: " + string(HPTimeResolutionToString.at(optsTime.StepResolution())) +
				                " (requested) vs " + string(HPTimeResolutionToString.at(t.StepResolution())) +
				                " (found)");
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

			if (options.level.Type() != l.Type())
			{
				itsLogger.Trace("Type: " + string(HPLevelTypeToString.at(options.level.Type())) + " (requested) vs " +
				                string(HPLevelTypeToString.at(l.Type())) + " (found)");
			}

			if (options.level.Value() != l.Value())
			{
				itsLogger.Trace("Value: " + string(boost::lexical_cast<string>(options.level.Value())) +
				                " (requested) vs " + string(boost::lexical_cast<string>(l.Value())) + " (found)");
			}

			if (options.level.Value2() != l.Value2())
			{
				itsLogger.Trace("Value2: " + string(boost::lexical_cast<string>(options.level.Value2())) +
				                " (requested) vs " + string(boost::lexical_cast<string>(l.Value2())) + " (found)");
			}

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

			if (options.ftype.Type() != ty.Type())
			{
				itsLogger.Trace("Type: " + string(HPForecastTypeToString.at(options.ftype.Type())) +
				                " (requested) vs " + string(HPForecastTypeToString.at(ty.Type())) + " (found)");
			}

			if (options.ftype.Value() != ty.Value())
			{
				itsLogger.Trace("Value: " + string(boost::lexical_cast<string>(options.ftype.Value())) +
				                " (requested) vs " + string(boost::lexical_cast<string>(ty.Value())) + " (found)");
			}

			return false;
		}
	}

	// END VALIDATION OF SEARCH PARAMETERS

	newInfo->Producer(prod);

	vector<param> theParams;

	theParams.push_back(p);

	newInfo->Params(theParams);

	vector<forecast_time> theTimes;

	theTimes.push_back(t);

	newInfo->Times(theTimes);

	vector<level> theLevels;

	theLevels.push_back(l);

	newInfo->Levels(theLevels);

	vector<forecast_type> theForecastTypes;

	theForecastTypes.push_back(ty);

	newInfo->ForecastTypes(theForecastTypes);

	unique_ptr<grid> newGrid = ReadAreaAndGrid();

	ASSERT(newGrid);

	std::vector<double> ab;

	if (l.Type() == himan::kHybrid)
	{
		long nv = itsGrib->Message().NV();

		if (nv > 0)
		{
			ab = itsGrib->Message().PV();
		}
	}

	newGrid->AB(ab);

	newInfo->Create(newGrid.get(), true);

	// Set descriptors

	newInfo->Param(p);
	newInfo->Time(t);
	newInfo->Level(l);
	newInfo->ForecastType(ty);

	/*
	 * Read data from grib *
	 */

	ReadData(newInfo, readPackedData);

	if (!dataIsValid)
	{
		return false;
	}

	newInfo->First();
	return true;
}

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options,
                                               bool readContents, bool readPackedData, bool readIfNotMatching) const
{
	vector<shared_ptr<himan::info>> infos;

	if (!itsGrib->Open(theInputFile))
	{
		itsLogger.Error("Opening file '" + theInputFile + "' failed");
		return infos;
	}

	int foundMessageNo = 0;

	if (options.prod.Centre() == kHPMissingInt && options.configuration->DatabaseType() != kNoDatabase)
	{
		itsLogger.Error("Process and centre information for producer " +
		                boost::lexical_cast<string>(options.prod.Id()) + " are undefined");
		return infos;
	}

	timer aTimer;
	aTimer.Start();

	while (itsGrib->NextMessage())
	{
		foundMessageNo++;
		auto newInfo = make_shared<info>();

		if (CreateInfoFromGrib(options, readPackedData, readIfNotMatching, newInfo) || readIfNotMatching)
		{
			infos.push_back(newInfo);
			newInfo->First();

			aTimer.Stop();

			if (!readIfNotMatching)
				break;  // We found what we were looking for
		}
	}

	long duration = aTimer.GetTime();
	long bytes = boost::filesystem::file_size(theInputFile);

	double speed = floor((static_cast<double>(bytes) / 1024. / 1024.) / (static_cast<double>(duration) / 1000.));

	itsLogger.Debug("Read file '" + theInputFile + "' (" + boost::lexical_cast<string>(speed) + " MB/s)");

	return infos;
}

vector<shared_ptr<himan::info>> grib::FromIndexFile(const string& theInputFile, const search_options& options,
                                                    bool readContents, bool readPackedData,
                                                    bool readIfNotMatching) const
{
	vector<shared_ptr<himan::info>> infos;

	if (!itsGrib->Open(theInputFile))
	{
		itsLogger.Error("Opening file '" + theInputFile + "' failed");
		return infos;
	}

	if (options.prod.Centre() == kHPMissingInt)
	{
		itsLogger.Error("Process and centre information for producer " +
		                boost::lexical_cast<string>(options.prod.Id()) + " are undefined");
		return infos;
	}

	timer aTimer;
	aTimer.Start();

	// TODO need to check what happens when multiple idx files or idx + grib files are provided as input.
	if (itsGrib->Message(OptionsToKeys(options)))
	{
		auto newInfo = make_shared<info>();
		if (CreateInfoFromGrib(options, readPackedData, readIfNotMatching, newInfo))
		{
			infos.push_back(newInfo);
			newInfo->First();
		}
	}
	aTimer.Stop();

	long duration = aTimer.GetTime();

	itsLogger.Debug("Read message using grib index file '" + theInputFile + "' in " +
	                boost::lexical_cast<std::string>(duration) + " ms");

	return infos;
}

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

std::map<string, long> grib::OptionsToKeys(const search_options& options) const
{
	// indicator of Parameter is not necessarily provided in search_options param
	// look this information up from database instead

	map<string, string> param;

	if (options.configuration->DatabaseType() == kRadon)
	{
		auto r = GET_PLUGIN(radon);
		auto levelInfo =
		    r->RadonDB().GetLevelFromDatabaseName(boost::to_upper_copy(HPLevelTypeToString.at(options.level.Type())));

		ASSERT(levelInfo.size());
		param = r->RadonDB().GetParameterFromDatabaseName(options.prod.Id(), options.param.Name(),
		                                                  stoi(levelInfo["id"]), options.level.Value());
	}

	auto time = options.time;

	std::map<string, long> theKeyValueMap;

	theKeyValueMap["level"] = static_cast<long>(options.level.Value());
	theKeyValueMap["step"] = static_cast<long>(options.time.Step());
	theKeyValueMap["centre"] = static_cast<long>(options.prod.Centre());
	theKeyValueMap["generatingProcessIdentifier"] = static_cast<long>(options.prod.Process());
	theKeyValueMap["date"] = stol(time.OriginDateTime().String("%Y%m%d"));
	theKeyValueMap["time"] = stol(time.OriginDateTime().String("%H%M"));

	//	if (param["version"] == "1")
	{
		theKeyValueMap["indicatorOfTypeOfLevel"] = static_cast<long>(options.level.Type());
		theKeyValueMap["indicatorOfParameter"] = stol(param["grib1_number"]);
	}
	/*
	    else if (param["version"] == "2")
	    {
	        // TODO check if this is giving correct type number (Grib2 != Grib1)
	        theKeyValueMap["typeOfFirstFixedSurface"] = static_cast<long>(options.level.Type());
	        theKeyValueMap["discipline"] = stol(param["grib2_discipline"]);
	        theKeyValueMap["parameterCategory"] = stol(param["grib2_category"]);
	        theKeyValueMap["parameterNumber"] = stol(param["grib2_number"]);
	    }
	*/
	return theKeyValueMap;
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

void EncodePrecipitationFormToGrib2(vector<double>& arr)
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

void DecodePrecipitationFormFromGrib2(vector<double>& arr)
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
