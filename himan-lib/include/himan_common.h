/**
 * @file himan_common.h
 *
 * @date Nov 17, 2012
 * @author partio
 */

#ifndef HIMAN_COMMON_H
#define HIMAN_COMMON_H

/**
 * @file himan_common.h
 *
 * Definitions common to all classes. Mostly enums.
 *
 */

// Work around "passing 'T' chooses 'int' over 'unsigned int'" warnings when T
// is an enum type:

#if !defined __clang__ && defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)) && ! __CUDACC__
#pragma GCC diagnostic ignored "-Wsign-promo"
#endif

#include <ostream>
#include <memory>
#include <boost/assign/list_of.hpp>
#include <boost/unordered_map.hpp>

namespace ba = boost::assign;

namespace himan
{

// Define some constants

const int kHPMissingInt = 999999;
const double kHPMissingValue = -999.;
const double kFloatMissing = 32700.; // From newbase

// Define different plugin types

enum HPPluginClass
{
	kUnknownPlugin = 0,
	kCompiled,
	kAuxiliary
};

// Define different logging levels

enum HPDebugState
{
	kTraceMsg = 0,
	kDebugMsg,
	kInfoMsg,
	kWarningMsg,
	kErrorMsg,
	kFatalMsg
};

// Define supported file types

enum HPFileType
{
	kUnknownFile = 0,
	kGRIB1,
	kGRIB2,
	kGRIB, // when we don't know if its 1 or 2
	kQueryData,
	kNetCDF,
	kCSV
};

const boost::unordered_map<HPFileType,std::string> HPFileTypeToString = ba::map_list_of
		(kUnknownFile, "unknown")
		(kGRIB1, "grib edition 1")
		(kGRIB2, "grib edition 2")
		(kGRIB, "grib edition 1 or 2")
		(kQueryData, "QueryData")
		(kNetCDF, "NetCDF")
		(kCSV, "CSV")
		;

// Defined external compression types

enum HPFileCompression
{
	kUnknownCompression = 0,
	kNoCompression,
	kGZIP,
	kBZIP2
};

const boost::unordered_map<HPFileCompression,std::string> HPFileCompressionToString = ba::map_list_of
				(kUnknownCompression, "unknown compression")
                (kNoCompression, "no compression")
                (kGZIP, "gzip compressed")
                (kBZIP2, "bzip2 compressed");

// Define supported projections
// Values equal to those in newbase

enum HPProjectionType
{
	kUnknownProjection = 0,
	kLatLonProjection = 10,
	kRotatedLatLonProjection = 11,
	kStereographicProjection = 13
};

const boost::unordered_map<HPProjectionType,std::string> HPProjectionTypeToString = ba::map_list_of
		(kUnknownProjection, "unknown projection")
		(kLatLonProjection, "ll")
		(kRotatedLatLonProjection, "rll")
		(kStereographicProjection, "polster");

// Define supported parameter units

enum HPParameterUnit
{
	kUnknownUnit = 0,
	kPa,
	kK,
	kC,
	kPas, // Pa/s
	kHPa,
	kPrcnt,
	kMs, // m/s
	kM, // meters
	kMm, // millimeters
	kGph, // geopotential height, m^2 / s^2
	kKgkg, // kg/kg
	kJm2, // J per square meter
	kKgm2, // kg/m^2
	kS2 // 1/s^2
};

enum HPInterpolationMethod
{
	kUnknownInterpolationMethod = 0,
	kBiLinear = 1,
	kNearestPoint = 2,
	kNearestPointValue // http://arxiv.org/pdf/1211.1768.pdf
};

const boost::unordered_map<HPInterpolationMethod,std::string> HPInterpolationMethodToString = ba::map_list_of
		(kUnknownInterpolationMethod, "unknown")
		(kBiLinear, "bilinear")
		(kNearestPoint, "nearest point")
		(kNearestPointValue, "nearest point value")
		;

const boost::unordered_map<std::string,HPInterpolationMethod> HPStringToInterpolationMethod = ba::map_list_of
		("unknown",kUnknownInterpolationMethod)
		("bilinear",kBiLinear)
		("nearest point",kNearestPoint)
		("nearest point value",kNearestPointValue)
		;


enum HPLevelType
{
	kUnknownLevel = 0,
	kGround = 1,
	kTopOfAtmosphere = 8,
	kLake = 21,
	kPressure = 100,
	kMeanSea = 102,
	kAltitude = 103,
	kHeight = 105,
	kHybrid = 109,
	kGndLayer = 112,
	kDepth = 160,
	kEntireAtmosphere = 200,
	kEntireOcean = 201
};

const boost::unordered_map<HPLevelType,std::string> HPLevelTypeToString = ba::map_list_of
		(kUnknownLevel, "unknown")
		(kGround, "ground")
		(kPressure, "pressure")
		(kMeanSea, "meansea")
		(kAltitude, "altitude")
		(kHeight, "height")
		(kHybrid, "hybrid")
		(kGndLayer, "gndlayer")
		(kDepth, "depth")
		(kTopOfAtmosphere, "top")
		(kEntireAtmosphere, "entatm")
		(kEntireOcean, "entocean")
		(kLake, "lake")
		;

const boost::unordered_map<std::string,HPLevelType> HPStringToLevelType = ba::map_list_of
		("unknown",kUnknownLevel)
		("ground",kGround)
		("pressure",kPressure)
		("meansea",kMeanSea)
		("altitude",kAltitude)
		("height",kHeight)
		("hybrid",kHybrid)
		("gndlayer",kGndLayer)
		("depth",kDepth)
		("top",kTopOfAtmosphere)
		("entatm", kEntireAtmosphere)
		("entocean", kEntireOcean)
		("lake", kLake)
		;

enum HPFileWriteOption
{
	kUnknownFileWriteOption = 0,
	kSingleFile,
	kMultipleFiles,
	kDatabase
};

const boost::unordered_map<HPFileWriteOption,std::string> HPFileWriteOptionToString = ba::map_list_of
		(kUnknownFileWriteOption, "unknown")
		(kSingleFile, "single file only")
		(kMultipleFiles, "multiple files")
		(kDatabase, "write to database")
		;


/**
 * @enum HPScanningMode
 *
 * @brief Describe different data scanning modes (ie in which direction the data is read)
 * 
 * Values match to newbase.
 */

enum HPScanningMode
{
	kUnknownScanningMode = 0,
	kTopLeft = 17, 		// +x-y
	kTopRight = 18,		// -x-y
	kBottomLeft = 33,	// +x+y
	kBottomRight = 34,	// -x+y

};

const boost::unordered_map<HPScanningMode,std::string> HPScanningModeToString = ba::map_list_of
		(kUnknownScanningMode, "unknown")
		(kTopLeft, "+x-y")
		(kTopRight, "+x+y")
		(kBottomLeft, "+x+y")
		(kBottomRight, "-x-y");

enum HPLevelOrder
{
	kUnknownLevelOrder = 0,
	kTopToBottom = 1,
	kBottomToTop = 2 
};

const boost::unordered_map<HPLevelOrder,std::string> HPLevelOrderToString = ba::map_list_of
	(kUnknownLevelOrder, "unknown")
	(kTopToBottom, "top to bottom")
	(kBottomToTop, "bottom to top");

enum HPExceptionType
{
	kUnknownException = 0,
	kFileMetaDataNotFound,
	kFileDataNotFound,
	kFunctionNotImplemented
};

/**
 * @enum HPDimensionType
 *
 * @brief Define all dimension types
 *
 * When looping over data, we can choose between a few dimensions.
 */
enum HPDimensionType
{
	kUnknownDimension = 0,
	kTimeDimension,
	kLevelDimension,
	kParamDimension,
	kLocationDimension,
	kForecastTypeDimension
};

const boost::unordered_map<HPDimensionType,std::string> HPDimensionTypeToString = ba::map_list_of
		(kUnknownDimension, "unknown")
		(kTimeDimension, "time dimension")
		(kLevelDimension, "level dimension")
		(kParamDimension, "param dimension")
		(kLocationDimension, "location dimension")
		(kForecastTypeDimension, "forecast type dimension")
		;

const boost::unordered_map<std::string,HPDimensionType> HPStringToDimensionType = ba::map_list_of
		("unknown",kUnknownDimension)
		("time",kTimeDimension)
		("level",kLevelDimension)
		("param",kParamDimension)
		("location",kLocationDimension)
		("forecast_type",kForecastTypeDimension)
		;

enum HPTimeResolution
{
	kUnknownTimeResolution = 0,
	kHourResolution,
	kMinuteResolution
};

const boost::unordered_map<HPTimeResolution,std::string> HPTimeResolutionToString = ba::map_list_of
		(kUnknownTimeResolution, "unknown")
		(kHourResolution, "hour")
		(kMinuteResolution, "minute");

enum HPPackingType
{
	kUnknownPackingType = 0,
	kUnpacked,
	kSimplePacking,
	kJpegPacking
};

enum HPAggregationType
{
	kUnknownAggregationType = 0,
	kAverage,
	kAccumulation,
	kMaximum,
	kMinimum,
	kDifference
};

const boost::unordered_map<HPAggregationType,std::string> HPAggregationTypeToString = ba::map_list_of
		(kUnknownAggregationType, "unknown")
		(kAverage, "average")
		(kAccumulation, "accumulation")
		(kMaximum, "maximum")
		(kMinimum, "minimum")
		(kDifference, "difference")
;

enum HPModifierType
{
	kUnknownModifierType = 0,
	kAverageModifier,
	kAccumulationModifier,
	kMaximumModifier,
	kMinimumModifier,
	kDifferenceModifier,
	kMaximumMinimumModifier,
	kCountModifier,
	kFindHeightModifier,
	kFindValueModifier,
	kIntegralModifier,
	kPlusMinusAreaModifier
};

const boost::unordered_map<HPModifierType,std::string> HPModifierTypeToString = ba::map_list_of
		(kUnknownModifierType, "unknown modifier")
		(kAverageModifier, "average modifier")
		(kAccumulationModifier, "accumulation modifier")
		(kMaximumModifier, "maximum modifier")
		(kMinimumModifier, "minimum modifier")
		(kDifferenceModifier, "difference modifier")
		(kMaximumMinimumModifier, "maximum minimum modifier")
		(kCountModifier, "count modifier")
		(kFindHeightModifier, "find height modifier")
		(kFindValueModifier, "find value modifier")
		(kIntegralModifier, "integral modifier")
		(kPlusMinusAreaModifier, "plus minus area modifier")
;

/// Precipitation forms as agreed by FMI

enum HPPrecipitationForm
{
	kDrizzle = 0,
	kRain,
	kSleet,
	kSnow,
	kFreezingDrizzle,
	kFreezingRain,
	kGraupel,
	kHail,
	kUnknownPrecipitationForm = 10
};

const boost::unordered_map<HPPrecipitationForm,const char*> HPPrecipitationFormToString = ba::map_list_of
		(kDrizzle, "drizzle")
		(kRain, "rain")
		(kSleet, "sleet")
		(kSnow, "snow")
		(kFreezingDrizzle, "freezing drizzle")
		(kFreezingRain, "freezing rain")
		(kGraupel, "graupel")
		(kHail, "hail")
		(kUnknownPrecipitationForm, "unknown");

enum HPGridType
{
	kUnknownGridType = 0,
	kRegularGrid,
	kIrregularGrid
};

const boost::unordered_map<HPGridType,std::string> HPGridTypeToString = ba::map_list_of
		(kUnknownGridType, "unknown")
		(kRegularGrid, "regular")
		(kIrregularGrid, "irregular");

enum HPDatabaseType
{
	kUnknownDatabaseType = 0,
	kNeons,
	kRadon,
	kNeonsAndRadon
};

const boost::unordered_map<HPDatabaseType,std::string> HPDatabaseTypeToString = ba::map_list_of
		(kUnknownDatabaseType, "unknown")
		(kNeons, "neons")
		(kRadon, "radon")
		(kNeonsAndRadon, "neons and radon");

enum HPForecastType
{
	kUnknownType = 0,
	kDeterministic,
	kAnalysis,
	kEpsPerturbation = 3,
	kEpsControl = 4
};

const boost::unordered_map<HPForecastType,std::string> HPForecastTypeToString = ba::map_list_of
		(kUnknownType, "unknown")
		(kDeterministic, "deterministic")
		(kAnalysis, "analysis")
		(kEpsControl, "eps control")
		(kEpsPerturbation, "eps perturbation")
		;

const boost::unordered_map<std::string,HPForecastType> HPStringToForecastType = ba::map_list_of
		("unknown",kUnknownType)
		("deterministic",kDeterministic)
		("analysis", kAnalysis)
		("eps control",kEpsControl)
		("eps perturbation",kEpsPerturbation)
		;

/**
 * @struct HPVersionNumber
 *
 * @brief Simple struct to hold plugin version number with major and minor digit.
 *
 */

struct HPVersionNumber
{
	unsigned short itsMajorVersion;
	unsigned short itsMinorVersion;

	unsigned short Minor()
	{
		return itsMinorVersion;
	}
	unsigned short Major()
	{
		return itsMajorVersion;
	}

	HPVersionNumber(unsigned short theMajorVersion, unsigned short theMinorVersion)
	{
		itsMinorVersion = theMinorVersion;
		itsMajorVersion = theMajorVersion;
	}

	std::ostream& Write(std::ostream& file) const
	{
		file << itsMajorVersion << "." << itsMinorVersion;
		return file;
	}

};

inline
std::ostream& operator<<(std::ostream& file, const HPVersionNumber& vers)
{
	return vers.Write(file);
}

namespace constants {

	/**
	 * @brief Zero degrees celsius in Kelvins
	 */

	const double kKelvin = 273.15;

	/**
	 * @brief Gas constant for water vapor (J / K kg)
	 */

	const double kRw = 461.5;

	/**
	 * @brief Latent heat for water vaporization or condensation (J / kg)
	 *
	 * http://glossary.ametsoc.org/wiki/Latent_heat
	 */

	const double kL = 2.5e6;

	/**
	 * @brief One radian in degrees (180 / PI)
	 */

	const double kRad = 57.29577951307855;

	/**
	 * @brief One degree in radians (PI / 180)
	 */

	const double kDeg = 0.017453292519944;

	/**
	 * @brief Dimensionless ratio of the specific gas constant of dry air to the specific gas constant for water vapor, ie kRd / kRw
	 *
	 *
	 * http://en.wikipedia.org/wiki/Lapse_rate#Saturated_adiabatic_lapse_rate
	 */

	const double kEp = 0.622;

	/**
	 * @brief Specific gas constant of dry air (J / K / kg)
	 */

	const double kRd = 287;

	/**
	 * @brief Specific heat of dry air at constant pressure (J / K / kg)
	 */

	const double kCp = 1003.5;

	/**
	 * @brief Gravity constant approximation (m/s^2)
	 */

	const double kG = 9.80665;

	/**
	 * @brief Inverse g constant (ie. 1/g)
	 */

	const double kIg = 0.10197;

	/**
	 * @brief R / Cp, where R = Rd. Unitless quantity.
	 */

	const double kRd_div_Cp = kRd / kCp;

	/**
 	 * @brief von Karman constant
 	 */

	const double kK = 0.41;	

	/**
	 * @brief Gas constant for water divided by latent heat (used in dewpoint)
	 */
	
	const double kRw_div_L = himan::constants::kRw / himan::constants::kL;

	/**
	 * @brief Mean radius of the earth in meters
	 */

	const double kR = 6371009;
	
	/**
	 * @brief Molar mass of water in g/mol
	 * 
	 * See also kEp = kMW / kMA
	 */
	
	const double kMW = 18.01528;

	/**
	 * @brief Molar mass of dry air in g/mol
	 * 
	 * See also kEp = kMW / kMA
	 */

	const double kMA = 28.9644;

} // namespace constants

} // namespace himan

#endif /* HIMAN_COMMON_H */
