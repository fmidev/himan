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

#ifndef __CUDACC__
#include <ostream>
#include <memory>
#include <boost/assign/list_of.hpp>
#include <boost/unordered_map.hpp>

namespace ba = boost::assign;

#endif

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
	kNetCDF
};

#ifndef __CUDACC__
const boost::unordered_map<HPFileType,const char*> HPFileTypeToString = ba::map_list_of
		(kUnknownFile, "unknown")
		(kGRIB1, "grib edition 1")
		(kGRIB2, "grib edition 2")
		(kGRIB, "grib edition 1 or 2")
		(kQueryData, "QueryData")
		(kNetCDF, "NetCDF");
#endif

// Define supported projections
// Values equal to those in newbase

enum HPProjectionType
{
	kUnknownProjection = 0,
	kLatLonProjection = 10,
	kRotatedLatLonProjection = 11,
	kStereographicProjection = 13
};

#ifndef __CUDACC__
const boost::unordered_map<HPProjectionType,const char*> HPProjectionTypeToString = ba::map_list_of
		(kLatLonProjection, "ll")
		(kRotatedLatLonProjection, "rll")
		(kStereographicProjection, "polster");
#endif

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
	kKgm2 // kg/m^2
};

enum HPInterpolationMethod
{
	kUnknownInterpolationMethod = 0,
	kBiLinear = 1,
	kNearestPoint = 2,
};

#ifndef __CUDACC__
const boost::unordered_map<HPInterpolationMethod,const char*> HPInterpolationMethodToString = ba::map_list_of
		(kUnknownInterpolationMethod, "unknown")
		(kBiLinear, "bi-linear")
		(kNearestPoint, "nearest point")
		;
#endif

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

#ifndef __CUDACC__
const boost::unordered_map<HPLevelType,const char*> HPLevelTypeToString = ba::map_list_of
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
#endif

enum HPFileWriteOption
{
	kUnknownFileWriteOption = 0,
	kSingleFile,
	kMultipleFiles,
	kDatabase
};

#ifndef __CUDACC__
const boost::unordered_map<HPFileWriteOption,const char*> HPFileWriteOptionToString = ba::map_list_of
		(kUnknownFileWriteOption, "unknown")
		(kSingleFile, "single file only")
		(kMultipleFiles, "multiple files")
		(kDatabase, "write to database")
		;
#endif

// Values match to newbase

/**
 * @enum HPScanningMode
 *
 * @brief Describe different data scanning modes (ie in which direction the data is read)
 */

enum HPScanningMode
{
	kUnknownScanningMode = 0,
	kTopLeft = 17, 		// +x-y
	kTopRight = 18,		// -x-y
	kBottomLeft = 33,	// +x+y
	kBottomRight = 34,	// -x+y

};

#ifndef __CUDACC__
const boost::unordered_map<HPScanningMode,const char*> HPScanningModeToString = ba::map_list_of
		(kUnknownScanningMode, "unknown")
		(kTopLeft, "+x-y")
		(kTopRight, "+x+y")
		(kBottomLeft, "+x+y")
		(kBottomRight, "-x-y");
#endif

enum HPLevelOrder
{
	kUnknownLevelOrder = 0,
	kTopToBottom = 1,
	kBottomToTop = 2 
};

#ifndef __CUDACC__
const boost::unordered_map<HPLevelOrder,const char*> HPLevelOrderToString = ba::map_list_of
	(kUnknownLevelOrder, "unknown")
	(kTopToBottom, "top to bottom")
	(kBottomToTop, "bottom to top");
#endif

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
	kLocationDimension
};

#ifndef __CUDACC__
const boost::unordered_map<HPDimensionType,const char*> HPDimensionTypeToString = ba::map_list_of
		(kUnknownDimension, "unknown")
		(kTimeDimension, "time dimension")
		(kLevelDimension, "level dimension")
		(kParamDimension, "param dimension")
		(kLocationDimension, "location dimension");
#endif

enum HPTimeResolution
{
	kUnknownTimeResolution = 0,
	kHourResolution,
	kMinuteResolution
};

#ifndef __CUDACC__
const boost::unordered_map<HPTimeResolution,const char*> HPTimeResolutionToString = ba::map_list_of
		(kUnknownTimeResolution, "unknown")
		(kHourResolution, "hour")
		(kMinuteResolution, "minute");
#endif

enum HPPackingType
{
	kUnknownPackingType = 0,
	kUnpacked,
	kSimplePacking
};

enum HPAggregationType
{
	kUnknownAggregationType = 0,
	kAverage,
	kAccumulation,
	kMaximum,
	kMinimum,
	kDifference,

	/*
	 * Parameters refer to another parameter minimum and maximum value, used in modifier.
	 * Not a perfect way to describe the relationship between two separate parameters
	 * but as we don't have combined parameters this is maybe the nicest way to do it.
	*/

	kExternalMinimum,
	kExternalMaximum

};

#ifndef __CUDACC__
const boost::unordered_map<HPAggregationType,const char*> HPAggregationTypeToString = ba::map_list_of
		(kUnknownAggregationType, "unknown")
		(kAverage, "average")
		(kAccumulation, "accumulation")
		(kMaximum, "maximum")
		(kMinimum, "minimum")
		(kDifference, "difference")
		(kExternalMinimum, "external minimum")
		(kExternalMaximum, "external maximum");
#endif

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

#ifndef __CUDACC__
const boost::unordered_map<HPModifierType,const char*> HPModifierTypeToString = ba::map_list_of
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
#endif

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

#ifndef __CUDACC__
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
#endif

enum HPGridType
{
	kUnknownGridType = 0,
	kRegularGrid,
	kIrregularGrid
};

enum HPDatabaseType
{
	kUnknownDatabaseType = 0,
	kNeons,
	kRadon,
	kNeonsAndRadon
};

#ifndef __CUDACC__
const boost::unordered_map<HPGridType,const char*> HPGridTypeToString = ba::map_list_of
		(kUnknownGridType, "unknown")
		(kRegularGrid, "regular")
		(kIrregularGrid, "irregular");
#endif

/**
 * @struct HPVersionNumber
 *
 * @brief Simple struct to hold plugin version number with major and minor digit.
 *
 */

#ifndef __CUDACC__
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
#endif

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
	 * @brief Specific gas constant of dry air (J / K kg)
	 */

	const double kRd = 287;

	/**
	 * @brief Specific heat of dry air at constant pressure (J / K kg)
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

	const double kRCp = 0.286;

	/**
 	 * @brief von Karman constant
 	 */

	const double kK = 0.41;	

	/**
	 * @brief Gas constant for water divided by latent heat (used in dewpoint)
	 */
	
	const double kRw_div_L = himan::constants::kRw / himan::constants::kL;

} // namespace constants

} // namespace himan

#endif /* HIMAN_COMMON_H */
