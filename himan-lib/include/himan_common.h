/*
 * himan_common.h
 *
 *  Created on: Nov 17, 2012
 *	  Author: partio
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

namespace himan
{

using boost::assign::map_list_of;

// Define some constants

const int kHPMissingInt = 999999;
const double kHPMissingValue = -999.;

// const float kFloatEpsilon = std::numeric_limits<float>::epsilon();

// Define different plugin types

enum HPPluginClass
{
	kUnknownPlugin = 0,
	kInterpreted,
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

const boost::unordered_map<HPFileType,const char*> HPFileTypeToString = map_list_of
		(kUnknownFile, "unknown")
		(kGRIB1, "grib edition 1")
		(kGRIB2, "grib edition 2")
		(kGRIB, "grib edition 1 or 2")
		(kQueryData, "QueryData")
		(kNetCDF, "NetCDF");

// Define supported projections
// Values equal to those in newbase

enum HPProjectionType
{
	kUnknownProjection = 0,
	kLatLonProjection = 10,
	kRotatedLatLonProjection = 11,
	kStereographicProjection = 13
};

const boost::unordered_map<HPProjectionType,const char*> HPProjectionTypeToString = map_list_of
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
	kKgm2 // kg/m^2
};

enum HPLevelType
{
	kUnknownLevel = 0,
	kGround = 1,
	kPressure = 100,
	kMeanSea = 102,
	kHeight = 105,
	kHybrid = 109,
	kGndLayer = 112,
	kDepth = 160
};

const boost::unordered_map<HPLevelType,const char*> HPLevelTypeToString = map_list_of
		(kUnknownLevel, "unknown")
		(kGround, "ground")
		(kPressure, "pressure")
		(kMeanSea, "meansea")
		(kHeight, "height")
		(kHybrid, "hybrid")
		(kGndLayer, "gndlayer")
		(kDepth, "depth")
		;

enum HPFileWriteOption
{
	kUnknownFileWriteOption = 0,
	kSingleFile,
	kMultipleFiles,
	kNeons
};

const boost::unordered_map<HPFileWriteOption,const char*> HPFileWriteOptionToString = map_list_of
		(kUnknownFileWriteOption, "unknown")
		(kSingleFile, "single file only")
		(kMultipleFiles, "multiple files")
		(kNeons, "write to neons")
		;

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

const boost::unordered_map<HPScanningMode,const char*> HPScanningModeToString = map_list_of
		(kTopLeft, "+x-y")
		(kTopRight, "+x+y")
		(kBottomLeft, "+x+y")
		(kBottomRight, "-x-y");

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

const boost::unordered_map<HPDimensionType,const char*> HPDimensionTypeToString = map_list_of
		(kUnknownDimension, "unknown")
		(kTimeDimension, "time dimension")
		(kLevelDimension, "level dimension")
		(kParamDimension, "param dimension")
		(kLocationDimension, "location dimension");

enum HPTimeResolution
{
	kUnknownTimeResolution = 0,
	kHourResolution,
	kMinuteResolution
};

const boost::unordered_map<HPTimeResolution,const char*> HPTimeResolutionToString = map_list_of
		(kUnknownTimeResolution, "unknown")
		(kHourResolution, "hour")
		(kMinuteResolution, "minute");

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

const boost::unordered_map<HPAggregationType,const char*> HPAggregationTypeToString = map_list_of
		(kUnknownAggregationType, "unknown")
		(kAverage, "average")
		(kAccumulation, "accumulation")
		(kMaximum, "maximum")
		(kMinimum, "minimum")
		(kDifference, "difference");

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
	kFindValueModifier
};

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

const boost::unordered_map<HPPrecipitationForm,const char*> HPPrecipitationFormToString = map_list_of
		(kDrizzle, "drizzle")
		(kRain, "rain")
		(kSleet, "sleet")
		(kSnow, "snow")
		(kFreezingDrizzle, "freezing drizzle")
		(kFreezingRain, "freezing rain")
		(kGraupel, "graupel")
		(kHail, "hail")
		(kUnknownPrecipitationForm, "unknown");


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

} // namespace himan

#endif /* HIMAN_COMMON_H */
