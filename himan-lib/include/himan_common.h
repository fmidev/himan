/**
 * @file himan_common.h
 *
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

#if !defined __clang__ && defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)) && !__CUDACC__
#pragma GCC diagnostic ignored "-Wsign-promo"
#endif

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6))
#define override  // override specifier not support until 4.8
#endif

#include "cuda_helper.h"
#include "debug.h"
#include <boost/assign/list_of.hpp>
#include <boost/unordered_map.hpp>
#include <math.h>
#include <memory>
#include <ostream>
#include <string>

namespace ba = boost::assign;

// clang-format off

namespace himan
{
// Define some missing value utilities
inline CUDA_HOST CUDA_DEVICE double MissingDouble() {return nan("0x7fffffff");}
inline CUDA_HOST CUDA_DEVICE float MissingFloat() {return nanf("0x7fffffff");} // Cuda version of nanf(char*) has a bug and does not respect the argument given.
										// Bug is fixed in a later Cuda release (> 9.1)
inline CUDA_HOST CUDA_DEVICE short MissingShort() { return static_cast<short> (-32768); }
inline CUDA_HOST CUDA_DEVICE unsigned char MissingUnsignedChar() { return static_cast<unsigned char> (255); }

const int kHPMissingInt = 999999;
const double kHPMissingValue = -999.;

inline CUDA_HOST CUDA_DEVICE bool IsMissingDouble(const double& value)
{
        double missingValue = MissingDouble();
        const uint64_t* _value = reinterpret_cast<const uint64_t*>(&value);
        const uint64_t* _missingValue = reinterpret_cast<const uint64_t*>(&missingValue);

        return (*_value == *_missingValue);
}

inline CUDA_HOST CUDA_DEVICE bool IsMissingFloat(const float& value)
{
        float missingValue = MissingFloat();
        const uint32_t* _value = reinterpret_cast<const uint32_t*>(&value);
        const uint32_t* _missingValue = reinterpret_cast<const uint32_t*>(&missingValue);

        return (*_value == *_missingValue);
}

inline CUDA_HOST CUDA_DEVICE bool IsMissing(double value) {return IsMissingDouble(value);}
inline CUDA_HOST CUDA_DEVICE bool IsMissing(float value) {return IsMissingFloat(value);}
inline CUDA_HOST CUDA_DEVICE bool IsMissing(short value) {return value == MissingShort(); }
inline CUDA_HOST CUDA_DEVICE bool IsMissing(unsigned char value) {return value == MissingUnsignedChar(); }

inline CUDA_HOST CUDA_DEVICE bool IsValid(double value) { return !IsMissingDouble(value);}
inline CUDA_HOST CUDA_DEVICE bool IsValid(float value) {return !IsMissingFloat(value);}
inline CUDA_HOST CUDA_DEVICE bool IsValid(short value) {return !IsMissing(value);}
inline CUDA_HOST CUDA_DEVICE bool IsValid(unsigned char value) {return !IsMissing(value);}

// templatized MissingValue() for double & float is useful in template functions
// where "typename" is float or double

template <typename T>
CUDA_HOST CUDA_DEVICE T MissingValue() { return std::numeric_limits<T>::max(); }

template <>
CUDA_HOST CUDA_DEVICE
inline unsigned char MissingValue() { return MissingUnsignedChar(); }

template <>
CUDA_HOST CUDA_DEVICE
inline short MissingValue() { return MissingShort(); }

template <>
CUDA_HOST CUDA_DEVICE
inline double MissingValue() { return MissingDouble(); }

template <>
CUDA_HOST CUDA_DEVICE
inline float MissingValue() { return MissingFloat(); }

template <typename T>
inline bool IsMissing(T value) { return value == MissingValue<T>(); }
template <typename T>
inline bool IsValid(T value) { return !IsMissing(value); }

inline CUDA_HOST CUDA_DEVICE bool IsKHPMissingValue(const double& x) {return x == -999;}

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
	kGRIB,  // when we don't know if its 1 or 2
	kGRIBIndex,
	kQueryData,
	kNetCDF,
	kCSV,
	kGeoTIFF,
	kNetCDFv4
};

const boost::unordered_map<HPFileType, std::string> HPFileTypeToString =
	ba::map_list_of
	(kUnknownFile, "unknown")
	(kGRIB1, "grib edition 1")
	(kGRIB2, "grib edition 2")
	(kGRIB, "grib edition 1 or 2")
	(kGRIBIndex, "grib index file")
	(kQueryData, "QueryData")
	(kNetCDF, "NetCDF")
	(kCSV, "CSV")
	(kGeoTIFF, "GeoTIFF")
	(kNetCDFv4, "NetCDFv4");

// Defined external compression types

enum HPFileCompression
{
	kUnknownCompression = 0,
	kNoCompression,
	kGZIP,
	kBZIP2
};

const boost::unordered_map<HPFileCompression, std::string> HPFileCompressionToString =
    ba::map_list_of
	(kUnknownCompression, "unknown compression")
	(kNoCompression, "no compression")
	(kGZIP, "gzip compressed")
	(kBZIP2, "bzip2 compressed");


enum HPInterpolationMethod
{
	kUnknownInterpolationMethod = 0,
	kBiLinear = 1,
	kNearestPoint = 2,
	kNearestPointValue  // http://arxiv.org/pdf/1211.1768.pdf
};

const boost::unordered_map<HPInterpolationMethod, std::string> HPInterpolationMethodToString =
    ba::map_list_of
	(kUnknownInterpolationMethod, "unknown")
	(kBiLinear, "bilinear")
	(kNearestPoint, "nearest point")
	(kNearestPointValue, "nearest point value");

const boost::unordered_map<std::string, HPInterpolationMethod> HPStringToInterpolationMethod =
    ba::map_list_of
	("unknown", kUnknownInterpolationMethod)
	("bilinear", kBiLinear)
	("nearest point", kNearestPoint)
	("nearest point value", kNearestPointValue);

enum HPWriteMode
{
	kUnknown = 0,
	kAllGridsToAFile,
	kFewGridsToAFile,
	kSingleGridToAFile,
	kNoFileWrite
};

const boost::unordered_map<HPWriteMode, std::string> HPWriteModeToString =
    ba::map_list_of
	(kUnknown, "unknown")
	(kAllGridsToAFile, "all fields to single file")
	(kFewGridsToAFile, "few fields to a file")
	(kSingleGridToAFile, "one field per file")
	(kNoFileWrite, "do not write file at all");


enum HPExceptionType
{
	kUnknownException = 0,
	kFileMetaDataNotFound,
	kFileDataNotFound,
	kFunctionNotImplemented,
	kInvalidWriteOptions,
	kPluginNotFound
};

enum HPTimeResolution
{
	kUnknownTimeResolution = 0,
	kHourResolution,
	kMinuteResolution,
	kYearResolution,
	kMonthResolution,
	kDayResolution
};

const boost::unordered_map<HPTimeResolution, std::string> HPTimeResolutionToString =
    ba::map_list_of
	(kUnknownTimeResolution, "unknown")
	(kHourResolution, "hour")
	(kMinuteResolution, "minute")
	(kYearResolution, "year")
	(kMonthResolution, "month")
	(kDayResolution, "day");

const boost::unordered_map<std::string, HPTimeResolution> HPStringToTimeResolution =
    ba::map_list_of
	("unknown", kUnknownTimeResolution)
	("hour", kHourResolution)
	("minute", kMinuteResolution)
	("year", kYearResolution)
	("month", kMonthResolution)
	("day", kDayResolution);

enum HPPackingType
{
	kUnknownPackingType = 0,
	kUnpacked,
	kSimplePacking,
	kJpegPacking,
	kCcsdsPacking
};

const boost::unordered_map<HPPackingType, std::string> HPPackingTypeToString =
    ba::map_list_of
	(kUnknownPackingType, "unknown")
	(kUnpacked, "unpacked")
	(kSimplePacking, "simple_packing")
	(kJpegPacking, "jpeg_packing")
	(kCcsdsPacking, "ccsds_packing");

const boost::unordered_map<std::string, HPPackingType> HPStringToPackingType =
    ba::map_list_of
	("unknown", kUnknownPackingType)
	("unpacked", kUnpacked)
	("simple_packing", kSimplePacking)
	("jpeg_packing", kJpegPacking)
	("ccsds_packing", kCcsdsPacking);


enum HPDatabaseType
{
	kUnknownDatabaseType = 0,
	kRadon,
	kNoDatabase
};

const boost::unordered_map<HPDatabaseType, std::string> HPDatabaseTypeToString =
	ba::map_list_of
	(kUnknownDatabaseType, "unknown")
	(kRadon, "radon")
	(kNoDatabase, "no database");


enum HPFileStorageType
{
	kUnknownStorageType = 0,
	kLocalFileSystem,
	kS3ObjectStorageSystem
};

const boost::unordered_map<HPFileStorageType, std::string> HPFileStorageTypeToString =
    ba::map_list_of
	(kUnknownStorageType, "unknown")
	(kLocalFileSystem, "local file system")
	(kS3ObjectStorageSystem, "s3 object storage system")
	;

const boost::unordered_map<std::string, HPFileStorageType> HPStringToFileStorageType =
    ba::map_list_of
	("unknown", kUnknownStorageType)
	("local", kLocalFileSystem)
	("s3", kS3ObjectStorageSystem)
	;

enum HPProgramName {
	kUnknownProgram = 0,
	kHiman,
	kGridToRadon
};

// clang-format on

namespace constants
{
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
 * @brief Dimensionless ratio of the specific gas constant of dry air to the specific gas constant for water vapor, ie
 * kRd / kRw
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
 * Sometimes referred to as Poisson constant for dry air,
 * kappa_d.
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

}  // namespace constants

}  // namespace himan

#endif /* HIMAN_COMMON_H */
