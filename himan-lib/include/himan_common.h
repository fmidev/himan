/*
 * himan_common.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
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

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4))
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
const float kHPMissingFloat = -999.f; // Hmmh?

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
    kMs // m/s
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

enum HPExceptionType
{
    kUnknownException = 0,
    kFileMetaDataNotFound,
    kFileDataNotFound,
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

enum HPTimeResolution
{
	kUnknownTimeResolution = 0,
	kHour,
	kMinute
};

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
