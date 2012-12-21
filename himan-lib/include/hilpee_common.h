/*
 * hilpee_common.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef HILPEE_COMMON_H
#define HILPEE_COMMON_H

// Work around "passing 'T' chooses 'int' over 'unsigned int'" warnings when T
// is an enum type:

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic ignored "-Wsign-promo"
#endif

// Check whether we can use std::shared_ptr and std::unique_ptr

#if defined __GNUC__ && (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#define HAVE_CPP11
#else
#undef HAVE_CPP11
#endif

#include <ostream>

namespace hilpee
{

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

enum HPProjectionType :
unsigned int
{
	kUnknownProjection = 0,
	kLatLonProjection = 10,
	kRotatedLatLonProjection = 11,
	kStereographicProjection = 13
};

// Define supported parameter units

enum HPParameterUnit
{
	kUnknownUnit = 0,
	kPa,
	kK,
	kC,
	kPas, // Pa/s
};

enum HPLevelType
{
	kUnknownLevel = 0,
	kGround = 1,
	kPressure = 100,
	kMeanSea = 102,
	kHeight = 105,
	kHybrid = 109,
	kDepth = 160
};

#if 0
enum HPDimensions
{
	kUnknownDimennsion = 0,
	kTimeDimension,
	kLevelDimension,
	kParamDimension,
};
#endif

// Version number for plugins

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

} // namespace hilpee

#endif /* HILPEE_COMMON_H */
