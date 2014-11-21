/**
 * @file util.h
 *
 * @date Dec 28, 2012
 * @author partio
 *
 * @brief Utility namespace for general helper functions and classes
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "himan_common.h"
#include "info.h"
#include <mutex>
#include <tuple>

namespace himan
{

/**
 * @namespace himan::util 
 * @brief Namespace for all utility functions and classes
 */
namespace util
{

/**
 * @brief Determine file type by looking at the first few bytes of the file
 * @return Filetype of given argument, one of: grib, netcdf, querydata (or unknown)
 */

HPFileType FileType(const std::string& theFile);

/**
 * @brief Creates a neons-style filename with path, but without file extension
 */

std::string MakeFileName(HPFileWriteOption fileWriteOption, std::shared_ptr <const info> info);

/**
 * @brief Splits a string and fills the gaps if requested
 * @param s Input string that should be splitted
 * @param delims String containing the characters that are used in splitting. If string length > 1, all characters are used
 * in splitting
 * @param fill Specify if gaps should be filled. For example string 1-4 can be splitted at - and filled with 2 and 3.
 * @return Vector or strings
 */

std::vector<std::string> Split(const std::string& s, const std::string& delims, bool fill);

/**
 * @brief Join a string-vector with given delimiter
 * @param elmements vector of strings
 * @param delim delimiter
 * @return Concatenated string
 */

std::string Join(const std::vector<std::string>& elements, const std::string& delim);

/**
 * @brief Calculate area coordinates from first gridpoint, scanning mode, grid size and distance between two gridpoints.
 *
 * Works for (rotated) latlon projections only!
 *
 * This function is the opposite of FirstGridPoint(). NOTE: scanning mode must already be set when calling this function!
 *
 * @param firstPoint Latitude and longitude of first gridpoint
 * @param ni Grid size in X direction
 * @param ny Grid size in Y direction
 * @param di Distance between two points in X direction
 * @param dj Distance between two points in Y direction
 * @param scanningMode Scanningmode of data
 *
 * @return std::pair of points, first=bottomLeft, second=topRight
 */

std::pair<point,point> CoordinatesFromFirstGridPoint(const point& firstPoint, size_t ni, size_t nj, double di, double dj, HPScanningMode scanningMode);

/**
 * @brief Calculate area coordinates from first gridpoint. Works for stereographic projections only!
 * @param firstPoint First point specified in grib
 * @param orientation Orientation of grid
 * @param ni Grid size in X direction
 * @param nj Grid size in Y direction
 * @param xSizeInMeters Distance between two points in meters in X direction
 * @param ySizeInMeters Distance between two points in meters in Y direction
 * @return
 */

std::pair<point,point> CoordinatesFromFirstGridPoint(const point& firstPoint,  double orientation, size_t ni, size_t nj, double xSizeInMeters, double ySizeInMeters);


/**
 * @brief Calculate coefficients for transforming U and V from grid relative to earth relative.
 *
 * Algorithm by J.E. HAUGEN (HIRLAM JUNE -92), modified by K. EEROLA
 * Algorithm originally defined in hilake/TURNDD.F
 *
 * All coordinate values are given in degrees N and degrees E (negative values for S and W)
 *
 * Function works only with rotated latlon projections.
 *
 * @param regPoint Latlon coordinates of the point in question in earth-relative form
 * @param rotPoint Latlon coordinates of the point in question in grid-relative form
 * @param southPole Latlon coordinates of south pole
 * @return Four coefficients for transforming U and V
 */

std::tuple<double,double,double,double> EarthRelativeUVCoefficients(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole);

/**
 * @brief Calculate coefficients for transforming U and V from earth relative to grid relative.
 *
 * Algorithm by J.E. HAUGEN (HIRLAM JUNE -92), modified by K. EEROLA
 * Algorithm originally defined in hilake/TURNDD.F
 *
 * All coordinate values are given in degrees N and degrees E (negative values for S and W)
 *
 * Function works only with rotated latlon projections.
 *
 * @param regPoint Latlon coordinates of the point in question in earth-relative form
 * @param rotPoint Latlon coordinates of the point in question in grid-relative form
 * @param southPole Latlon coordinates of south pole
 * @return Four coefficients for transforming U and V
 */

std::tuple<double,double,double,double> GridRelativeUVCoefficients(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole);

/**
 * @brief If U and V components of a parameter are grid relative, transform them to be earth-relative
 *
 * Algorithm used is
 *
 * Ugeo = Ustereo * cos(x) + Vstereo * sin(x)
 * Vgeo = -Ustereo * sin(x) + Vstereo * cos(x)
 *
 * Where x is longitude of the point east of reference longitude
 * Note: The reference longitude is not always Greenwich longitude
 *
 * Algorithm originally defined in hilake/VPTOVM.F
 *
 * Function works only with stereographic projections.
 *
 * !!! FUNCTION HAS NOT BEEN THOROUGHLY TESTED DUE TO LACK OF INPUT UV DATA IN STEREOGRAPHIC PROJECTION !!!
 *
 * @param longitude Reference longitude
 * @param rotatedUV U and V in grid-relative form
 * @return U and V in earth-relative form
 */

himan::point UVToGeographical(double longitude, const himan::point& stereoUV);

/**
 * @brief Raise value to power. Function originated from grib_api.
 * @param value Value that's raised
 * @param power To which power value is raised
 * @return `value` raised to `power`:th power
 */

double ToPower(double value, double power);

/**
 * @brief Unpack grib simple_packing
 *
 * This function can be called on CPU to unpack the data on CUDA and return
 * the results to CPU memory.
 *
 * @param grids List of grids that are unpacked.
 */

void Unpack(std::initializer_list<grid*> grids);

/**
 * @brief Compute convolution of matrix A by matrix B
 * @param A Data
 * @param B Convolution kernel
 * @return Data convolved by kernel
 */

himan::matrix<double> Filter2D(himan::matrix<double>& A, himan::matrix<double>& B);

/**
 * @brief Round a double to requested precision
 * @param val Value that is going to be rounded
 * @param numdigits How many digits to preserve
 * @return Rounded value
 */

double round(double val,unsigned short numdigits);

} // namespace util
} // namespace himan


#endif /* UTIL_H_ */
