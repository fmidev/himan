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
 * @brief If U and V components of wind are grid relative, transform them to be earth-relative.
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
 * @param UV U and V in grid-relative form
 * @return U and V in earth-relative form
 */

himan::point UVToEarthRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV);

/**
 * @brief If U and V components of wind are earth relative, transform them to be grid-relative.
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
 * @param UV U and V in earth-relative form
 * @return U and V in grid-relative form
 */

himan::point UVToGridRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV);

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
 * @brief Calculates Relative Topography between the two given fields in Geop
 * @param level1 Value of pressure level1
 * @param level2 Value of pressure level2
 * @param z1 Geopotential height of level1, Use pressure if level1 = 1000
 * @param z2 Geopotential height of level2
 * @return Relative Topography in Geopotential
 */

double RelativeTopography(int level1, int level2, double z1, double z2);

/**
 * @brief Calculates low convection for a point.
 * @param T2m Value of 2m temperature
 * @param T850 Value of temperature at 850 hPa pressure level
 * @return convection value
 */
int LowConvection(double T2m, double T850);


/**
 * @brief Calculate water vapor saturated pressure in hPa
 *
 * Equation found in f.ex. Smithsonian meteorological tables or
 * http://www.srh.noaa.gov/images/epz/wxcalc/vaporPressure.pdf
 *
 * If temperature is less than -5, use ice instead of water for
 * calculations.
 *
 * @param T Temperature in C
 * @return Saturated water vapor pressure in hPa
 */

double Es(double T);

/**
 * @brief Calculates pseudo-adiabatic lapse rate
 *
 * Originally author AK Sarkanen May 1985.
 * 
 * @param P Pressure in hPa
 * @param T Temperature in C
 * @return Lapse rate in C/km
 */

double Gammas(double P, double T);

/**
 * @brief Calculates the temperature, pressure and specific humidity (Q) of
 * a parcel of air in LCL
 *
 * Original author AK Sarkanen/Kalle Eerola
 *
 * @param P Pressure in hPa
 * @param T Temperature in C
 * @param TD Dew point temperature in C
 * @return Pressure, temperature and specific humidity (g/kg) for LCL (in this order).
 */

const std::vector<double> LCL(double P, double T, double TD);

} // namespace util
} // namespace himan


#endif /* UTIL_H_ */
