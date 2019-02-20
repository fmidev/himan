/**
 * @file util.h
 *
 *
 * @brief Utility namespace for general helper functions and classes
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "configuration.h"
#include "grid.h"
#include "himan_common.h"
#include "info.h"
#include "search_options.h"
#include <boost/iterator/zip_iterator.hpp>
#include <memory>
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

template <typename T>
std::string MakeFileName(HPFileWriteOption fileWriteOption, const info<T>& info, const configuration& conf);

/**
 * @brief Splits a string and fills the gaps if requested
 * @param s Input string that should be splitted
 * @param delims String containing the characters that are used in splitting. If string length > 1, all characters are
 * used
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

std::tuple<double, double, double, double> EarthRelativeUVCoefficients(const himan::point& regPoint,
                                                                       const himan::point& rotPoint,
                                                                       const himan::point& southPole);

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

std::tuple<double, double, double, double> GridRelativeUVCoefficients(const himan::point& regPoint,
                                                                      const himan::point& rotPoint,
                                                                      const himan::point& southPole);

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
 * @param grids List of infos that are unpacked.
 */

template <typename T>
void Unpack(std::vector<std::shared_ptr<info<T>>> infos, bool addToCache = true);

/**
 * @brief Compute the x/y-derivative of input A
 * @param A Data field of input values
 * @param dx vector of x-spacing as function of y
 * @param dy vector of y-spacing as function of x
 * @return pair of dA/dx,dA/Dy
 */

template <typename T>
std::pair<himan::matrix<T>, himan::matrix<T>> CentralDifference(himan::matrix<T>& A, std::vector<T>& dx,
                                                                std::vector<T>& dy);

/**
 * @brief Round a double to requested precision
 * @param val Value that is going to be rounded
 * @param numdigits How many digits to preserve
 * @return Rounded value
 */

double round(double val, unsigned short numdigits);

/**
 * @brief Calculate the length of a latitude on a sphere with mean radius of the earth
 * @param phi Latitude that is going to be calculated
 * @return Length of Latitude
 */

double LatitudeLength(double phi);

/**
 * @brief Make SQL style interval value from forecast time
 *
 * SQL interval is like: 01:00:00, as in 1 hour 0 minutes 0 seconds
 *
 * This function will not handle seconds.
 *
 * @param theTime The forecast time
 * @return Interval string
 */

std::string MakeSQLInterval(const himan::forecast_time& theTime);

/**
 * @brief Expand possible environmental variables from a string.
 *
 * For example $HOME --> /home/weto
 *
 * @param in string where there might be env variables
 * @return string that contains the variables expanded
 */

std::string Expand(const std::string& in);

/**
 * @brief Dump contents of vector to stdout. Name is used to identify dump, when
 * multiple vectors are dumped sequentially.
 *
 * @param arr vector of doubles
 * @param name identifier for vector
 */

template <typename T>
void DumpVector(const std::vector<T>& arr, const std::string& name = "");
void DumpVector(const std::vector<double>& arr, const std::string& name = "");
void DumpVector(const std::vector<float>& arr, const std::string& name = "");

/**
   @brief Get the value of the specified environment variable.
   Throws when the supplied key is not found.

   @return value of the supplied key
 */

std::string GetEnv(const std::string& key);

/**
 * @brief Parse csv line(s) to an info
 *
 * CSV format needs to be:
 *
 * station_id,origintime,validtime,levelname,levelvalue,levelvalue2,forecasttypeid,forecasttypevalue,paramname,value
 *
 * For example
 *
 * 134254,2017-04-13 00:00:00,2017-04-13 03:00:00,HEIGHT,0,-1,1,-1,T-K,5.3
 */

template <typename T>
std::shared_ptr<info<T>> CSVToInfo(const std::vector<std::string>& csv);

/**
 * @brief Return the percentage of missing values in info for all grids.
 */

double MissingPercent(const himan::info<double>& info);

/**
 * @brief Parse boolean value from string
 */

bool ParseBoolean(const std::string& val);

/**
 * @brief create an empty grid for a given geom_name from db
 */

std::unique_ptr<grid> GridFromDatabase(const std::string& geom_name);

/**
 * @brief Flip matrix value around x-axis
 *
 * Used to be class function in grids called Swap().
 */

template <typename T>
void Flip(matrix<T>& mat);

/**
 *
 * @brief Create unique string identifier from search_options or info
 *
 */

std::string UniqueName(const plugin::search_options& options);

template <typename T>
std::string UniqueName(const info<T>& info);

template <class... Conts>
inline auto zip_range(Conts&... conts)
    -> decltype(boost::make_iterator_range(boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
                                           boost::make_zip_iterator(boost::make_tuple(conts.end()...))))
{
	return {boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
	        boost::make_zip_iterator(boost::make_tuple(conts.end()...))};
}

}  // namespace util
}  // namespace himan

#endif /* UTIL_H_ */
