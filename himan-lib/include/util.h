/**
 * @file util.h
 *
 *
 * @brief Utility namespace for general helper functions and classes
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "configuration.h"
#include "ensemble.h"
#include "grid.h"
#include "himan_common.h"
#include "info.h"
#include "search_options.h"
#include <boost/iterator/zip_iterator.hpp>
#include <memory>

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
 * @brief Splits a string and returns part in a vector
 * @param s Input string that should be splitted
 * @param delims String containing the characters that are used in splitting. If string length > 1, all characters are
 * used in splitting
 * @return Vector of strings
 */

std::vector<std::string> Split(const std::string& s, const std::string& delims);

template <typename T>
std::vector<T> Split(const std::string& s, const std::string& delims);

/**
 * @brief Expand a string containing commas and dashes to a list (vector)
 *
 * For example: 4,5,10-12 becomes vector of 4,5,10,11,12
 *
 * @param identifier which is splitted and expanded
 * @return Vector of ints, the expanded values
 */

std::vector<int> ExpandString(const std::string& identifier);
std::vector<time_duration> ExpandTimeDuration(const std::string& identifier);

/**
 * @brief Join a string-vector with given delimiter
 * @param elmements vector of strings
 * @param delim delimiter
 * @return Concatenated string
 */

std::string Join(const std::vector<std::string>& elements, const std::string& delim);

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
 * @param jPositive bool that indicates y-axis direction in the data
 * @return pair of dA/dx,dA/Dy
 */

template <typename T>
std::pair<himan::matrix<T>, himan::matrix<T>> CentralDifference(himan::matrix<T>& A, std::vector<T>& dx,
                                                                std::vector<T>& dy, bool jPositive);

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

template <typename T>
T LatitudeLength(T phi);

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

/**
   @brief Get the value of the specified environment variable.
   Throws when the supplied key is not found.

   @return value of the supplied key
 */

std::string GetEnv(const std::string& key);

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

/**
 * @brief Initialize parameter with given information
 *
 * Aggregation and processing type are determined from parameter name.
 */

param InitializeParameter(const producer& prod, const param& par, const level& lvl);

/**
 * @brief Try to determine aggregation from parameter name
 *
 * Intermediate solution while waiting for a better way to do this,
 * perhaps through database.
 *
 */

aggregation GetAggregationFromParamName(const std::string& name, const forecast_time& ftime);

/**
 * @brief Try to determine processing type from parameter name
 *
 * See comments from above function
 */

processing_type GetProcessingTypeFromParamName(const std::string& name);

/**
 * @brief Get a scaled long valie from a double
 *
 * @return scaled value and scale factor
 */

std::pair<long, long> GetScaledValue(double v);

/**
 * @brief map and reduce in single function call
 *
 * Function naming does not follow usual Himan syntax but is
 * in line with standard library.
 *
 * https://stackoverflow.com/a/52400920
 */

template <class InputIt, class OutputIt, class Pred, class Fct>
void transform_if(InputIt first, InputIt last, OutputIt dest, Pred pred, Fct transform)
{
	while (first != last)
	{
		if (pred(*first))
			*dest++ = transform(*first);

		++first;
	}
}

/**
 * @brief Convert vector from float to double or vice versa, retaining correct
 * missing value.
 *
 * Template specialization to avoid resource-wasting conversion from double --> double
 * or float --> float
 *
 * T = from
 * U = to
 */

template <typename T, typename U>
std::vector<U> Convert(const std::vector<T>&);

template <>
inline std::vector<double> Convert(const std::vector<double>& v)
{
	return v;
}

template <>
inline std::vector<float> Convert(const std::vector<float>& v)
{
	return v;
}

template <>
inline std::vector<double> Convert(const std::vector<float>& v)
{
	std::vector<double> ret(v.size());
	replace_copy_if(
	    v.begin(), v.end(), ret.begin(), [](const float& val) { return IsMissing(val); }, MissingDouble());
	return ret;
}

template <>
inline std::vector<float> Convert(const std::vector<double>& v)
{
	std::vector<float> ret(v.size());
	replace_copy_if(
	    v.begin(), v.end(), ret.begin(), [](const double& val) { return IsMissing(val); }, MissingFloat());
	return ret;
}

/**
 * @brief Create a list (vector) of forecast types based on string specifier
 *
 * For example pf1-10 will create a list of 10 forecast_types with numbers 1...10
 *
 */

std::vector<forecast_type> ForecastTypesFromString(const std::string& type);

/**
 * @brief Create an ensemble from options found in configuration file
 *
 */

std::unique_ptr<ensemble> CreateEnsembleFromConfiguration(const std::shared_ptr<const plugin_configuration>& conf);

/**
 * @brief Create a hybrid or generalized vertical layer
 *
 * Necessary information is fetched from radon
 *
 */

level CreateHybridLevel(const producer& prod, const std::string& first_last);

template <typename T>
std::vector<T> RemoveMissingValues(const std::vector<T>& vec);

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
