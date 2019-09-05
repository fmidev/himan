#pragma once

#include "ensemble.h"
#include "himan_common.h"
#include "info.h"

namespace PROB
{
/*
 * struct partial_param_configuration
 *
 * This struct is filled from configuration file information and if we calculate station-wise
 * probabilites database is also used.
 *
 * It's called partial because the thresholds are stored as strings. When doing the actual
 * calculation we 'instantiate' the partial struct to actual param_configuration. This two-struct
 * strategy is used because if we'd have only one instantiated struct for data, we would have
 * to also templetize the plugin which would change the whole himan structure. Or pull all
 * helper functions from compiled_plugin_base to this plugin which also not very tempting.
 */

struct partial_param_configuration
{
	std::vector<std::string> thresholds;
	std::map<int, std::string> stationThresholds;

	// Output parameter, the result
	himan::param output;

	// Input parameter used for calculating the 'target'.
	himan::param parameter;

	// if parameter has gaussian distribution, the probabilities can be derived
	// from the distribution
	bool useGaussianSpread;
};

/*
 * struct param_configuration
 *
 * Describes how output parameters are calculated from input parameters.
 * The template parameter is describes the limit(s) used. For most of the
 * cases, it is a double i.e. we are only comparing to a single number.
 *
 * In some cases though we can check if a data value is *one of several values*,
 * and in this case these values are stored in a vector.
 *
 * So therefore the template arguments that are supported as of writing this
 * first implementation are:
 * - double
 * - std::vector<double>
 *
 */

template <typename T>
struct param_configuration
{
	std::vector<T> thresholds;

	// Output parameter, the result
	himan::param output;

	// Input parameter used for calculating the 'target'.
	himan::param parameter;

	bool useGaussianSpread;
};

}  // namespace PROB
