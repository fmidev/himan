#pragma once

#include "ensemble.h"
#include "himan_common.h"
#include "info.h"
#include <functional>

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

/*
 * struct partial_condition
 *
 * An additional condition that an ensemble member has to fulfill, in addition to the primary
 * comparison, to be counted in the probability. Like the thresholds of the primary comparison,
 * the value is stored as a string until the calculation is started.
 */

struct partial_condition
{
	// Input parameter that is compared
	himan::param parameter;

	// Comparison operator
	himan::HPProcessingType comparison;

	// One value, or comma-separated list of values for operators =[] and []
	std::string value;
};

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

	// Additional conditions, all of them (and the primary comparison) must be true
	// for an ensemble member to be counted
	std::vector<partial_condition> conditions;
};

/*
 * struct condition
 *
 * Instantiated partial_condition. The comparison is already bound to the value(s).
 */

struct condition
{
	himan::param parameter;
	std::function<bool(float)> compare;
};

/*
 * struct rule_input
 *
 * One rule of a joint probability for a single grid point: the values of all ensemble members
 * for one parameter, and what a value must fulfill for the member to be counted.
 */

struct rule_input
{
	// Value of each ensemble member
	std::vector<float> values;

	// True if the member fulfills the rule. Not called with missing values.
	std::function<bool(float)> compare;

	// If true, a missing value means 'rule not fulfilled' but the member is still part of the sample.
	// Otherwise a member with a missing value is left out of the sample.
	bool allowMissing;
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

	std::vector<condition> conditions;
};

}  // namespace PROB
