# Summary

probability plugin is used for ensemble data to calculate the probability of any parameter using user given thresholds. Plugin also supports time-lagged ensemble.

# Required source parameters

Any single source parameter. For wind speed, plugin will automatically fetch U and V components.

# Output parameters

User defined output parameters, see per-plugin configuration options.

# Method of calculation

Plugin will check how many ensemble members forecast values fulfill the given conditition (often being larger or smaller than some threshold). Plugin does not support multi-parameter probabilities.
If ensemble contains missing values, they are removed before producing the probability value.

# Per-plugin configuration options

name: what is the resulting parameter of calculation

    "name" : "PROB-TC-0"

ensemble_size: define the expected ensemble size.

    "ensemble_size" : "<number>"

max_missing_forecasts: define how many forecasts are allowed to be missing from the ensemble (default is 0)

    "max_missing_forecasts" : "<number>"

ensemble_type: define the type of ensemble. Currently supported is the traditional ensemble with multiple perturbed forecasts, and a time_ensemble where multiple different times produce an ensemble. Default value is the traditional ensemble.

    "ensemble_type" : "ensemble | time_ensemble"

input_param: name of first input parameter

    "input_param" : "T-K"

threshold: threshold for given parameter, in parameter units.

    "threshold" : "273.15"

comparison: specify a comparison operator (default: ">=")

    "comparison" : "<="

Possible values for comparison operator are

    ">=": greater than or equal
    "<=": less than or equal
    "=": equal to
    "!=" or "<>": not equal to
    "=[]": value belongs to a set of values. The set of values are defined with key "threshold" separated by commas
    "[)": value belongs to a range bounded by [lower value, upper value), lower endpoint included. The set of values are defined with key "threshold" separated by commas, exactly two values must be given.

Note that all comparison are made with floating point values.

normalized_results: Define if plugin should scale the probability values to [0,1] (default: [0,100])

    "normalized_results" : "true"

lag: Define is plugins should use time-lagging for how many instances (default: 0)

    "lag" : "1"

lagged_steps: Define how many hours is one lagged forecast (the temporal distance between two consecutive forecast runs)

    "lagged_steps" : "12"
