# Summary

probability plugin is used for ensemble data to calculate the probability of any parameter using user given thresholds. Plugin also supports time-lagged ensemble.

# Required source parameters

Any single source parameter.

# Output parameters

User defined output parameters, see per-plugin configuration options. The probability values are written to range 0 .. 1.

# Method of calculation

Plugin will check how many ensemble members forecast values fulfill the given conditition (often being larger or smaller than some threshold). Plugin does not support multi-parameter probabilities.
If ensemble contains missing values, they are removed before producing the probability value.

For some parameters that follow the normal distribution, the probabilities are calculated using the distribution. This removes some of the ill effects of having too small sample size.

The parameters are:

* T-K
* T-C
* TD-K
* TD-C
* WATLEV-CM
* P-PA
* P-HPA

# Per-plugin configuration options

Check file "plugin-fractile.md" to see how ensembles configuration cat be defined.

name: what is the resulting parameter of calculation

    "name" : "PROB-TC-0"

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
