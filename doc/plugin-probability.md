# Summary

probability plugin is used for ensemble data to calculate the probability of any parameter using user given thresholds. Plugin also supports time-lagged ensemble.

# Required source parameters

Any single source parameter.

# Output parameters

User defined output parameters, see per-plugin configuration options. The probability values are written to range 0 .. 1.

# Method of calculation

Plugin will check how many ensemble members forecast values fulfill the given conditition (often being larger or smaller than some threshold). Probabilities that depend on several parameters can be calculated with additional conditions, see "conditions" below.
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

conditions: optional list of additional conditions that an ensemble member must fulfill, in addition to the comparison above, to be counted. The probability is the fraction of ensemble members that fulfill the comparison *and* all of the conditions, so the conditions are evaluated for the same ensemble member of each parameter. For example, the probability of rain (precipitation form 1) with precipitation rate greater than 0.04 kg/m2/h:

    {
      "name" : "PROB-RR1-1",
      "input_param" : "RRR-KGM2",
      "threshold" : "0.04",
      "comparison" : ">",
      "conditions" : [ { "input_param" : "PRECFORM2-N", "comparison" : "=", "value" : "1" } ]
    }

The comparison defined with "input_param", "threshold" and "comparison" is the primary comparison: it defines the threshold that is stored to the output parameter and the one that is used with station-wise thresholds ("threshold_<station id>" or the database). Station-wise values cannot be defined for conditions.

Each condition has the following keys:

    "input_param": name of the input parameter (required)
    "comparison": comparison operator, any of the operators listed above (required)
    "value": value to compare to, in parameter units. With operators "=[]" and the range operator the values are separated by commas (required)
    "aggregation": aggregation of the input parameter (optional). Unlike with the primary comparison, the aggregation of the primary input parameter is not applied to conditions.

Key "threshold" is not allowed in conditions. An empty list of conditions is ignored.

Missing values are handled separately for each input parameter: for parameters that allow missing values (for example PRECFORM2-N) a missing value means that the comparison is not fulfilled but the member is still counted, otherwise the member is removed from the calculation. If all members are removed, the probability is missing. If data is not found for any of the parameters, the probability is not produced. All parameters must have the same number of ensemble members.

Conditions cannot be used with parameters for which the probabilities are calculated from the normal distribution (see above).
