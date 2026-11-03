# Summary

probability plugin is used for ensemble data to calculate the probability of any parameter using user given thresholds. Plugin also supports time-lagged ensemble.

# Required source parameters

Any single source parameter.

# Output parameters

User defined output parameters, see per-plugin configuration options. The probability values are written to range 0 .. 1.

# Method of calculation

Plugin will check how many ensemble members forecast values fulfill the given conditition (often being larger or smaller than some threshold). Probabilities that depend on several parameters (joint probabilities) can be calculated with the key "joint_probability", see below.
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

joint_probability: list of rules that is used instead of "input_param", "threshold", "comparison" and "aggregation" when the probability depends on several parameters. The probability is the fraction of ensemble members that fulfill *all* of the rules, so the rules are evaluated for the same ensemble member of each parameter. This is the probability of the rules being true at the same time, not a conditional probability. For example, the probability of rain (precipitation form 1) with precipitation rate greater than 0.04 kg/m2/h:

    {
      "name" : "PROB-RR1-1",
      "joint_probability" : [ { "input_param" : "RRR-KGM2", "comparison" : ">", "value" : "0.04" },
                              { "input_param" : "PRECFORM2-N", "comparison" : "=", "value" : "1" } ]
    }

Each rule has the following keys:

    "input_param": name of the input parameter (required)
    "comparison": comparison operator, any of the operators listed above (required)
    "value": value to compare to, in parameter units. With operators "=[]" and the range operator the values are separated by commas (required)
    "aggregation": aggregation of the input parameter (optional)

Key "threshold" is not allowed in a rule, use "value". The rules cannot be nested and a joint probability with a single rule is the same as the plain probability. The order of the rules does not change the result, but the first rule has a special role: it defines the ensemble that the others are paired with, the threshold and comparison that are stored in the metadata of the output parameter and the aggregation of the output parameter. The other rules are not stored in the metadata, so the meaning of a joint probability output is defined by its parameter in radon: each joint probability needs its own output parameter (name and GRIB parameter number) that is documented with its rules. Station-wise thresholds ("threshold_<station id>" or the database) are not supported with "joint_probability", and the options "input_param", "threshold", "comparison" and "aggregation" cannot be given with it.

Missing values are handled separately for each input parameter: for parameters that allow missing values (for example PRECFORM2-N) a missing value means that the rule is not fulfilled but the member is still counted, otherwise the member is removed from the calculation. If all members are removed, the probability is missing. If data is not found for any of the parameters, the probability is not produced. All parameters must have the same number of ensemble members.

A joint probability of several rules cannot be used when the first rule is for a parameter for which the probabilities are calculated from the normal distribution (see above).
