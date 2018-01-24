# Summary

Fractile plugin is used for ensemble data and it calculates a set of fractiles for a given paramerer. The ensemble can be a traditional ensemble with multiple perturbed forecasts, or a time ensemble consisting of consecutive times.

The default fractiles are:

* 0th fractile (smallest value)
* 10th fractile (first decile)
* 25th fractile (first quartile)
* 50th fractile (median)
* 75th fractile (third quartile)
* 90th fractile (ninth decile)
* 100th fractile (largest value)

The exact fractile value is interpolated linearly, as recommended by NIST (http://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm).

Plugin will also calculate mean value and spread/standard deviation.

# Required source parameters

Any single source parameter.

# Output parameters

Any wanted fractile, the naming of parameters is F<fractile>-<paramname> (eg. F100-T-K), mean value <param>-MEAN-<unit> (T-MEAN-K) and spread <param>-STDDEV-<unit> (T-STDDEV-K).

# Method of calculation

Plugin will sort the input values and calculate the required fractiles. Missing values are excluded.

# Per-plugin configuration options

param: what parameter is used for fractile calculation

    "param" : "T-K"

ensemble_size: define the expected ensemble size.

    "ensemble_size" : "<number>"

If key is not specified, ensemble size is fetched from database table producer_meta.

max_missing_forecasts: define how many forecasts are allowed to be missing from the ensemble (default is 0)

    "max_missing_forecasts" : "<number>"

fractiles: which set of fractiles to calculate.

    "fractiles" : "100,90,75,50,25,20,10,0"

ensemble_type: define the type of ensemble. Currently supported is the traditional ensemble with multiple perturbed forecasts, and a time_ensemble where multiple different times produce an ensemble. Default value is the traditional ensemble.

    "ensemble_type" : "ensemble | time_ensemble"
