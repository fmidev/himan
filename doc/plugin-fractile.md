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
For time ensemble, the ensemble_size number determines how many steps to history data is read.

max_missing_forecasts: define how many forecasts are allowed to be missing from the ensemble (default is 0)

    "max_missing_forecasts" : "<number>"

fractiles: which set of fractiles to calculate.

    "fractiles" : "100,90,75,50,25,20,10,0"

ensemble_type: define the type of ensemble. Currently supported are:

* the traditional ensemble with multiple perturbed forecasts
* a lagged version of the traditional ensemble, where n previous forecast runs are included
* time ensemble where multiple different times produce an ensemble.

Default value is the traditional ensemble.

    "ensemble_type" : "perturbed ensemble" | "time ensemble" | "lagged ensemble"


## Defining a lagged ensemble

Three different ways of defining a lagged ensemble.

### Method A

lag: Define is plugins should use time-lagging for how many instances (default: 0)

    "lag" : "1"

lagged_steps: Define how many hours is one lagged forecast (the temporal distance between two consecutive forecast runs)

    "lagged_steps" : "12"

### Method B

named_ensemble: Some more complicated ensemble setups have been defined in code, and can be referred to with name.

    "named_ensemble" : "MEPS_SINGLE_ENSEMBLE | MEPS_LAGGED_ENSEMBLE"

### Method C

lagged_members: list of members and their numbers

    "lagged_members" : "cf0,pf1-2"

lags: list of lags, each member listed with `lagged_members` must have a lag defined

    "lags" : "00:00:00,-01:00:00,-02:00:00"


## Defining a time ensemble

secondary_time_span: for time ensemble, specify secondary time mask type

    "secondary_time_span" : "hour" | "day" | "month" | "year"

secondary_time_len: for time ensemble, specify secondary time mask length

    "secondary_time_len" : "<number>"

secondary_time_step: for time ensemble, specify secondary time mask step

    "secondary_time_step" : "<number>"

If time_ensemble primary time span is year (this is hard coded for now), another time span can be defined to increase the sample size of the data. By default the secondary
time span is zero, ie. no span. The secondary mask works so that if the length is for example 3, step is 1 and mask type is hour, then data is being read from the current time
+3,+2,+1,0,-1,-2,-3 hours --> total of 7 fields. 

Example: from past 10 years, read data for current date +/- 12 hours hourly

ensemble_size=10,secondary_time_span=hour,secondary_time_len=12,secondary_time_step=1, the total amount of fields required is 10 * (2 * (12/1) + 1) = 250.

Example: from past 10 years, read data for current data +/- 72 hours, but read only the *times* that match current time

ensemble_size=10,secondary_time_span=hour,secondary_time_len=72,secondary_time_step=24, the total amount of fields required is 10 * (2 * (72/24) + 1) = 70.
