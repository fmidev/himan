# Summary

snow_drift plugin calculates snow drift index.

# Required source parameters

* meter air temperature (K)
* wind speed (m/s)
* snow fall rate (mm/h)

# Output parameters

SNOWDRIFT-N

Parameter can have values:

* 0 - no drift
* 1 - light drift
* 2 - moderate drift
* 3 - high drift

# Method of calculation

Algorithm is developed by

* M. Hippi (Finnish Meteorological Institute, Helsinki, Finland)
* S. Thordarson (Vegsyn Consult, Hafnarfjordur, Iceland) 
* H. Thorsteinsson (Icelandic Meteorological Office, Ísafjörður, Iceland)
 
http://sirwec.org/wp-content/uploads/Papers/2014-Andorra/D-38.pdf

When calculating snow drift index forecast, the calculation is 'calibrated'
with observed DA & SA values (for more details about these acronyms see the paper).

This means that the plugin needs to be calculated to at least one observation
analysis producer (ie. currently LAPS, METAN, MNWC, MESAN etc), and any number 
of forecast producers.

Ice cover data is read from observation analysis (Icemap2 in FMIs case) and
if ice cover is less than 70%, snow drift is not calculated for that grid point.
If ice cover value is missing (for example land point), snow drift is calculated
normally. If ice coverage data is not found, land-sea mask is used with threshold 0.5.

# Per-plugin configuration options

Key word "reset" can be used to force a reset for the calculations (does not try
to fetch previous data)

    "reset" : true


