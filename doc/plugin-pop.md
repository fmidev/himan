# Summary

pop plugin calculates the probability of precipitation (any amount).

# Required source parameters

* precipitation rate from the primary model of the configuration.
* PROB-RR1-1 from ecgepsmta, if disable_meps=true (limit 0.14mm/h)
* PROB-RR-7 from mepsmta, if disable_meps=false (limit 0.025mm/h) 
* PROB-RR-4 from ecgepsmta (limit 0.2mm/3h)
* PROB-RR3-6 from ecgepsmta (limit 0.4mm/6h)

# Output parameters

POP-0TO1

Unit of resulting parameter is a number between 0..1.

# Method of calculation

Use probabilities derived from ensemble forecasts. Source data is MEPS for short leadtimes and ECMWF
for longer ones.

Plugin can be forced to use only ECMWF data.

# Per-plugin configuration options

select meps probability data geometry name

    meps_geom: >geometry_name>

select ecmwf probability data geometry name

    ecgeps_eom: <geometry_name>

replace meps with ecmwf data

    disable_meps: true

