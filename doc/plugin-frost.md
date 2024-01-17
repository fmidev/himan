# Summary

frost plugin calculates frost probability.

# Required source parameters

* temperature (T-K)
* dew point temperature (TD-K)
* ground temperature (TG-k)
* probability of temperature below 0 from ECMWF (PROB-TC-0)
* probability of temperature below 0 from MEPS (PROB-TC-0)
* wind gust (FFG-MS)
* cloudiness (N-PRCNT)
* radiation (RADGLO-WM2)
* sea ice (IC-0TO1)
* land cover (LC-0TO1)
* elevation angle of the sun
* date

# Output parameters

PROB-FROST-1

# Method of calculation

According to method provided by Juha Jantunen.

# Per-plugin configuration options

All options are optional.

ecmwf_geometry: define geometry for ecmwf data

    "ecmwf_geometry" : "..."

ecmwfeps_geometry: define geometry for ecmwf eps data

    "ecmwfeps_geometry" : "..."

meps_geometry: define geometry for meps data

    "meps_geometry" : "..."

