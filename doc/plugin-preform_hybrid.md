# Summary

preform_hybrid plugin calculates the form of precipitation using surface and htbrid level information. Possible values are drizzle, rain, sleet, snow, freezing drizzle and freezing rain.

# Required source parameters

Surface 

* air temperature (K)
* relative humidity (%)
* precipitation rate (mm/h)

Hybrid levels (up to 5000m)

* vertical velocity (m/s)
* pressure (Pa)
* height (m)
* relative humidity (%)
* total cloud cover (%)

# Output parameters

PRECFORM2-N and POTPRECFORM-N.

Unit of resulting parameter is a code table.

The output values are dependend on output file type.

| Precipitation Form | Querydata | GRIB1 | GRIB2 |
|---|---|---|---|
| drizzle | 0 | 0 | 11 | 
| rain  | 1 | 1 | 1 |
| sleet  | 2 | 2 | 7 | 
| snow | 3 | 3 | 5 | 
| freezing drizzle | 4 | 4 | 12 | 
| freezing rain | 5 | 5 | 3 | 


# Method of calculation

Algorithm checking the weather conditions on earths surface and upper and lower atmosphere.

# Per-plugin configuration options

potential precipitation form: calculate form even if no precipitation is forecasted

    "potential_precipitation_form" : "true"
