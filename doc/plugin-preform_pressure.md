# Summary

preform_pressure plugin calculates the form of precipitation using only surface and pressure level information. Possible values are drizzle, rain, sleet, snow, freezing drizzle and freezing rain.

# Required source parameters

* air temperature (K)
    * surface, pressure levels 700, 850, 925
* relative humidity (%)
    * surface, pressure levels 700, 850, 925
* snow fall rate (mm/h)
* precipitation rate (mm/h)
* surface pressure (Pa)
* vertical velocity (m/s)
    * pressure levels 850, 925

# Output parameters

PRECFORM-N

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

None