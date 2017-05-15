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

Precipitation form is determined in the following order:

1. Pre-conditions
  Model needs to have precipitation (RR>0; RR = rainfall + snowfall, [RR]=mm/h)

2. **Freezing drizzle** if
   * -10 < T2m <= 0C
   * freezing stratus (~-10 < T < -0)
   * lifting motion in stratus
   * weak precipiation
   * no precipitation from middle clouds

3. **Freezing rain** if
   * T2m <= 0C
   * warm layer T > 0C above surface
   * layer must have high enough humidity

4. **Snow** if
   * snowfall / RR > 0.8 or tai T <= 0C

5. **Sleet** if
   * 0.15 < snowfall / RR < 0.8

6. **Rain** or **drizzle** if
   * snowfall / RR < 0.15

    * **Drizzle** if
       * stratus with small precipitation intensity
       * no middle cloud

# Per-plugin configuration options

None