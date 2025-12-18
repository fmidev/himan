# Summary

preform_hybrid plugin calculates the form of precipitation using surface and hybrid level information. Possible values are drizzle, rain, sleet, snow, freezing drizzle and freezing rain.

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

## PRECFORM2-N 

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

GRIB2 numbering scheme follows the [WMO standard](https://codes.ecmwf.int/grib/param-db/260015). 

## POTPRECFORM-N

Unit of resulting parameter is a code table.

The output values are:

| Potential Precipitation Form | Value |
|---|---|
| drizzle | 0 | 
| rain  | 1 |
| sleet  | 2 | 
| snow | 3 | 
| freezing drizzle | 4 | 
| freezing rain | 5 |

# Method of calculation

Precipitation form is determined in the following order:

1. Pre-conditions
  Model needs to have precipitation (RR>0; RR = rainfall + snowfall, [RR]=mm/h)

2. **Freezing drizzle** if
   * RR <= 0.2
   * -10 < T2m < 0
   * stratus exists (base<305m and quantitity at least 4/8)
   * weak lifting at stratus (0 < wAvg < 50mm/s)
   * stratus is thick enough (dz > 700m)
   * temperature at stratus top > -12C
   * average temperature at stratus > -12C
   * dry layer above stratus (thickness > 1.5km, where N < 30%)

3. **Freezing rain** if
   * T2m <= 0
   * thick enough melting layer above surface (area > 100mC [meters * C])
   * thick enough freezing layer below melting layer (area < -100mC)
   * if stratus exists, melting layer above it must not be dry

4. **Drizzle** or **water**, if
   * melting layer above surface (area > 200mC)

    * **Drizzle** if
      * RR <= 0.3
      * stratus (base < 305m and quantity at least 4/8)
      * stratus thick enough (dz > 400m)
      * dry layer above stratus (dz > 1.5km, where N < 30%)

    * Otherwise **water**

    * If surface melting layer is dry (rhAvg < rhMelt), form is **sleet**

4. **Sleet** if
    * thin enough melting layer above surface (50 mC < area < 200mC)

    * If surface melting layer is dry (rhAvg<rhMelt), form is **snow**

5. Otherwise **snow**
   * Only thin melting layer above surface is allowed (area < 50mC)

# Per-plugin configuration options

potential precipitation form: calculate form even if no precipitation is forecasted

    "potential_precipitation_form" : "true"
