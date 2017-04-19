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

Precipitation form is determined in the following order:

1. Pre-conditions
  Model needs to have precipitation (RR>0; RR = rainfall + snowfall, [RR]=mm/h)

2. **Freezing drizzle** if
  RR <= 0.2
  -10 < T2m < 0
  precipitation is not convective
  stratus exists (base<300m and quantitity at least 5/8)
  weak lifting at stratus (0 < wAvg < 50mm/s)
  stratus is thick enough (dz > 800m)
  temperature at stratus top Ttop > -12C
  average temperature at stratus avgT > -12C
  dry layer above stratus (thickness > 1.5km, where N < 30%)

3. **Freezing rain** if
   T2m <= 0
   thick enough melting layer above surface (area > 100mC [meters * C])
   thick enough freezing layer below melting layer (area < -100mC)
   if stratus exists, melting layer above it must not be dry

4. **Drizzle** or **water**, if
    melting layer above surface

    * **Drizzle** if
      RR <= 0.3
      stratus (base < 300m and quantity at least 5/8)
      stratus thick enough (dz > 500m)
      dry layer above stratus (dz > 1.5km, where N < 30%)

    * Otherwise **water**

    * If surface melting layer is dry (rhAvg < rhMelt), form is **sleet**

4. **Sleet** if
  thin enough melting layer above surface

    * If surface melting layer is dry (rhAvg<rhMelt), form is **snow**

5. Otherwise **snow**
  Only thin melting layer above surface is allowed

# Per-plugin configuration options

potential precipitation form: calculate form even if no precipitation is forecasted

    "potential_precipitation_form" : "true"
