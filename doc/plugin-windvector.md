# Summary

windvector plugin is used to calculate speed and direction from components. Source data can be wind, wind gust, ice or ocean current.

Historically this plugin was also responsible for rotating the vectors of projected areas from grid north to earth north; this code has been later moved to himan-core and this plugin only accesses that code.

Plugin is optimized for GPU use.

# Required source parameters

* u = vector component in west-east direction (east positive) (m/s)
* v = vector component in sout-north direction (north positive) (m/s)

# Output parameters

wind: FF-MS (m/s) and DD-D (degrees)
wind gust: FFG-MS (m/s)
ice: IFF-MS (m/s) and IDD-D (degrees)
sea currents: SFF-MS (m/s) and SDD-D (degrees)

# Method of calculation

    speed = sqrt(u^2 + v^2)
    offset = 180 if wind or wind_gust
    dir = atan2(u,v) + offset

# Per-plugin configuration options

for_ice: define if ice speed and direction should be calculated (default: false)

    "for_ice" : true

for_sea: define if sea current speed and direction should be calculated (default: false)

    "for_sea" : true

for_gust: define if wind gust speed should be calculated (default: false)

    "for_gust" : true

for_wind: define if wind speed and direction should be calculated (default: true)

    "for_wind" : true

Only one options can be set per plugin call.