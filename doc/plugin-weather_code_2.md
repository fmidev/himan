# Summary

weather_code_2 plugin is used to calculate a weather code, a single number to describes the current weather. 

# Required source parameters

* total cloudiness (%)
* low cloudiness (%)
* middle cloudiness (%)
* high cloudiness (%)
* cloud symbol (code)
* fog symbol (code)
* precipitation form (code)
* air temperature (K)
    * ground, pressure level 850
* precipitation sum (mm/h)
* k-index (code)
* humidity (%)
    * pressure level 850

# Output parameters

ILSAA1-N

Unit of resulting parameters is a code table.

cloudiness

* 0 = clear (total cloudiness 0%)
* 1 = weak high cloud
* 2 = almost clear (10-30%)
* 3 = half cloudy (30-60%)
* 4 = almost cloudy (70-80%)
* 5 = cloudy (>80%)

haze

* 6 = haze

mist

* 10 = mist

sandstorm

* 34 = sandstorm

fog

* 40 = fog

drizzle

* 51 = weak drizzle
* 53 = drizzle
* 56 = weak freezing drizzle
* 57 = freezing drizzle

rain

* 61 = weak rain
* 63 = moderate rain
* 65 = heavy rain

freezing rain

* 66 = weak freezing rain
* 67 = freezing rain

sleet

* 68 = weak sleet
* 69 = moderate or heavy sleet

snow

* 71 = weak snow fall
* 73 = moderate snow fall
* 75 = heavy snow fall

showers

* 80 = weak showers
* 81 = moderate or heavy showers
* 82 = extreme showers
* 83 = weak sleet showers
* 84 = moderate or heavy sleet showers
* 85 = weak snow showers
* 86 = moderate or heavy snow showers

thunderstorms

* 95 = weak or moderate thunderstorms
* 96 = weak or moderate thunderstorms including hail
* 97 = heavy thunderstorms
* 98 = heavy thunderstorms including hail

# Method of calculation

# Per-plugin configuration options

None
