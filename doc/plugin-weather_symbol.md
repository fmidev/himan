# Summary

weather_symbol plugin is used to calculate a weather code, a single number to describes the current weather. This is a legacy parameter and is used only as input to oother parameters. For a better weather symbol algorithm, one should check plugin weather_code_2.

# Required source parameters

* cloud symbol (code)
* rain code (code)

# Output parameters

HESSAA-N

Unit of resulting parameters is a code table.

* 1 = clear sky or little cloud 
* 2 = moderate cloud cover, no precipitation
* 3 = overcast sky, no precipitation
* 21 = moderately cloudy, occasional weak rain (continuous or showers)
* 22 = moderately cloudy, occasional rain (continuous or showers)
* 23 = moderately cloudy, occasianl heavy rain (continuous or showers)
* 31 = overcast sky, occasional weak rain (continuous or showers)
* 32 = overcast sky, rain
* 33 = overcast sky, heavy rain
* 51 = moderately cloudy, occasional weak snow fall
* 52 = moderately cloudy, occasional moderate snow fall
* 53 = overcast sky, heavy snow fall
* 61 = weak thunderstorms
* 62 = heavy thunderstorms

# Method of calculation

# Per-plugin configuration options

None
