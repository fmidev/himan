# Summary

weather_code_1 plugin is used to calculate a rain code. Yes, that is correct. Despite the name, the output parameter from this plugin actually describes only weather in terms of rain. For an actual weather code algorithm, one should check plugin weather_code_2.

# Required source parameters

* total cloudiness (%)
* geopotential (m^2/s^2)
    * pressure levels 1000, 850
* air temperature (K)
    * surface, pressure level 850
* cloud symbol (code)
* precipitation sum (mm/h)
* k-index (code)

# Output parameters

HSADE-N

Unit of resulting parameters is a code table.

The output values follow the present weather parameter numbering from manual weather stations:

* 0 = no rain
* 50 = Drizzle, not freezing, intermittent, slight at time of ob
* 51 = Drizzle, not freezing, continuous, slight at time of ob
* 56 = Drizzle, freezing, slight
* 57 = Drizzle, freezing, moderate or heavy (dense)
* 60 = Rain, not freezing, intermittent, slight at time of ob
* 61 = Rain, not freezing, continuous, slight at time of ob
* 63 = Rain, not freezing, continuous, moderate at time of ob
* 65 = Rain, not freezing, continuous, heavy at time of ob
* 66 = Rain, freezing, slight
* 67 = Rain, freezing, moderate or heavy
* 68 = Rain or drizzle and snow, slight
* 70 = Intermittent fall of snowflakes, slight at time of ob
* 71 = Continuous fall of snowflakes, slight at time of ob
* 73 = Continuous fall of snowflakes, moderate at time of ob
* 75 = Continuous fall of snowflakes, heavy at time of ob
* 78 = Isolated star-like snow crystals (with or without fog)
* 80 = Rain shower(s), slight
* 82 = Rain shower(s), violent
* 85 = Snow shower(s), slight
* 86 = Snow shower(s), moderate or heavy
* 95 = Thunderstorm, slight or moderate, without hail, but with rain and/or snow at time of observation
* 97 = Thunderstorm, heavy, without hail, but with rain and/or snow at time of observation

# Method of calculation

# Per-plugin configuration options

None
