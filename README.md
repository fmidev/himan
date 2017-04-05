# himan

Himan is a weather parameter post processing tool.

# Properties

* Forecast provider -agnostic post processing
* Able to produce several useful weather parameters not produced by the forecasts themselves
* Main design principles are speed and maintainability
* Written in C++11
* Partial support for GPU's through Cuda
* Actively developed and support by Finnish Meteorological Institute

# Parameter list

Himan is able to produce the following parameters, given required source data:

* cloud code
* dewpoint
* fog type
* pressure at hybrid level
* (metric) height of hybrid level
* icing in the atmosphere
* icing at sea (sea spray icing)
* stability indices
  * k-index
  * lifted index
  * vertical totals index
  * cross totals index
  * total totals index
  * showalter index
* height of 0 and -20 isotherms
* precipitation form using pressure level information (less accurate)
* precipitation form using hybrid level information (more accurate)
* precipitation type (continuous, showers)
* relative humidity
* accumulated precipitation and radiation to rates and powers
* potential temperarute
* pseudo-adiabatic potential temperature
* equivalent potential temperature
* vertical velocity in meters per seconds
* speed and direction from vector components
* density of dry air
* liquid and solid precipitation from mixing ratios
* absolute humidity
* CAPE
* CIN
* wind shear
* weather symbol
* monin-obukhov length
* wind gust
* qnh pressure
* cloud ceiling in feet
* probability of thunderstorms
* snow depth
* toplink winter weather indices 2 & 3
* modified clear air turbulence indices 1 & 2
* lapse rat
* cloud types
* visibility
* fractiles from ensemble
* probabilities from ensemble
* probability of precipitation

# Contributing

* Feel free the submit issues or feature requests
* Pull requests are accepted
* CLA is required for code contribution
