# Himan

Himan is a weather parameter post processing tool.

# Properties

* Forecast provider agnostic post processing
* Able to produce several useful weather parameters not produced by the forecasts themselves
* Main design principles are performance and maintainability
* Written in C++11
* Partial support for GPU's through Cuda
* Actively developed and supported by the Finnish Meteorological Institute
* yum repository available for RHEL7
* Licensed under MIT license

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
* potential temperature
* pseudo-adiabatic potential temperature
* equivalent potential temperature
* vertical velocity in meters per seconds
* speed and direction from vector components
* density of dry air
* liquid and solid precipitation from mixing ratios
* absolute humidity
* sounding indices
  * CAPE
  * CIN
  * LCL, LFC, EL height and temperature
  * wind shear
* weather symbol (two different versions)
* monin-obukhov length
* wind gust
* qnh pressure
* cloud ceiling in feet
* probability of thunderstorms
* snow depth
* toplink winter weather indices 2 & 3
* modified clear air turbulence indices 1 & 2
* lapse rate
* cloud types
* visibility
* fractiles from ensemble
* probabilities from ensemble
* probability of precipitation
* turbulent kinetic energy
* unstaggering of Arakawa C grids

# Architecture

Himan is split into three separate but not independent components: executable (himan-bin), library (himan-lib) and plugins (himan-plugins).

The binary (himan) is a simple frontend and does little else but parse the command line options and initiate the execution of plugins listed in the configuration file.

The library (libhiman.so) contains common code that is used by the plugins. This common code includes for example interpolation routines, json parser, metadata classes such as parameter and level, meteorological formulas and so forth.

The plugins (libX.so) contain the actual core and idea of Himan. Each plugin is a shared library that exposes only one function that himan executable is calling. Everything else is free game for the plugin. So there is a very large degree of freedom for a plugin to do whatever it needs to do. All plugins share code from a common parent class which reduces the amount of boiler plate code; the common code can be overwritten if needed.

Plugins are split into two types: helper (auxiliary) plugins and core ("compiled") plugins. The helper plugins provide several necessary functions for the other plugins to interact with the environment. These functions are for example fetching and writing data, accessing databases, storing and fetching data from cache and so forth. Two separate but important helper plugins are hitool and luatool. Hitool exposes functions that examine the state of the atmosphere in a certain way. For example it can be used to get minimum/maximum value of some parameter in a given vertical height range. Height can be specified in meters or in Pascals. Luatool plugin is a shallow lua-language wrapper on top of Himan. With luatool one can create lua-language scripts which are much faster to write and more flexible than the C++-based plugins, especially if the given task is light weight.

The purpose of the core plugins is to provide some output data given required input data. The range of operations stretches from very simple (calculate radiation power from accumulation) to more complex algorithms (determine the precipitation form). Certain plugins that operate with hybrid levels are also optimized for Nvidia GPU processors. This an addition to the normal CPU based functionality: Himan does function perfectly without GPU's.

# The name

Himan stands for HIlojen MANipulaatio, "grid manipulation" in Finnish.

# Getting started

[Getting started](https://github.com/fmidev/himan/tree/master/doc/getting-started.md)

# Contributing

* Feel free the submit issues or feature requests
* Pull requests are accepted
* CLA is required for code contribution

# Communication

You may contact us from following channels:

* Email: beta@fmi.fi
* Facebook: https://www.facebook.com/fmibeta/
* GitHub: issues

Our public web pages: http://en.ilmatieteenlaitos.fi/open-source-code
