# himan

Himan is a weather parameter post processing tool.

# Properties

* Forecast provider agnostic post processing
* Able to produce several useful weather parameters not produced by the forecasts themselves
* Main design principles are performance and maintainability
* Written in C++11
* Partial support for GPU's through Cuda
* Actively developed and supported by the Finnish Meteorological Institute

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

Himan can be either built from source, or installed using pre-built rpm packages (add link here). The latter is recommended for a quick start.

In operative environments Himan relies heavily on a database that's providing all data and metadata. This database schema will be open sourced later this year. In the meanwhile, Himan can be tested using a "no database" mode.

Example: running seaicing plugin for Hirlam data. Seaicing plugin calculates an index that describes the amount of ice that is built up on a ship's superstructure. Files for this example are located at example/seaicing.

## Define a json configuration

```
{
	"bbox" : "5,45,30,65",
	"scanning_mode" : "+x+y",
	"ni" : "150",
	"nj" : "150",
	"source_producer" : "999999",
	"target_producer" : "999999",
	"hours" : "3",
	"origintime" : "2017-04-05 00:00:00",
	"file_write" : "multiple",

	"processqueue" : [
	{
		"leveltype" : "height",
		"levels" : "0",
		"plugins" : [ { "name" : "seaicing" } ]
	} 
	]
}
```

Breaking the configuration into pieces:

```
	"bbox" : "5,45,30,65",
	"scanning_mode" : "+x+y",
	"ni" : "150",
	"nj" : "150",
```

The target area for the calculation has lower-left longitude 5 degrees and latitude 10 degrees. The corresponding top-right coordinates are 30,65. Scanning mode for this grid is +x+y, which means that reading starts from the bottom left corner and goes right and up. The grid for this area has size 150x150 grid points. Source area does not need to be defined: Himan will determine it from the source file and will interpolate all data to target area and grid. Himan supports a few of the most common projections used in the meteorological domain. In operational environments it might be prudent (and faster!) to produce the data in the same grid as the source data.


```
	"source_producer" : "999999",
	"target_producer" : "999999",
```

Both source and target producer id's are 99999. This is corresponds to missing value: because the example is run using no-database mode, Himan is not able to fetch the real producer ids.

```
	"hours" : "3",
	"origintime" : "2017-04-05 00:00:00",
```

The analysis time (or origin time) for the forecast is 5th of April, 2017 00z. The calculation is done for forecast hour (leadtime) 3, ie. valid time is 2017-04-05 03:00:00.

```
	"file_write" : "multiple",

```

Output data is written so that each field is written to a separate file. In the context of this example this does not really matter because the calculation outputs only one field. The default output file type is Grib.

```

	"processqueue" : [
	{
		"leveltype" : "height",
		"levels" : "0",
		"plugins" : [ { "name" : "seaicing" } ]
	} 
	]
```

The processqueue (list of plugins that are executed) consists only one plugin: seaicing. The resulting data is written to leveltype height/0.

## Run Himan

```
$ himan -f seaicing.json --no-database --param-file param-file.txt seaicing.grib

************************************************
* By the Power of Grayskull, I Have the Power! *
************************************************

Info::himan Found 46 plugins
Debug::himan Processqueue size: 1
Info::himan Calculating seaicing
Info::compiled_plugin: Thread 1 starting
Info::seaicingThread #1 Calculating time 201704050300 level height/0
Debug::fetcher Start full auxiliary files read
Debug::grib Read file 'seaicing.grib' (36 MB/s)
Debug::fetcher Auxiliary files read finished, cache size is now 3
Info::seaicingThread #1 [CPU] Missing values: 0/22500
Info::grib Wrote file './ICING-N_height_0_ll_150_150_0_003.grib' (1 MB/s)
 
```

## Check contents

```
$ grib_histogram ICING-N_height_0_ll_150_150_0_003.grib 

min=0 max=3 size=22500
 0:0.3 0.3:0.6 0.6:0.9 0.9:1.2 1.2:1.5 1.5:1.8 1.8:2.1 2.1:2.4 2.4:2.7 2.7:3
 21606 0 0 883 0 0 10 0 0 1
```

Note that seaicing plugin does not separate land points from sea points: the index is mostly zero due to warm conditions on the baltic sea.

# Contributing

* Feel free the submit issues or feature requests
* Pull requests are accepted
* CLA is required for code contribution
