Configuring himan is done through json-files, and all the common rules of writing json apply:

* (Almost) all keys and values must be enclosed in quotes ("")
* Key and value are separated with colons
* Elements are separated by commas

The json file can be divided into two parts: the global part, configuration apply for all himan operations, ant the local part (processqueue) where some of the global options can be overwritten and extra options given.

##### Table of Contents  
* [Target area](#Target_area)  
  * [Method 1: Using pre-defined areas from database ](#Target_area_1)
  * [Method 2: Defining area with `bbox` ](#Target_area_2)
  * [Method 3: Defining area in detail ](#Target_area_3)
  * [Method 4: List of points ](#Target_area_4)
* [Target grid](#Target_grid)
* [Source area and grid](#Source_area_and_grid)
* [Producers](#Producers)
* [Time](#Time)
* [Analysis time](#Time_1)
  * [Lead time method 1: Listing hours](#Time_2)
  * [Lead time method 2: Setting start and stop values](#Time_3)
* [Levels](#Levels)
* [Plugins](#Plugins)
* [Other parametrization](#Other_parametrization)
  * [File write options](#File_write_options)
  * [Database access](#Database_access)
  * [Forecast types](#Forecast_types)
  * [Memory usage](#Memory_usage)
  * [Asynchronous execution](#Asynchronous_execution)
* [Full examples](#Full_examples)

<a name="Target_area"/>

# Target area

Target area can be defined in four different ways:
* Using pre-defined areas from database
* Defining an area with key `bbox`
* Defining area coordinates in latlon
* Defining a list of points or station ids

<a name="Target_area_1"/>

## Method 1: Using pre-defined areas from database 

The key target_geom_name needs to match the database geometry name.

    "target_geom_name" : "GFS0500"

**If this method is used, grid information does not need to be added separately because that is also found from the database.**

<a name="Target_area_2"/>

## Method 2: Defining area with `bbox`

An area can be defined by listing bottom left and top right coordinates in latitude and longitudes (lon,lat,lon,lat). This method will automatically force the grid type ("projection") to be latlon, plate carree.

    "bbox" : "25,60,27,65"

bottom left: lon=25, lat=60

top right: lon=27, lat=60

Grid information needs to be defined separately when using `bbox`.

<a name="Target_area_3"/>

## Method 3: Defining area in detail

Himan supports following grid types or projectios: 
* latitude longitude
* rotated latitude longitude
* stereographic
* reduced gaussian
* lambert conformal conic

Target area is defined by giving bottom left and top right coordinates, wanted projection and projection special parameters.

Projection is defined with key `projection`:

    "projection" : "stereographic" | "latlon" | "rotated_latlon | reduced_gg | lcc"

Area corners area bottom left and top right latitude longitude values.

    "bottom_left_latitude" : "<degrees>",
    "bottom_left_longitude" : "<degrees>",
    "top_right_latitude" : "<degrees>",
    "top_right_longitude" : "<degrees>",

In stereographic projectio, the orientation of the area is given with key orientation

    "orientation" = "<degrees>"

In rotated latlon projection the coordinates of south pole are given with keys `south_pole_longitude` and  `south_pole_latitude`

    "south_pole_longitude" : "<degrees>",
    "south_pole_latitude" : "<degrees>",

Example:

    "projection" : "stereographic",
    "bottom_left_latitude" : "51.3",
    "bottom_left_longitude" : "6.0",
    "top_right_latitude" : "70.2",
    "top_right_longitude" : "49",
    "orientation" : "20"

    "projection" : "rotated_latlon",
    "bottom_left_latitude" : "-24",
    "bottom_left_longitude" : "-33.5",
    "top_right_latitude" : "31.42",
    "top_right_longitude" : "36.47",
    "south_pole_longitude" : "0",
    "south_pole_latitude" : "-30",

<a name="Target_area_4"/>

## Method 4: List of points

List of points isn't exactly an area, but Himan treats it as one. It can be defined in two ways: 
* with key `points`, point coordinates are given in the configuration file, or 
* with key `stations`, point coordinates are fetched from database.

`points` key can be used in two different way, either specifying just the coordinates or by defining coordinates and and id and name for a point. Different points are separated with commas, different values for a point are separated with a whitespace. Order of in-point elements is: id name longitude latitude.

    "points" : "25 60,27 61",

    "points" : "1 Helsinki 25 60, 2 Kotka 27 61",

With key stations Himan fetches point (station) information from a database based on the given id.

    "stations" : "100971,101030",

<a name="Target_grid"/>

# Target grid

Target grid size is defined with keys `ni`and `nj`.

**Note! If target area was defined using database information, grid does not need to be defined separately.**

    "ni" : "<number of grid points in x-axis direction>",
    "nj" : "<number of grid points in y-axis direction>"

Grid scanning mode (direction of reading) is given with key `scanning_mode`. Possible values are
* `+x+y`, reading starts at bottom left corner, and 
* `+x-y`, reading starts at top left corner

`"scanning_mode" : "+x+y" | "+x-y"`

Example:

    "ni" : "220",
    "nj" : "220",
    "scanning_mode" : "+x+y"

<a name="Source_area_and_grid"/>

# Source area and grid

If source data is read from database (as is the case in FMI operations), key `source_geom_name` can be used to define one or more source areas. By default (if key is not set) Himan will read any geometry it founds from the database that matches the producer. Key `source_geom_name` can contain multiple values separated with a comma, and the order of the geometries is preserved. 

Note that if multiple geometries exist for a producer, and the first read does not cover the target area completely, Himan will not try amend the data using other geometries. Therefore the order of the geometries does matter! 

    "source_geom_name" : "GEOM1,GEOM2"

Example:

    "source_geom_name" : "ECEDIT125,MTEDIT125"

    "source_geom_name" : "ECEDIT125"

<a name="Producers"/>

# Producers

Producer numbers are found from the database (they are not simply the number found from a grib).

Source producer is defined with key `source_producer`, and it can contain multiple values. Data is read repeatedly from all defined producers in the order they are specified, until correct data is found.

Target producer is a single value.

    "source_producer" : "<source producer id(s)>"
    "target_producer" : "<target producer id>"

Note! The more producers are defined, the longer the fetching process will last, especially if the producer that the data is found from is the last in the list.

Example:

    "source_producer" : "1",
    "target_producer" : "230"
    

    "source_producer" : "1,230",
    "target_producer" : "230"

<a name="Time"/>

# Time

Time information contains forecast analysis time and the forecast lead times (steps). Lead time is always an offset from the analysis time.

Lead times can be listed in two ways:
* listing all hours that should be calculated, or
* listing start time and stop time and a step

Time must be a top level element, but it can also be specified in processqueue level.

<a name="Time_1"/>

## Analysis time

Analysis time is given with key `origintime`, using standard timestamp format `YYYY-MM-DD HH24:MI:SS` (or `%Y-%m-%d %H:%M:%S`). Key value can also be `latest`, then Himan will try to fetch the latest forecast found from the database. If penultimate forecast is required, value `latest-1` can be used.

    "origintime : "latest"|"latest-X"|"<timestamp>"

Multiple origin times can also be specified with key `origintimes`.

    "origintimes" : "<timestamp>,<timestamp>,..."

<a name="Time_2"/>

## Lead time method 1: Listing hours

With key `hours` the hours (lead times) that should be calculated are listed. Values are separated with a comma, if a dash ('-') is used, Himan will interpolate values to fill the gap.

    hours : <list of hours>

Example:

    "hours" : "1,2,3-8"

This example will result to hours 1,2,3,4,5,6,7 and 8.

<a name="Time_3"/>

## Lead time method 2: Setting start and stop values

Starting hour is defined with key `start_hour` .

    "start_hour" : "<start hour as offset from analysis time>"

Stop hour is defined with key `stop_hour` (value is inclusive).

    "stop_hour" : "<stop hour as offset from analysis time>"

If forecast has minute resolution, the corresponding keys are `start_minute` and `stop_minute`. 

The step time resolution is always the same that is given with keys `start_x` and `stop_x`. The step is given with key `step`.

    "step" : "<step>,"

Example:

    "origintime" : "latest",
    "start_hour" : "0",
    "stop_hour" : "120",
    "step" : "3",


    "origintime" : "2016-09-22 00:00:00",
    "start_minute" : "1440",
    "stop_minute" : "1470",
    "step" : "15",

<a name="Levels"/>

# Levels

Target level for calculation is defined with key `leveltype`.

    "leveltype" : "height" | "pressure" | "hybrid" | "ground"

The level value is given with key `levels`, values are separated with a comma, and if a dash is given Himan will try to fill the gap.

    "levels" : "<list of levels>",

Most (but not all) plugins respect the level definition and will use it when processing data. Notable exceptions are ate least cape-plugin.

Example:

    "leveltype" : "hybrid"
    "levels" : "4,5,6,7-10"

<a name="Plugins"/>

# Plugins

Started plugins are listed as json dictionary in a json table. The order of plugins is preserved.

Plugin name can be given by using the full path (`himan::plugin::pluginin_name` or just the shorter name `plugin_name`. The only mandatory key for the dictionary is `name`; all other possible keys are read and passed on the called plugin. It is the responsibility of the plugin the to handle the extra parameters.

Plugins are defined with key `plugins`

    "plugins" : [ { "name" : "plugin" ] ]

Example:

    "plugins" : [ { "name" : "tpot" } ]

    "plugins" : [ 
      { "name" : "hybrid_pressure" },
      { "name" : "windvector", "for_air" : true } 
    ]

<a name="Processqueue"/>

# Processqueue

The list of levels and plugins are tied with key `processqueue`. That is a defining key in the whole json and contains the ordered list of processed items. Some of the global options can be overwritten in processqueue  level. 

The required keys for a processqueue item are: `leveltype`, `levels` and `plugins`. Beside these, other options can be defined.

    "processqueue" : [ 
      { 
        "leveltype" : "<leveltype>",
        "levels" : "<level value>",
        "plugins" : ["<list of plugins>"] 
      } 
    ]

Example:

    "processqueue" : [
      {
        "use_cache" : false,
        "leveltype" : "height",
        "levels" : "0",
        "plugins" : [	{ "name" : "split_sum", "rrr" : true } ]
      },
      {
        "leveltype" : "height",
        "levels" : "2",
        "plugins" : [   
          { "name" : "relative_humidity" }
        ]
      },
      {
        "leveltype" : "hybrid",
        "levels" : "1-65",
        "plugins" : [ 	
          { "name" : "hybrid_pressure" },
          { "name" : "hybrid_height" }
        ]
      }

<a name="Other_parametrization"/>

# Other parametrization

<a name="File_write_options"/>

## File write options

Output file type is controlled with key `file_type`. This information can also be given as a command line argument for Himan executable. The command line value will override configuration file values.

    "file_type" : "grib|grib1|grib2|fqd"

Possible values are:

* `grib` or `grib1`: grib edition 1
* `grib2`: grib edition 2
* `fqd`: querydata, fmi data format compatible with Smartmet Workstation

Default value for key is `grib`.

The write destination is defined with key `file_write`. Possible values are:

* `single`, all data is written to a single file
* `multiple`, each grid is written to a separate file
* `database`, same as multiple but the file information is also stored to database
* `cache only`, file is only written to cache

`"file_write" : "single | multiple | database | cache only",`

Default value for key is `single`. Key can be set in the global or processqueue scope.

Himan can pack the written files with either gzip or bzip2 methods. The key that controls this is called `file_compression`. 

    "file_compression" : "none | gzip | bzip2",

Default value for key is `none`.

<a name="Database_access"/>

## Database access

Himan can be forced to read data from command line given arguments only. This behavior can be set with key `read_data_from_database`. Default value for key is `true`.

    "read_data_from_database" : true | false

Example:

    "read_data_from_database" : false,

<a name="Forecast_types"/>

## Forecast types

With key `forecast_type` the source forecast type is set. Using this, Himan can be used to calculate parameters to for example for all ensemble members. Default value for key is `deterministic`, meaning that Himan will read data from deterministic forecasts only.

    "forecast_type" : "deterministic" | "cf" | "pf1-2"

With value `cf` the control forecast of an ensemble is chosen, and with value `pfVALUE-VALUE` a range of ensemble members (permutations) are chosen. For example the value `pf1-10` means that permutation 1,2,3,4,5,6,7,8,9,10 are read. Values can be combined with a comma.

Key can be set in the global or processqueue scope.

Example:

    "forecast_type" : "deterministic",

    "forecast_type" : "cf,pf1-50",

<a name="Memory_usage"/>

## Memory usage

Himan uses an in-memory cache for all read and written data. The cache can be turned off with key `use_cache`. Default value is `true`.

    "use_cache" : "true | false",

Key can be set in the global or processqueue scope.

Example:

    "use_cache" : false,

Memory cache size can be controlled with key `cache_limit`. The value of the key specifies the maximum number of fields (grids) Himan will hold in memory. When the limit is reached, data is evicted using an LRU algorithm. Valid values for key are >= 1. The actual size of the cache in bytes depends on the size of the grids. 

    "cache_limit" : "<integer value larger than 0>",

Default value for key is -1, ie. no upper limit for cache size.

Example:

    "cache_limit" : "200",

By default himan will allocate all necessary memory when it starts. In low-memory environments this might be problematic. With key `dynamic_memory_allocation`, Himan can be forced to allocate memory dynamically (reserving it just before needed, and releasing immediately afterwards).

    "dynamic_memory_allocation" : true | false,

Default value is `false`.

<a name="Asynchronous_execution"/>

## Asynchronous execution

By default Himan plugins are executed in a serialized fashion, mostly because many of the plugins are dependent on the output of others. Some plugins, however, lie necessary at the end of the execution line, and when those plugins are in the critical chain of execution they will (unnecessarily) slow down the the total execution of Himan. 

This can be avoided by using the configuration file key 'async'. By default the value is false, confirming to the serialized execution. It can be set to true for any individual plugin or processqueue element scope.


                {
			"async" : true,
                        "leveltype" : "...",
                        "levels" : "...",
                        "plugins" : [ { "name" : "...", "async" : false } ]
                }

Asynchronous execution should only be enabled for those plugins that have no other plugins as dependants!

<a name="Full_examples"/>

# Full examples

The examples are using FMI defined geometry names and producers ids.

Calculating weather code -parameter from ECMWF forecast, using two source geometries and two source producers.

    {
        "source_geom_name" : "ECEDIT125,MTEDIT125",
        "source_producer" : "131,240",
        "target_producer" : "240",
        "target_geom_name" : "MTEDIT125",
        "hours" : "1,2,3,4,5",
        "file_write" : "multiple",
        "origintime" : "latest",

        "processqueue" : [
                {
                        "leveltype" : "height",
                        "levels" : "0",
                        "plugins" : [ { "name" : "weather_code_1" } ]
                }
        ]
    }

