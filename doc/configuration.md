Configuring himan is done through json-files, and all the common rules of writing json apply:

* (Almost) all keys and values must be enclosed in quotes ("")
* Key and value are separated with colons
* Elements are separated by commas

The json file can be divided into three main parts:
* global scope
* processqueue scope
* plugin scope

Almost all configuration options can be specified in any of the scopes. The more fine-grained scopes override the values
defined in more general scopes. For example some option specified at plugin scope will override the same option defined
in global scope.

##### Table of Contents  
* [Target area](#Target_area)  
  * [Method 1: Using pre-defined areas from database ](#Target_area_1)
  * [Method 2: Defining area with `bbox` ](#Target_area_2)
  * [Method 3: Defining area in detail ](#Target_area_3)
  * [Method 4: Using proj4 string ](#Target_area_4)
  * [Method 5: List of points ](#Target_area_5)
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
  * [Storage type](#Storage_type)
  * [Allowed missing values](#Allowed_missing_values)
  * [ss_state table name](#ss_state_table_name)
  * [grib decimal precision](#grib_decimal_precision)
  * [Write between plugin calls](#write_between_plugin_calls)
  * [Validate metadata](#Validate_metadata)
* [Environment variables](#Environment_variables)
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

In stereographic projection, the orientation of the area is given with key `orientation`

    "orientation" = "<degrees>"

In rotated latlon projection the coordinates of south pole are given with keys `south_pole_longitude` and  `south_pole_latitude`

    "south_pole_longitude" : "<degrees>",
    "south_pole_latitude" : "<degrees>",

Example:

    "projection" : "stereographic",
    "bottom_left_latitude" : "51.3",
    "bottom_left_longitude" : "6.0",
    "orientation" : "20"

    "projection" : "rotated_latlon",
    "bottom_left_latitude" : "-24",
    "bottom_left_longitude" : "-33.5",
    "top_right_latitude" : "31.42",
    "top_right_longitude" : "36.47",
    "south_pole_longitude" : "0",
    "south_pole_latitude" : "-30",

<a name="Target_area_4"/>

## Method 4: Using proj4 string for area details

An area can be defined by giving the corresponding proj4 string.

    "proj4": "+proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +x_0=1058511.28262988 +y_0=1298134.00243759 +R=6367470 +units=m +no_defs"


Grid information needs to be defined separately when using `proj4`. Basically the following keys also need to be defined:

* scanning_mode
* first_point_longitude
* first_point_latitude
* di
* dj
* ni
* nj

<a name="Target_area_5"/>

## Method 5: List of points

List of points isn't exactly an area, but Himan treats it as one. It can be defined in two ways: 
* with key `points`, point coordinates are given in the configuration file, or 
* with key `stations`, point coordinates are fetched from database.

`points` key can be specified in two different ways, either by just the coordinates or by defining id, name and coordinates for a point. Different points are separated by a comma, different parameters for a point are separated with a whitespace. Order of in-point elements is: id name longitude latitude.

    "points" : "25 60,27 61",

    "points" : "1 Helsinki 25 60, 2 Kotka 27 61",

With key `stations` Himan fetches point (station) information from a database using on the given station-id.

    "stations" : "100971,101030",

<a name="Target_grid"/>

# Target grid

Target grid size is defined with keys `ni`and `nj`.

**Note! If target area was defined using database information, grid does not need to be defined separately.**

    "ni" : "<number of grid points in x-axis direction>",
    "nj" : "<number of grid points in y-axis direction>"

Grid scanning mode (direction of reading) is given with key `scanning_mode`. Possible values are
* `+x+y`, reading starts at bottom left corner
* `+x-y`, reading starts at top left corner

`"scanning_mode" : "+x+y" | "+x-y"`

Example:

    "ni" : "220",
    "nj" : "220",
    "scanning_mode" : "+x+y"

<a name="Source_area_and_grid"/>

# Source area and grid

If source data is read from database (as is the case in FMI operations), key `source_geom_name` can be used to define one or more source areas. By default (if key is not set) Himan will read any geometry it finds from the database that matches the producer. Key `source_geom_name` can contain multiple values separated with a comma, and the order of the geometries is preserved. 

**Note! that if multiple geometries exist for a producer, and the first read does not cover the target area completely, Himan will not try amend the data using other geometries. Therefore the order of the geometries does matter!**

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

**Note! The more producers are defined, the longer the fetching process will last, especially if the producer that the data is found from is the last in the list.**

Example:

    "source_producer" : "1",
    "target_producer" : "230"
    

    "source_producer" : "1,230",
    "target_producer" : "230"

Keys can be set in the global or processqueue scope.

<a name="Time"/>

# Time

Time information contains forecast analysis time and the forecast lead times (steps). Lead time is always an offset from the analysis time.

Lead times can be listed in two ways:
* listing all hours that should be calculated, or
* listing start time, stop time and a step

Time must be a top level element, but it can also be specified in processqueue level.

<a name="Time_1"/>

## Analysis time

Analysis time is given with key `origintime`, using standard timestamp format `YYYY-MM-DD HH24:MI:SS` (or `%Y-%m-%d %H:%M:%S`). Key value can also be `latest`, then Himan will try to fetch the latest forecast found from the database. If penultimate forecast is required, value `latest-1` can be used.

    "origintime : "latest"|"latest-X"|"<timestamp>"

Multiple origin times can also be specified with key `origintimes`.

    "origintimes" : "<timestamp>,<timestamp>,..."

<a name="Time_2"/>

## Lead time method 1: Listing times

With key `hours` the hours (lead times) that should be calculated are listed. Values are separated with a comma, if a hyphen ('-') is used, Himan will interpolate values to fill the gap.

    hours : <list of hours>

Example:

    "hours" : "1,2,3-8"

This example will result to hours 1,2,3,4,5,6,7 and 8.

By default when expanding a list the step value is 1, but it can be changed:

    "hours" : "1,2-8-2"

Result: 1,2,4,6,8

For sub-hour values key `times` can be used:

    "times" : "0:00:00-9:00:00-0:15:00"

Result: 0:00:00,0:15:00,0:30:00,...,9:00:00


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

start_time and stop_time can be used to specify time duration value

    "start_time" : "0:00:00",
    "stop_time" : "1:00:00",
    "step" : "0:15:00"


<a name="Levels"/>

# Levels

Target level for calculation is defined with key `leveltype`.

    "leveltype" : "height" | "pressure" | "hybrid" | "ground"

The level value is given with key `levels`, values are separated with a comma, and if a hyphen is given Himan will try to fill the gap.

    "levels" : "<list of levels>",

Most (but not all) plugins respect the level definition and will use it when processing data. Notable exceptions are at least cape-plugin.

Example:

    "leveltype" : "hybrid"
    "levels" : "4,5,6,7-10"

For a level that has two values, they should be separated with an underscore

    "leveltype" : "general"
    "levels" : "74_75, 73_74"


<a name="Plugins"/>

# Plugins

Started plugins are listed as json dictionary in a json table. The order of plugins is preserved.

Plugin name can be given by using the full path (`himan::plugin::pluginin_name` or just the shorter name `plugin_name`. The only mandatory key for the dictionary is `name`; all other possible keys are read and passed on the called plugin. It is the responsibility of the plugin to handle the extra parameters.

Plugins are defined with key `plugins`

    "plugins" : [ { "name" : "plugin" ] ]

Example:

    "plugins" : [ { "name" : "tpot" } ]

    "plugins" : [ 
      { "name" : "hybrid_pressure" },
      { "name" : "windvector", "for_air" : true } 
    ]

Some plugins support additional configuration options, like windvector in the example above takes `"for_air": true`. The plugin documentation should be consulted for the available options.

An environment variable value can be passed to a plugin configuration like so (using the same example):

    { "name": "windvector", "for_air": "{env:ENVIRONMENT_VARIABLE_NAME}" }


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

The write mode is defined with key `write_mode`. Possible values are:

* `single`, each grid is written to its own file
* `few`, each plugin results are written to its own file
* `all`, all grids are written to one file
* `no`, file is only written to cache

    "write_mode" : "single | few | all | no"

Default value for key is `single`. Key can be set in the global or processqueue scope.

Himan can pack the written files with either gzip or bzip2 methods. The key that controls this is called `file_compression`. 

    "file_compression" : "none | gzip | bzip2"

Default value for key is `none`.

Filename can be controlled with key `filename_template`. If no value is given, Himan will use default template.

Allowed template values are:
* {analysis_time:DATE_FORMAT_SPECIFIER}            - analysis time
* {forecast_time:DATE_FORMAT_SPECIFIER}            - forecast time
* {step:DURATION_FORMAT_SPECIFIER}                 - leadtime of forecast
* {geom_name}                                      - geometry name
* {grid_name}                                      - grid (projection) short name
* {grid_ni:NUMBER_FORMAT_SPECIFIER}                - grid size in x direction, only for regular grids
* {grid_nj:NUMBER_FORMAT_SPECIFIER}                - grid size in y direction, only for regular grids
* {param_name}                                     - parameter name
* {aggregation_name}                               - aggregation name
* {aggregation_duration:DURATION_FORMAT_SPECIFIER} - aggregation duration
* {processing_type_name}                           - processing type name
* {processing_type_value:NUMBER_FORMAT_SPECIFIER}  - processing type value
* {processing_type_value:NUMBER_FORMAT_SPECIFIER}  - second possible processing type value
* {level_name}                                     - level name
* {level_value:NUMBER_FORMAT_SPECIFIER}            - level value
* {level_value2:NUMBER_FORMAT_SPECIFIER}           - second possible level value
* {forecast_type_name}                             - forecast type short name, like 'sp' or 'det'
* {forecast_type_id:NUMBER_FORMAT_SPECIFIER}       - forecast type id, 1 .. 5
* {forecast_type_value:NUMBER_FORMAT_SPECIFIER}    - possible forecast type value
* {producer_id:NUMBER_FORMAT_SPECIFIER}            - radon producer id
* {file_type}                                      - file type extension, like grib, grib2, fqd, ...
* {wall_time:DATE_FORMAT_SPECIFIER}                - wall clock time
* {env:ENVIRONMENT_VARIABLE_NAME}                  - value from an environment variable

Format specifiers:
* DATE_FORMAT_SPECIFIER: usual c-style strftime formatting (%Y%m%d...)
* DURATION_FORMAT_SPECIFIER: custom time duration formatting
  * %H - Total hours % 24
  * %M - Total minutes % 60
  * %S - Total seconds % 60
  * %d - Total days
  * %h - Total hours
  * %m - Total minutes
  * %s - Total seconds
  * Each value can have further printf-style identifiers, like  %03h --> total hours with up to 3 leading zeros
* NUMBER_FORMAT_SPECIFIER: usual c-style printf formatting

Example:

    "filename_template" : "fc{analysis_time:%Y%m%d%H%M}_{step:%03h}_{level_name}.{file_type}"

Default formatters for different keys are:

```
    case Component::kAnalysisTime:
    case Component::kForecastTime:
        return "%Y%m%d%H%M";
    case Component::kStep:
    case Component::kAggregationDuration:
        return "%Hh%Mm";
    case Component::kProcessingTypeValue:
    case Component::kProcessingTypeValue2:
    case Component::kLevelValue:
    case Component::kLevelValue2:
    case Component::kForecastTypeValue:
        return "%.0f";
    case Component::kForecastTypeId:
    case Component::kProducerId:
    case Component::kGridNi:
    case Component::kGridNj:
        return "%d";
    case Component::kWallTime:
        return "%Y%m%d%H%M%S";
```


Himan can pack data with a few different methods. Packing is controlled with key `packing_type`.

    "file_packing_type" : "simple_packing" | "jpeg_packing" | "ccsds_packing"

Note! Only GRIB2 files support all three packing types. Default value is `simple_packing`.

<a name="Database_access"/>

## Database access

Himan can be forced to read data from command line given arguments only. This behavior can be set with key `read_from_database`. Default value for key is `true`.

    "read_from_database" : true | false

Example:

    "read_from_database" : false,

Himan can write file metadata to radon database. This is controlled with `write_to_database`. Default value for key is `false`.

    "write_to_database" : true | false

Example:

    "write_to_database" : false,

<a name="Forecast_types"/>

## Forecast types

With key `forecast_type` the source forecast type is set. Using this, Himan can be used to calculate parameters for all ensemble members for example. Default value for key is `deterministic`, meaning that Himan will read data from deterministic forecasts only.

    "forecast_type" : "deterministic" | "cf" | "pf1-2"

With value `cf` the control forecast of an ensemble is chosen, and with value `pfVALUE-VALUE` a range of ensemble members (permutations) are chosen. For example the value `pf1-10` means that permutation 1,2,3,4,5,6,7,8,9,10 are read. Values can be combined with a comma. Control member has value 0 by default, but it can be changed by postfixing an integer value to "cf", ie "cf1" would define control forecast with value 1.

Key can be set in the global or processqueue scope.

Example:

    "forecast_type" : "deterministic",

    "forecast_type" : "cf,pf1-50",

<a name="Memory_usage"/>

## Memory usage

Himan uses an in-memory cache for all read and written data.

The cache for writes can be turned off with key `use_cache_for_writes`. Default value is `true`.
If set to false, no calculated data is inserted to cache.

    "use_cache_for_writes" : "true | false",

Key can be set in the global or processqueue scope.

Example:

    "use_cache_for_writes" : false,

The cache for reads can be turned off with key `use_cache_for_reads`. Default value is `true`.
If set to false, no data read is inserted to cache.

    "use_cache_for_reads" : "true | false",

Key can be set in the global or processqueue scope.

Example:

    "use_cache_for_reads" : false,

Note: this key was previously called `use_cache`, and that is still supported for backwards compatibility.

Memory cache size can be controlled with key `cache_limit`. Minimum cache limit if 100Mi ie 104857600. Mi and Gi modifiers are supported. The value of the key specifies the maximum number of data bytes Himan will hold in memory. Note that this is not the overall total memory consumption.

When the limit is reached, data is evicted using an LRU algorithm.

    "cache_limit" : "<integer value larger or equal to than 104857600 OR integer{Mi|Gi}>",

Default value for key is 0, i.e. no upper limit for cache size.

Example:

    "cache_limit" : "104857600"
    "cache_limit" : "6Gi"

By default himan will allocate all necessary memory when it starts. In low-memory environments this might be problematic. With key `dynamic_memory_allocation`, Himan can be forced to allocate memory dynamically (reserving it just before needed, and releasing immediately afterwards).

    "dynamic_memory_allocation" : true | false,

Default value is `false`.

<a name="Asynchronous_execution"/>

## Asynchronous execution

By default Himan plugins are executed in a serialized fashion, mostly because many of the plugins are dependent on the output of others. Some plugins, however, lie necessarily at the end of the execution line, and when those plugins are in the critical chain of execution they will (unnecessarily) slow down the the total execution of Himan. 

This can be avoided by using the configuration file key 'async'. By default the value is false, confirming to the serialized execution. It can be set to true for any individual plugin or processqueue element scope.

Example:

    {
	"async" : true,
        "leveltype" : "...",
        "levels" : "...",
        "plugins" : [ { "name" : "...", "async" : false } ]
    }

**Note! Asynchronous execution should only be enabled for those plugins that have no other plugins as dependants!**

<a name="Storage_type"/>

## Storage type

Himan can write output files into two different storage types: local POSIX file system (default), and S3 object storage.

If one wants to write to S3, that needs to be configured with configuration file key 'write_storage_type'.

NB! This only affects writing; Himan can _read_ from both local file system and S3 storage simultaneously without any
explicit configuration.

Example:

    "write_storage_type" : "local | s3",

<a name="Allowed_missing_values"/>

## Allowed missing values

By default Himan does not care if some plugin produces missing values. This behavior can be controlled with configuration file key 'allowed_missing_values'.

The key can contain the maximum number of missing values in a single grid in either absolute numbers or as a percentage of the grid size.
If the value is reached, Himan will stop running immediately.

The default behavior is to never abort. Key can be set in either main level or processqueue scope.

Example:

    "allowed_missing_values" : 20000
    "allowed_missing_values" : "10%"

<a name="ss_state_table_name"/>

## ss_state table name

In radon database a table 'ss_state' is updated after each Himan execution to provide information about the newly created data for
smartmet server. With configuration file option 'ss_state_table_name' the table name that is reported to ss_state can be changed.
This is required in some circumstances, for example when a special database view is used to alter the metadata before it is presented
to smartmet server.

Note: This option does not provide a way to change the name of the ss_state table itself!

Example:

    "ss_state_table_name" : "new_name"


<a name="grib_decimal_precision"/>

## grib decimal precision

When writing output files in grib format, the precision is controlled in two ways: in radon database table param_precision contains
the decimal precision (how many numbers after decimal point) for many parameters. This can be overridden with configuration file option
'write_options.precision'. Option can only be specified in plugin scope. If neither is present, default setting is used.

Example:


    {
        "...",
        "plugins" : [ { "name" : "...", "write_options.precision" : 2 } ]
    }

<a name="extra_file_metadata"/>

## extra file metadata

Pass any custom metadata to resulting files, overriding any existing or database information. Implementation and support depends on the filetype. Currently only grib is supported. Arguments are given in key-value pairs, comma separated. Data type of value can be string (:s), integer (:i) or float (:d). Default is integer.


    "plugins" : [ { "name" : "...", "write_options.extra_metadata" : "shapeOfTheEarth=6,latitudeOfFirstGridPointInDegrees:d=10.01" } ]


This functionality was previously a part of transformer plugin, where the property was called "extra_file_metadata". This key is now an alias for write_options.extra_metadata.


<a name="write_between_plugin_calls"/>

## Write between plugin calls

When writing to s3, all data is hold in cache and is written only when all plugins have been executed, as appending to s3 is not possible.
With configuration option 'write_to_object_storage_between_plugin_calls' writes can be done between plugin calls, for example to reduce
the amount of data held in memory. By default this options has value 'false', and it is only applied when writing to s3. Option can
only be set at top-level configuration.

It is the users' responsibility to set 'filename_template' such that successive write calls do not override previous data.

Example:


    {
        "write_to_object_storage_between_plugin_calls" : true,
    }


<a name="Validate_metadata"/>

## Validate metadata

By default Himan will validate all metadata it reads from a file, against information received from radon. If the validation is not successful,
Himan will discard the data.

By setting 'validate_metadata' key to 'false', Himan will trust radon database and will not do validation. Default value for this key is 'true'.

Example:

    {
        "validate_metadata" : false,
    }


<a name="Environment_variables"/>

# Environment variables

Himan behavior is controlled with a group of environment variables.

* HIMAN_LIBRARY_PATH

Controls where Himan is trying to find plugins. Default location is /usr/lib64/himan

* MASALA_PROCESSED_DATA_BASE

Controls where the resulting files are written if they are written to database. This variable gives the "base" directory
and Himan will add subdirectories

* RADON_HOSTNAME

Specify hostname of radon database

* RADON_DATABASENAME

Specify name of radon database

* RADON_PORT

Specify port of radon database

* RADON_WETODB_PASSWORD

Specify the password for database user wetodb which Himan is using

* S3_ACCESS_KEY_ID

When accessing S3 storage, specify access key id

* S3_SECRET_ACCESS_KEY

When accessing S3 storage, specify access key

* S3_SESSION_TOKEN

When accessing S3 storage, specify (optional) session token

* S3_HOSTNAME

When writing to S3 storage, specify hostname (for example s3.eu-west-1.amazonaws.com)

* FMIDB_DEBUG

Not really a Himan environment variable, but very useful still. Setting any value will print all sql queries to stdout.

The following environment variables control options that can also be specified as command line options.
If both and environment variable and a command line option is specified, the latter will take precedence.

* HIMAN_OUTPUT_FILE_TYPE

Possible values: grib, grib2, querydata, csv, geotiff

* HIMAN_COMPRESSION

Possible values: gz, bzip2

* HIMAN_CONFIGURATION_FILE

Possible values: filename or '-' for stdin

* HIMAN_THREADS

Number of threads to use

* HIMAN_DEBUG_LEVEL

Possible values: 0-5

* HIMAN_STATISTICS

Enable statistics by giving any string as label

* HIMAN_CUDA_DEVICE_ID

Explicitly set device id

* HIMAN_NO_CUDA

Disable all cuda usage

* HIMAN_NO_CUDA_UNPACKING

Disable cuda unpacking of grib data

* HIMAN_NO_CUDA_PACKING

Disable cuda packing of grib data

* HIMAN_NO_DATABASE

Disable use of radon database

* HIMAN_PARAM_FILE

Specify location of parameter metadata file (to be used with no-database mode)

* HIMAN_NO_AUXILIARY_FILE_FULL_CACHE_READ

Disable reading of all auxiliary files at first call

* HIMAN_NO_SS_STATE_UPDATE

Disable updating of ss_state table in radon

* HIMAN_NO_STATISTICS_UPLOAD

Disable statistics upload to radon

* HIMAN_AUXILIARY_FILES

Specify list of auxiliary files to read (whitespace-separated)

* HIMAN_NUM_THREADS

Specify at most how many threads should be used for data processing

* HIMAN_TEMP_DIRECTORY

Specify where to write temporary files if needed. Default: /tmp

<a name="Full_examples"/>

# Full examples

The examples are using FMI defined geometry names and producers ids.

Calculating weather-code parameter from ECMWF forecast, using two source geometries and two source producers.

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

