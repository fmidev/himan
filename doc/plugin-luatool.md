# Summary

luatool-plugin enables himan core functionality to be used through lua scripts. Lua is a dynamically typed scripting language, and is very useful when only very simple and computationally modest processing is needed. Luatool internally uses luabind-library to expose certain parts of the core functionality.

Basic functionality:

* luatool-plugin works like any other post processing plugin, although it's classified as an infrastructure plugin
* himan will launch as many threads as are needed and each of the threads will execute the script(s) simultaneously
* lua scripts control the writing of output files
* lua scripts can access hitool-plugin functionality
* interpolation, cache, database access etc. are all done automatically like with any other plugin
* luatool plugin does not support cuda calculation, but it can use cuda to unpack grib data (through the regular himan functionlity)
* luatool-plugin can execute one or more scripts
* in lua language, array indexing starts with one. luatool-plugin automatically converts indexes between lua and C++ when converting data from std::vector<double> to lua native table and vice versa

# Enumerators

In lua enumerators are accessed using a class prefix, like 

    local x = HPLevelType.kGround

```
HPLevelType
    kUnknownLevel = 0
    kGround = 1
    kMaximumWind = 6
    kTopOfAtmosphere = 8
    kIsoThermal = 20
    kLake = 21
    kPressure = 100
    kPressureDelta = 101
    kMeanSea = 102
    kAltitude = 103
    kHeight = 105
    kHeightLayer = 106
    kHybrid = 109
    kGroundDepth = 112
    kDepth = 160
    kEntireAtmosphere = 200
    kEntireOcean = 201
    kMaximumThetaE = 246

HPParameterUnit
    kHPa = 5 // hectopascal
    kM = 8 // meters

HPTimeResolution
    kUnknownTimeResolution = 0
    kHourResolution = 1
    kMinuteResolution = 2
    kYearResolution = 3
    kMonthResulution = 4
    kDayResolution = 5

HPFileType
    kUnknownFile = 0
    kGRIB1 = 1
    kGRIB2 = 2
    kGRIB = 3
    kQueryData = 4
    kNetCDF = 5

HPProjectionType
    kUnknownProjection = 0
    kLatitudeLongitude
    kStereographic
    kAzimuthalEquidistant 
    kRotatedLatitudeLongitude
    kReducedGaussian
    kPointList
    kLambertConformalConic

HPScanningMode
    kUnknownScanningMode = 0
    kTopLeft
    kTopRight
    kBottomLeft
    kBottomRight

HPAggregationType
    kUnknownAggregationType = 0
    kAverage
    kAccumulation
    kMaximum
    kMinimum
    kDifference

enum HPProcessingType
    kUnknownProcessingType = 0
    kProbabilityGreaterThan
    kProbabilityLessThan
    kProbabilityBetween
    kProbabilityEquals
    kProbabilityNotEquals
    kProbabilityEqualsIn
    kFractile
    kEnsembleMean
    kSpread
    kStandardDeviation
    kEFI

HPForecastType
    kUnknownType = 0
    kDeterministic
    kAnalysis
    kEpsControl
    kEpsPerturbation

HPGridType
    kUnknownGridType = 0
    kRegularGrid
    kIrregularGrid
```

# Classes

## aggregation

aggregation is a parameter component, defining for example that it is an accumulation.

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| HPAggregationType | GetType | | Returns aggregation type (for example: cumulative) |
|   | SetType | HPAggregationType  | Set aggregation type |
| time_duration | GetTimeDuration | | Returns the time duration of the aggregation (for example: one hour) |
|   | SetTimeDuration | time_duration | Set time duration |
| time_duration | GetTimeOffset | | Returns the time offset (beginning of aggregation period), usally a negation of time duration |
|   | SetTimeOffset | time_duration | Set time offset |

## configuration

configuration class instance is automatically assigned to a lua script. It represents the configuration created from command line options
and without any plugin specific options. See also plugin_configuration.

| Return value  | Name | Arguments | Description |
|---|---|---|---|
| string | ClassName | | Returns class name |
| string | GetOutputFileType | | Returns the type of output file (grib, querydate, ...) |
| producer | GetSourceProducer | number | Returns the configured source producer, indexing starts from 0 |
| producer | GetTargetProducer | | Returns target producer |
| time_duration | GetForecastStep | | Returns the forecast step that's configured in the configuration file (if applicable) |
| bool | GetUseCuda | | Returns true if usage of gpu functions is not disabled |

## forecast_time

forecast_time contains both analysis time and valid time.

    local analysis = raw_time("2015-01-02 00:00:00")
    local forecast = raw_time("2015-01-02 03:00:00")
 
    local t = forecast_time(analysis, forecast)

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| raw_time | GetOriginDateTime | | Returns origin date time (forecast analysis time) as full timestamp |
|   | SetOriginDateTime | raw_time  | Set origin date time |
| raw_time | GetValidDateTime | | Returns valid date time (forecast "lead time") as full timestamp |
|   | SetValidDateTime | raw_time  | Set valid date time |
| number | GetStep | | Returns time step (valid time - origin time) as time_duration |

## info

info-class combines all different pieces of metadata into one. This class instance is not usually created but it's returned for example when reading data. In luatool info class is not as important as in C++ himan, since in common cases only data is returned (ie. lua table), not the info itself.

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| | First | | Sets all iterators to initial value |
| | ResetParam | | Reset parameter iterator to point value before first value |
| bool | FirstParam | | Set parameter iterator to initial value |
| bool | NextParam | | Set parameter iterator to next value |
| | ResetLevel | | Reset level iterator to point value before first value |
| bool | FirstLevel | | Set level iterator to first value |
| bool | NextLevel | | Set level iterator to next value |
| | ResetTime | | Reset time iterator to point value before first value |
| bool | FirstTime | | Set time iterator to first value |
| bool | NextTime | | Set time iterator to next value |
| number | SizeParams | | Returns number of parameters |
| number | SizeLevels | | Returns number of level |
| number | SizeTimes | | Returns number of times |
| number | SizeLocations | | Returns number of locations (points) in the current grid |
| param | GetParam | | Returns current parameter |
| level | GetLevel | | Returns current level |
| forecast_time | GetTime | | Returns current time |
| | SetParam | param | Sets (replaces) current parameter |
| | SetLevel | level | Sets (replaces) current level |
| | SetTime | forecast_time | Sets (replaces) current time |
| point | GetLatLon | number | Returns latlon coordinates of given grid point |
| table | GetValues | | Returns grid data contents |
| | SetValues | table | Sets grid data contents from a lua table |
| | SetValuesFromMatrix | matrix | Sets grid data contents from a Himan matrix |
| number | GetMissingValue | | Returns missing value |
| | SetMissingValue | number | Sets missing value |

## level

level class instance consists of level type (HPLevelType), and level value (one or two).

    local l = level(HPLevelType.kHeight, 2)

Typically most levels only have a single value.

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| HPLevelType | GetType | | Return level type (enum HPLevelType) |
| | SetTYpe | HPLevelType | Set level type |
| number | GetValue | | Returns first level value |
| | SetValue | number | Set first level value |
| number | GetValue2 | | Returns second level value |
| | SetValue2 | number | Set second level value |

## logger

A logger instance is automatically assigned for a lua script as a global variable. 

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| | Trace | string | Log something in trace level |
| | Debug | string | Log something in debug level |
| | Info | string | Log something in info level |
| | Warning | string | Log something in warning level |
| | Error | string | Log something in error level |
| | Fatal | string | Log something in fatal level |

## matrix

Matrix is a wrapper class for the actual container holding the data (ie. std::vector).
It provides basic 2D accessing capabilities.

| Return value  | Name | Arguments | Description |
|---|---|---|---|
| | SetValues | table | Set matrix values |
| table | GetValues | | Return matrix values |
| | Fill | | Fill matrix with given value |

## modifier

modifier class is used to process data and height. Different modifiers exist for calculating min, max, mean etc for grids with given heights. modifier is usually used through
hitool-plugin, which will feed data to it.

Available modifiers are:

| Name | Description |
|---|---|
| max | Calculate maximum value for each gridpoint |
| min | Calculate minimum value |
| maxmin | Calculate maximum and minimum value |
| count | Calculate number of parameter value occurrences |
| mean | Calculate mean value |
| findvalue | Find value for a parameter from a given height
| findheight | Find the height of given parameter value |
| findheight_lt | Find the first height where parameter value is less than given value |
| findheight_gt | Find the first height where parameter value is more than given value |

Example:

```
mean = modifier_mean()

mean:SetLowerHeightGrid(lower)
mean:SetUpperHeightGrid(upper)

mean:Process(data, heights)
```

Note! In most cases hitool-plugin should be used to automate fetching of data!

## param

A basic identifer for a param is its name. The name is always in the form of NAME-UNIT, for example T-K (temperature in Kelvins). 

    local p = param("T-K")

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| string | GetName | | Return parameter name |
| | SetName | string | Set parameter name |
| number | GetGrib2Number | | Returns grib2 parameter number (if applicable) |
| | SetGrib2Category | number | Set grib2 parameter number |
| number | GetGrib2Category | | Returns grib2 parameter category (if applicable) |
| | SetGrib2Category | number | Set grib2 parameter category |
| number | GetGrib2Discipline | | Returns grib2 parameter discipline (if applicable) |
| | SetGrib2Discipline | number | Set grib2 parameter discipline |
| number | GetGrib1Parameter | | Returns grib1 parameter number (if applicable) |
| | SetGrib1Number | number | Set grib1 parameter |
| number | GetGrib1TableVersion | | Returns grib1 table version number (if applicable) |
| | SetGrib1TableVersion | number | Set grib1 table version number |
| number | GetUnivId | | Returns universal id number (fmi internal numbering, if applicable) |
| | SetUnivId | number | Set universal id number |
| aggregation | GetAggregation | | Return aggregation of parameter (if applicable) |
| | SetAggregation | aggregation | Set aggregation for a parameter

## point

point represent an xy or latlon point.

    local p = point(25, 60) -- x=25, y=60

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| number | GetX | Returns x or longitude value |
| | SetX | number | Set x value |
| number | GetY | Returns y or latitude value |
| | SetY | number | Set y value |

## plugin_configuration

plugin_configuration class instance is automatically assigned to a lua script. It represents the configuration that launched the calculation.
This class is inheriting from configuration class.

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| string | GetValue | string | Return a per-plugin configuration value, if given in the plugin configuration file |
| bool | Exists | string | Checks if some per-plugin configuration value is set in the plugin configuration file |

## processing_type

processing_type is another parameter component, defining for example that it is a derived probability.

| Return value  | Name | Arguments | Description |
|---|---|---|---|
| string | ClassName | | Returns class name |
| HPProcessingType | GetType | | Returns processing type type (for example: probability greater than) |
|   | SetType | HPProcessingType  | Set processing type type |
| number | GetValue | | Returns first value related to processing type |
|   | SetValue | number  | Set first value related to processing type |
| number | GetValue2 | | Returns second value related to processing type |
|   | SetValue2 | number  | Set second value related to processing type |

## producer

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| string | GetName | | Return producer name |
| | SetName | string | Sets producer name |
| number | GetId | | Return producer database id |
| | SetId | number | Sets producer database id |
| number | GetProcess | | Returns producer grib id |
| | SetProcess | | Sets producer grib id |
| number | GetCentre | | Returns producer grib centre |
| | SetCentre | number | Sets producer grib centre |

## raw_time

raw_time represents a timestamp and is a thin wrapper over boost::posix_time::ptime.

    local r = raw_time("2015-01-02 12:00:00")

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| string | String | string | Returns the time in user-given format (for format details see boost documentation) |
| | Adjust | HPTimeResolution, number | Adjust time as needed |
| bool | Empty | | Checks if time is valid |

## time_duration

time_duration represents a time interval or duration, and it's a thin wrapper over boost::posix_time::time_duration.

    local td = time_duration("01:00") -- one hour
    local td = time_duration(kHourResolution, 1)

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| number | Hours | | Return number of hours in time interval (not normalized) |
| number | Minutes | | Return number of minutes in time interval (not normalized) |
| bool | Empty | | Checks if time is valid |

## ensemble

ensemble is a representation of an ensemble forecast as a unit. This enables processing that is done on the whole
forecast.

	local e = ensemble(param("T-K"), ensemble_size)

| Return value | Name | Arguments | Description |
|---|---|---|---|
| string | ClassName | | Returns class name |
| table | Fetch | plugin_configuration, forecast_time, forecast_level | Returns a fetched array of infos |
| number | Value | integer | Returns the value of the ensemble member at the current location |
| table | Values | | Returns an array of all the ensemble member values at the current location |
| table | SortedValues | | Returns an array of all the ensemble member values at the current location, sorted in increasing order |
| bool | ResetLocation | | Reset the location iterator to undefined state |
| bool | FirstLocation | | Set the location iterator to the first position |
| bool | NextLocation | | Advance the location iterator |
| number | Mean | | Returns the mean of the ensemble member values at the current location |
| number | Variance | | Returns the variance of the ensemble member values at the current location |
| number | CentralMoment | integer | Returns the Nth central moment of the ensemble member values at the current location |
| number | Size | | Returns the size of the currently fetched ensemble |
| number | ExpectedSize | | Returns the expected size of the ensemble, which can be different from the current size of the ensemble |
| | SetMaximumMissingForecasts | integer | Set the number of allowed missing forecasts in the ensemble |
| number | GetMaximumMissingForecasts | | Get the number of allowed missing forecasts in the ensemble |
| info | GetForecast | number | Get the data of an ensemble member with given order number |

## lagged_ensemble

lagged_ensemble is an ensemble that is composed of ensembles from different analysis times.

	local e = lagged_ensemble(param("T-K"), ensemble_size, HPTimeResolution.kHourResolution, lag, steps)

lagged_ensemble is derived from ensemble.

| Return value | Name | Arguments | Description |
|---|---|---|---|
| string | ClassName | | Returns class name |
| table | Fetch | plugin_configuration, forecast_time, forecast_level | Returns a fetched array of infos |
| number | Value | integer | Returns the value of the ensemble member at the current location |
| table | Values | | Returns an array of all the ensemble member values at the current location |
| table | SortedValues | | Returns an array of all the ensemble member values at the current location, sorted in increasing order |
| bool | ResetLocation | | Reset the location iterator to undefined state |
| bool | FirstLocation | | Set the location iterator to the first position |
| bool | NextLocation | | Advance the location iterator |
| time_duration | Lag | | Returns the lag of the ensemble |
| number | Mean | | Returns the mean of the ensemble member values at the current location |
| number | Variance | | Returns the variance of the ensemble member values at the current location |
| number | CentralMoment | integer | Returns the Nth central moment of the ensemble member values at the current location |
| number | Size | | Returns the size of the currently fetched ensemble |
| number | ExpectedSize | | Returns the expected size of the ensemble, which can be different from the current size of the ensemble |
| number | NumberOfSteps | | Returns the number of lagged steps |
| | SetMaximumMissingForecasts | integer | Set the number of allowed missing forecasts in the ensemble |
| number | GetMaximumMissingForecasts | | Get the number of allowed missing forecasts in the ensemble |

# Variables

Several global variables have been introduced automatically to lua scripts.

## hitool

hitool instance can be used to examine the properties of the atmosphere. See hitool documentation for more details.

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| table | VerticalMaximum | param, number, number | Returns maximum value found within the height limits given (minimum height in meters, maximum height in meters), single height values for all grid points | 
| table | VerticalMaximumGrid | param, table, table | Returns maximum value found within the height limits given (minimum height in meters, maximum in meters), individual height values for all grid points |
| table | VerticalMinimum | param, number, number | Returns minimum value found within the height limits given (minimum height in meters, maximum height in meters), single height values for all grid points | 
| table | VerticalMinimumGrid | param, table, table | Returns minimum value found within the height limits given (minimum height in meters, maximum in meters), individual height values for all grid points |
| table | VerticalSum | param, number, number | Returns the sum of all values found within the height limits given (minimum height in meters, maximum height in meters), single height values for all grid points | 
| table | VerticalSumGrid | param, table, table | Returns the sum of all values found within the height limits given (minimum height in meters, maximum in meters), individual height values for all grid points |
| table | VerticalAverage | param, number, number | Returns the average value found within the height limits given (minimum height in meters, maximum height in meters), single height values for all grid points | 
| table | VerticalAverageGrid | param, table, table | Returns the average value found within the height limits given (minimum height in meters, maximum in meters), individual height values for all grid points |
| table | VerticalCount | param, number, number, number | Returns the number of values found within the height limits given (minimum height in meters, maximum height in meters), single height values for all grid points, single value to search for all grid points | 
| table | VerticalCountGrid | param, table, table, table | Returns the number of values found within the height limits given (minimum height in meters, maximum in meters), individual height values for all grid points, individual value to search for all grid points |
| table | VerticalHeight | param, number, number, number | Returns the height (in meters) of a given parameter value within the height limits given (minimum height in meters, maximum height in meters), single height values for all grid points | 
| table | VerticalHeightGrid | param, table, table, table, number | Returns the height (in meters) value found within the height limits given (minimum height in meters, maximum in meters), individual value to search for all grid points, last parameter defines which height to return if multiple are encountered |
| table | VerticalHeightGreaterThan | param, number, number, number | Same as VerticalHeight, but also considers the case when the value is encountered when entering wanted level zone | 
| table | VerticalHeightGreaterThanGrid | param, table, table, table, number | Same as VerticalHeightGrid, but also considers the case when the value is encountered when entering wanted level zone |
| table | VerticalHeightLessThan | param, number, number, number | Same as VerticalHeight, but also considers the case when the value is encountered when entering wanted level zone | 
| table | VerticalHeightLessThanGrid | param, table, table, table, number | Same as VerticalHeightGrid, but also considers the case when the value is encountered when entering wanted level zone |
| table | VerticalValue | param, number | Returns the value of a parameter from a given height, single value for all grid points | 
| table | VerticalValueGrid | param, table | Returns the value of a parameter from a given height, individual value for all grid points |


## luatool

luatool variable represents the plugin itself. This variable is used to fetch and write data.

    local grid = luatool:Fetch(current_time, current_level, par1)

    luatool:WriteToFile(result)

Note! The argument for WriteToFile MUST be 'result'!

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| table | Fetch | forecast_time, level, param | Fetch data with given search arguments |
| table | FetchWithType | forecast_time, level, param, forecast_type | Fetch data with given search arguments including forecast_type |
| info | FetchInfo | forecast_time, level, param | Fetch data with given search arguments, return info |
| info | FetchInfoWithTypw | forecast_time, level, param, forecast_type | Fetch data with given search arguments including forecast_type, return info |
| | WriteToFile | table | Writes gived data to file |

## Missing

Variable Missing is used in Himan to represent a missing value. It's value is NaN.

## neons

Variable neons is used to connect to Oracle-based database. DEPRECATED.

## radon

Variable radon is used to connect to PostgreSQL-based database.

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| string | GetProducerMetaData | producer, string | Return a piece of metadata from some producer. Table in database is producer_meta |

## result

Variable result represents the metadata for the calculation result. It is an info class instance.

## write_options

Variable write_options can be used to control the process of writing file to disk. So far the only option available is to define whether or not to use a bitmap for missing values. 

    write_options.use_bitmap = false

| Option | Description | Default value |
|---|---|---|
| use_bitmap | Control whether bitmap is used for missing values (grib) | true |


# Other functions

## Es_

Calculate water vapor saturated pressure in Pa.

```
-- Es_(Temperature)
-- Temperature in Kelvin
-- If dewpoint temperature used instead of air temperature, 
-- actual water vapour pressure is calculated.
 
local T = 269.3
 
local E = Es_(T)
-- Returned value is (saturated) water vapour pressure in Pa
```

## LCL_

Find LCL (lifted condensation level) properties.

```
-- LCL_(Pressure, Temperature, Dewpoint)
-- Pressure in Pa, temperatures in Kelvin
 
local P = 100100
local T = 273.15
local TD = 270
 
local lcl = LCL_(P, T, TD)
 
-- Returned data is in Pa, K and g/kg
print("lcl pressure: " .. lcl.P .. " temperature: " .. lcl.T .. " specific humidity: " .. lcl.Q)
```

# Examples

To launch a lua script, we'll need the lua script itself and a himan configuration in json format.

The following json file can be used to launch a lua script.

## json

```
{
    "source_geom_name" : "RCR068",
    "target_geom_name" : "RCR068",
    "source_producer" : "1,230",
    "target_producer" : "230",
    "start_hour" : "1",
    "stop_hour" : "1",
    "step" : "1",
    "file_write" : "multiple",
    "origintime" : "latest",
 
    "processqueue" : [
    {
        "leveltype" : "height",
        "levels" : "2",
        "plugins" : [ {
            "name" : "luatool",
            "luafile" : [ "dewpoint-deficit.lua" ]
        } ]
    }
    ]
}
```

## ws5000m

Interpolates wind speed to 5000 meters

```
--[[
ws5000m.lua
 
Example lua script to calculate wind speed at 5000 meters.
]]
 
logger:Info("Calculating wind speed at 5000 meters")
 
wsparam = param("FF-MS")
 
-- "hitool" is a global variable which corresponds to hitool-plugin
-- returned variable data is a lua table
 
data = hitool:VerticalValue(wsparam, 5000)
 
result:SetParam(param("WS5000M-MS"))
 
-- SetValues() sets the table to the result-class which holds all the needed metadata
 
result:SetValues(data)
 
logger:Info("Writing results")
 
luatool:WriteToFile(result)
```

## Cloud base

Calculates cloud base height in feet.

```
-- Max height to check for cloud base [m]
local maxH = 14000
 
-- Threshold for N (cloud amount) to calculate base [%]
local Nthreshold = 0.55
 
local N = param("N-0TO1")
 
-- findh: 1 = first value (of threshold) from sfc upwards
local Nheight = hitool:VerticalHeight(N, 0, maxH, Nthreshold, 1)
 
if not Nheight then
    N = param("N-PRCNT")
     
    Nthreshold = 100 * Nthreshold
 
    Nheight = hitool:VerticalHeight(N, 0, maxH, Nthreshold, 1)
end
 
if not Nheight then
    logger:Error("No data found")
    return
end
 
-- N max value near the surface (15m ~ 50ft)
local lowNmax = hitool:VerticalMaximum(N, 0, 15)
 
if not lowNmax then
    logger:Error("No data found")
    return
end
 
local ceiling = {}
 
for i = 1, #lowNmax do
    local nh = Nheight[i]
    local nmax = lowNmax[i]
 
    local ceil = kFloatMissing
 
    -- Nthreshold is not always found for low clouds starting from ~sfc
 
    if nmax > Nthreshold then
        nh = 14
    end
 
    -- Result converted to feet:
    -- below 100ft at 50ft resolution  (15m ~ 50ft)
    -- below 10000ft at 100ft resolution  (10000ft = 3048m)
    -- above 10000ft at 500ft resolution
 
    if nh < 15 then
        ceil = math.floor(0.5 + nh/30.48*2) * 50
    elseif nh < 3048 then
        ceil = math.floor(0.5 + nh/30.48) * 100
    else
        ceil = math.floor(0.5 + nh/304.8*2) * 500
    end
 
    ceiling[i] = ceil
end
 
result:SetParam(param("CL-FT"))
result:SetValues(ceiling)
 
luatool:WriteToFile(result)
```
