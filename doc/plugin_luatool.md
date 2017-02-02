luatool-plugin enables himan core functionality to be used through lua scripts. Lua is a dynamically typed scripting language, and is very useful when only very simple and computationally modest processing is needed. Luatool internally uses luabind-library to expose certain parts of the core functionality.

Basic functionality:

* luatool-plugin works like any other post processing plugin, although it's classified as an infrastructure plugin
* himan will launch as many threads as are needed and each of the threads will execute the script(s) simultaneously
* lua scripts control the writing of output files
* lua scripts can access hitool-plugin functionality
* interpolation, cache, database access etc are all done automatically like with any other plugin
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
    kTopOfAtmosphere = 8
    kPressure = 100
    kMeanSea = 102
    kAltitude = 103
    kHeight = 105
    kHybrid = 109
    kGndLayer = 112
    kDepth = 160
    kEntireAtmosphere = 200
    kEntireOcean = 201

HPParameterUnit
    kHPa = 5 // hectopascal
    kM = 8 // meters

HPTimeResolution
    kUnknownTimeResolution = 0
    kHourResolution = 1
    kMinuteResolution = 2

HPFileType
    kUnknownFile = 0
    kGRIB1 = 1
    kGRIB2 = 2
    kGRIB = 3
    kQueryData = 4
    kNetCDF = 5

HPProjectionType
    kUnknownProjection = 0
    kLatLonProject = 1
    kRotatedLatLonProjection = 2
    kStereographicProjection = 3

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

aggregation is a parameter component, defining for example that it is a cumulative

| Return value  | Name | Arguments | Description | 
|---|---|---|---|
| string | ClassName | | Returns class name |
| HPAggregationType | GetType | | Returns aggregation type (for example: cumulative) |
|   | SetType | HPAggregationType  | Set aggregation type |
| HPTimeResolution | GetTimeResolution | | Returns time resolution type (for example: hour) |
|   | SetTimeResolution | HPTimeResolution  | Set time resolution type |
| number | GetTimeResolution | | Returns time resolution value |
|   | SetTimeResolution | number  | Set time resolution value |

## aggregation

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
| number | GetStep | | Returns time step (valid time - origin time) |
| HPTimeResolution | GetStepResolution | | Returns step time resolution type (minute, hour) |
| | SetStepResolution | HPTimeResolution | Set step time resolution type|

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
| point | GetLatLon | number | Returns latlon coordinates of given grid point id |
| table | GetValues | | Returns grid data contents |
| | SetValues | table | Sets grid data contents |
| number | GetMissingValue | | Returns missing value |
| | SetMissingValue | number | Sets missing value |


# Examples

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

## Cloud baes

Calculates cloud base height in feet.

```
local kFloatMissing = kFloatMissing
 
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
 
    if nh ~= kFloatMissing and nmax ~= kFloatMissing then
 
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
    end
 
    ceiling[i] = ceil
end
 
result:SetParam(param("CL-FT"))
result:SetValues(ceiling)
 
luatool:WriteToFile(result)
```