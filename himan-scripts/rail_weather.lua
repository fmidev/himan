local next_time = forecast_time(current_time:GetOriginDateTime(),current_time:GetValidDateTime())
next_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, 6)
local prev_time = forecast_time(current_time:GetOriginDateTime(),current_time:GetValidDateTime())
prev_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -6)

-- Implementation of the Regulation 5 conditions for railroad operation
-- Conditions in particular: 
-- 1. Temperature -5C or less
-- 2. Snowfallrate 5cm/12h; Calculate +/-6h
-- 3. Wind Speed 5 m/s or above

local min_t = luatool:Fetch(current_time, level(HPLevelType.kHeight,2), param("TMIN6H-K"), current_forecast_type)

if not min_t then
	min_t = luatool:Fetch(current_time,
	level(HPLevelType.kHeight,2), param("TMIN3H-K"), current_forecast_type)
end

local timestep = configuration:GetForecastStep():Hours()
local gust_param = param("FFG-MS", aggregation(HPAggregationType.kMaximum, time_duration(HPTimeResolution.kHourResolution, timestep)), processing_type())

local max_ws = luatool:Fetch(current_time, level(HPLevelType.kHeight,10), gust_param, current_forecast_type)

local snow_next = luatool:Fetch(next_time, current_level, param('SNACC-KGM2'), current_forecast_type) 
local snow_prev = luatool:Fetch(prev_time, current_level, param('SNACC-KGM2'), current_forecast_type)

if min_t == nil or snow_next == nil or (snow_prev == nil and current_time:GetStep():Hours() ~= 6) or max_ws == nil then  
    return
end

local res = {}
local snow = {}

scale_unit = 1

if currentProducerName == "MEPSMTA" then
    scale_unit = 1000
end

for i=1, #min_t do
  res[i] = 0

  -- Calculate the index
  -- Determine forecast value Missing or 1;

  -- for first time step no snow accumulation is available
  if current_time:GetStep():Hours() == 6 then
    snow[i] = snow_next[i]
  else
    snow[i] = snow_next[i] - snow_prev[i]
  end

  if ( min_t[i] <= (-4.9 + kKelvin) and snow[i] >= 0.0049 and max_ws[i] >= 4.8) then
    res[i] = 1
  end
end

result:SetParam(param("RAIL-N"))
result:SetValues(res)
luatool:WriteToFile(result)
