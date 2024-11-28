--
-- Make sure that wind speed and gust are in correct relation to each other.
--

local ffgparam = param("FFG-MS")
local ffparam = param("FF-MS")


function GetGust()
  local step = configuration:GetForecastStep():Hours()
  -- typically gust is available for 1, 3 and 6 hour aggregation periods
  if step == 2562047788 then
    steps = {1,3,6}
    for i=1,#steps do
	step=steps[i]
      ffgparam:SetAggregation(aggregation(HPAggregationType.kMaximum, time_duration(HPTimeResolution.kHourResolution, step)))
      local FFG = luatool:Fetch(current_time, current_level, ffgparam, current_forecast_type)
      if FFG then
	return FFG
      end
    end
  else
    ffgparam:SetAggregation(aggregation(HPAggregationType.kMaximum, time_duration(HPTimeResolution.kHourResolution, step)))
    return luatool:Fetch(current_time, current_level, ffgparam, current_forecast_type)
  end

  return nil
end

local FF = luatool:Fetch(current_time, current_level, ffparam, current_forecast_type)
local FFG = GetGust()

if not FF or not FFG then
  return
end

local min_gust_factor = 1.1
local max_gust_base = 25

local _FF = {}
local _FFG = {}

for i=1, #FF do
  local ff = FF[i]
  local ffg = FFG[i]

  -- Wind gust must be
  -- * at least as high as min_gust_factor * wind speed
  -- * at most as high as max_gust_base + wind speed
  ffg = math.max(ffg, ff * min_gust_factor)
  ffg = math.min(ffg, ff + max_gust_base)

  _FFG[i] = ffg
end

result:SetParam(ffgparam)
result:SetValues(_FFG)
luatool:WriteToFile(result)
