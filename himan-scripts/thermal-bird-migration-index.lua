-- bird migration index, 0: bad weather for migrating birds, 1: good weather for migrating birds

-- global shortwave radiation W/m2
local globparam = param("RADGLO-WM2")

-- boundary level height m
local blheightparam = param("MIXHGT-M")

-- boundary level wind speed m/s, taken at 100m
local windparam = param("FF-MS")

-- empirically derived limits for each parameter

-- global radiation
-- upward slope
local glob_min = 150
local glob_max = 300

-- boundary layer height
-- upward slope
local bl_min = 200
local bl_max = 800

-- wind speed at 100m
-- downward slope
local ws_min = 7
local ws_max = 11

local heightlevel = level(HPLevelType.kHeight, 0)

local globdata = luatool:FetchWithType(current_time, heightlevel, globparam, current_forecast_type)
local blheightdata = luatool:FetchWithType(current_time, heightlevel, blheightparam, current_forecast_type)

hitool:SetHeightUnit(HPParameterUnit.kM)
local winddata = hitool:VerticalValue(windparam, 100)

if not globdata or not blheightdata or not winddata then
  logger:Error("Some data not found")
  return
end

function interpolate(x, x1, x2, y1, y2)
  if x <= x1 then
    return y1
  elseif x >= x2 then
    return y2
  end

  -- linear interpolation between x1 and x2
  return ((x2 - x) * y1 + (x - x1) * y2) / (x2 - x1)

end

local bird_index = {}
for i=1,#globdata do
  local glob_index = interpolate(globdata[i], glob_min, glob_max, 0, 1)
  local blheight_index = interpolate(blheightdata[i], bl_min, bl_max, 0, 1)
  local wind_index = interpolate(winddata[i], ws_min, ws_max, 1, 0)
  bird_index[i] = math.min(glob_index, blheight_index, wind_index)
end

result:SetParam(param("TBINDEX-0TO1"))
result:SetValues(bird_index)
luatool:WriteToFile(result)
