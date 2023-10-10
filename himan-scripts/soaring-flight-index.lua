-- soaring flight index, 0: bad weather, 1: good weather

local MISS = missing

-- global shortwave radiation W/m2
local globparam = param("RADGLO-WM2")

-- boundary level height m
local blheightparam = param("MIXHGT-M")

-- boundary level wind speed, taken at 100m
local windparam = param("FF-MS")

-- middle and high cloud cover 0-1
local cmparam = param("NM-0TO1")
local chparam = param("NH-0TO1")

-- land sea mask 0-1
local lsmparam = param("LC-0TO1")

-- global radiation
-- upward slope
local glob_min = 250
local glob_max = 400

-- boundary layer height
-- upward slope
local bl_min = 500
local bl_max = 1800

-- boundary layer height wind speed
-- downward slope
local ws_min = 6
local ws_max = 10

local heightlevel = level(HPLevelType.kHeight, 0)

local globdata = luatool:Fetch(current_time, heightlevel, globparam, current_forecast_type)
local blheightdata = luatool:Fetch(current_time, heightlevel, blheightparam, current_forecast_type)
local chdata = luatool:Fetch(current_time, heightlevel, chparam, current_forecast_type)
local cmdata = luatool:Fetch(current_time, heightlevel, cmparam, current_forecast_type)
local lsmdata = luatool:Fetch(current_time, heightlevel, lsmparam, current_forecast_type)

hitool:SetHeightUnit(HPParameterUnit.kM)
local winddata = hitool:VerticalValue(windparam, 100)

if not globdata or not blheightdata or not winddata or not chdata or not cmdata or not lsmdata then
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

local soaring_index = {}
for i=1,#globdata do

  local glob_index = interpolate(globdata[i], glob_min, glob_max, 0, 1)
  local blheight_index = interpolate(blheightdata[i], bl_min, bl_max, 0, 1)
  local wind_index = interpolate(winddata[i], ws_min, ws_max, 1, 0)

  local lsm_index = MISS

  -- step function at 0.5

  lsm_index = math.floor(lsmdata[i] + 0.5)

  local cc_index = MISS

  cc = math.max(chdata[i], cmdata[i])

  if cc <= 0.1 then
    cc_index = interpolate(cc, 0, 0.1, 0.8, 1)
  elseif cc >= 0.8 then
    cc_index = interpolate(cc, 0.8, 1.0, 1.0, 0.4)
  else
    cc_index = 1
  end

  soaring_index[i] = math.min(glob_index, blheight_index, wind_index, cc_index, lsm_index)

end

result:SetParam(param("SFINDEX-0TO1"))
result:SetValues(soaring_index)
luatool:WriteToFile(result)
