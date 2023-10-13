-- Convert ECMWF precipitation form values to match FMI values.
--
-- 0: No precipitation form -------> missing
-- 1: Rain -----> 1 rain
-- 3: Freezing rain -----> 5 freezing rain
-- 5: Snow -----> 3 snow
-- 6: Wet snow (starting to melt) -----> 2 sleet
-- 7: Mixture of rain and snow -----> 2 sleet
-- 8: Ice pellets -----> 8 ice pellets
-- 12: Freezing drizzle -----> 4 freezing drizzle
--

logger:Info("Converting ECMWF precipitation form values to FMI precipitation form values")

local MISS = missing
local pf = luatool:Fetch(current_time, level(HPLevelType.kGround, 0), param("PRECFORM-N"), current_forecast_type)

if not pf then
  return
end

local i = 0
local fmipf = {}

for i=1, #pf do
  local _pf = pf[i]
  local res = MISS

  if _pf == 0 then
    -- no precipitation form
    res = MISS
  elseif _pf == 1 then
    -- rain
    res = 1
  elseif _pf == 3 then
    -- freezing rain
    res = 5
  elseif _pf == 5 then
    -- snow
    res = 3
  elseif _pf == 6 or _pf == 7 then
    -- sleet
    res = 2
  elseif _pf == 8 then
    -- ice pellets
    res = 8
  elseif _pf == 12 then
    -- freezing drizzle
    res = 4
  end

  fmipf[i] = res
end

result:SetValues(fmipf)
result:SetParam(param("PRECFORM4-N"))

luatool:WriteToFile(result)
