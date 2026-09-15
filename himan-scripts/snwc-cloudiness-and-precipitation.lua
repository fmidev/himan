-- Check consistency between total cloudiness and precipitation; first one
-- is modified accordingly.
--
-- Remove light precipitation during summer time, as sometime insects and birds
-- show up as precipitation in weather radar images (PDTK-74).
-- 
-- Remove light precipitation depending on precipitation form value (PDTK-229).
--
-- This script is nearly identical to precipitation-limit-values.lua, which is run
-- for the long Vire forecast. The only difference is that we also adjust cloudiness here,
-- whereas in long Vire cloud-consensus.lua handles the consistency of precipitation and clouds.

local CC = luatool:Fetch(current_time, current_level, param("N-0TO1"), current_forecast_type)
local RR = luatool:Fetch(current_time, current_level, param("RRR-KGM2"), current_forecast_type)
local POT_PRECF = luatool:Fetch(current_time, current_level, param("POTPRECF-N"), current_forecast_type)

if not CC or not RR or not POT_PRECF then
  return
end

local _CC = {}
local _RR = {}

local mon = tonumber(current_time:GetValidDateTime():String("%m"))

for i=1,#CC do
  _CC[i] = CC[i]
  _RR[i] = RR[i]

  -- PDTK-74
  if mon >= 5 and mon <= 8 and RR[i] <= 0.09 then
    _RR[i] = 0
  end

  -- PDTK-229
  if POT_PRECF[i] == 0 and _RR[i] > 0 and _RR[i] < 0.015 then
    _RR[i] = 0
  end
  if POT_PRECF[i] == 1 and _RR[i] > 0 and _RR[i] < 0.1 then
    _RR[i] = 0
  end
  if POT_PRECF[i] == 2 and _RR[i] > 0 and _RR[i] < 0.075 then
    _RR[i] = 0
  end
  if POT_PRECF[i] == 3 and _RR[i] > 0 and _RR[i] < 0.05 then
    _RR[i] = 0
  end
  if POT_PRECF[i] == 4 and _RR[i] > 0 and _RR[i] < 0.01 then
    _RR[i] = 0
  end
  if POT_PRECF[i] == 5 and _RR[i] > 0 and _RR[i] < 0.02 then
    _RR[i] = 0
  end

  -- If there is even light precipitation, there should also be clouds
  if _RR[i] > 0.01 then
    _CC[i] = math.max(_CC[i], 0.5)
  end
end

result:SetParam(param("N-0TO1"))
result:SetValues(_CC)
luatool:WriteToFile(result)

rrparam = param("RRR-KGM2")
rrparam:SetAggregation(aggregation(HPAggregationType.kAccumulation, time_duration("01:00")))
result:SetParam(rrparam)
result:SetValues(_RR)
luatool:WriteToFile(result)
