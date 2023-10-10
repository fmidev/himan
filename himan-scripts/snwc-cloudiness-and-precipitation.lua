--
-- SmartMet NWC parameters
--
-- Check consistency between total cloudiness and precipitation; first one
-- is modified accordingly.
--
-- Also remove light precipitation during summer time, as sometime insects and birds
-- show up as precipitation in weather radar images (PDTK-74).
-- 

local CC = luatool:Fetch(current_time, current_level, param("N-0TO1"), current_forecast_type)
local RR = luatool:Fetch(current_time, current_level, param("RRR-KGM2"), current_forecast_type)

if not CC or not RR then
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
