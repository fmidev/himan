--
-- Remove light precipitation during summer time, as sometime insects and birds
-- show up as precipitation in weather radar images (PDTK-74).
-- 
-- Remove light precipitation based on the precipitation form (PDTK-229).
--

local RR = luatool:Fetch(current_time, current_level, param("RRR-KGM2"), current_forecast_type)
local POT_PRECF = luatool:Fetch(current_time, current_level, param("POTPRECF-N"), current_forecast_type)

if not RR then
  plogger:Error("VIRE RRR-KGM2 data not found, aborting")
  return
end
if not POT_PRECF then
  plogger:Error("VIRE POTPRECF-N data not found, aborting")
  return
end

local _RR = {}

local mon = tonumber(current_time:GetValidDateTime():String("%m"))

if mon >= 5 and mon <= 8 then
  for i=1,#RR do
    -- PDTK-74
    if RR[i] > 0 and RR[i] <= 0.09 then
      RR[i] = 0
    end
  end
end

-- PDTK-229
for i = 1, #RR do
  if POT_PRECF[i] == 0 and RR[i] > 0 and RR[i] < 0.015 then
    RR[i] = 0
  end
  if POT_PRECF[i] == 1 and RR[i] > 0 and RR[i] < 0.1 then
    RR[i] = 0
  end
  if POT_PRECF[i] == 2 and RR[i] > 0 and RR[i] < 0.075 then
    RR[i] = 0
  end
  if POT_PRECF[i] == 3 and RR[i] > 0 and RR[i] < 0.05 then
    RR[i] = 0
  end
  if POT_PRECF[i] == 4 and RR[i] > 0 and RR[i] < 0.01 then
    RR[i] = 0
  end 
  if POT_PRECF[i] == 5 and RR[i] > 0 and RR[i] < 0.02 then
    RR[i] = 0
  end
end


rrparam = param("RRR-KGM2")
rrparam:SetAggregation(aggregation(HPAggregationType.kAccumulation, time_duration("01:00")))
result:SetParam(rrparam)
result:SetValues(RR)
luatool:WriteToFile(result)