--
-- Remove light precipitation during summer time, as sometime insects and birds
-- show up as precipitation in weather radar images (PDTK-74).
-- 

local RR = luatool:Fetch(current_time, current_level, param("RRR-KGM2"), current_forecast_type)

if not RR then
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

rrparam = param("RRR-KGM2")
rrparam:SetAggregation(aggregation(HPAggregationType.kAccumulation, time_duration("01:00")))
result:SetParam(rrparam)
result:SetValues(RR)
luatool:WriteToFile(result)
