--
-- SmartMet NWC parameters
--
-- Check consistency between total cloudiness and precipitation; first one
-- is modified accordingly
-- 

local CC = luatool:FetchWithType(current_time, current_level, param("N-0TO1"), current_forecast_type)
local RR = luatool:FetchWithType(current_time, current_level, param("RRR-KGM2"), current_forecast_type)

if not CC or not RR then
  return
end

local _CC = {}

for i=1,#CC do
  _CC[i] = CC[i]

  if RR[i] > 0.01 then
    _CC[i] = math.max(_CC[i], 0.5)
  end
end

result:SetParam(param("N-0TO1"))
result:SetValues(_CC)
luatool:WriteToFile(result)

