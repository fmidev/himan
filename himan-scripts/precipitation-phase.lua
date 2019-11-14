--
-- Probability of precipitation phase P(W) given 2 meter temperature (C) and relative humidity (%).
--
-- Algorithm is a statistical fitting between conventional observations and radar data.
--
-- Jarmo Koistinen FMI, Gregorz Ciach
--
-- https://wiki.fmi.fi/pages/viewpage.action?pageId=21139101&preview=/21139101/21397830/IL_olomuototuote_JK.ppt
--
-- The following thresholds have been found by FMI:
--
-- water, when P(W) > 0.8
-- sleet, when 0.2 <= P(W) <= 0.8
-- snow,  when P(W) < 0.2


local T = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 2), param("T-K"), current_forecast_type)
local RH = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 2), param("RH-0TO1"), current_forecast_type)
local RHscale = 100

if not RH then
  RH = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 2), param("RH-PRCNT"), current_forecast_type)
  RHscale = 1
end

if not RH or not T then
  return
end

local res = {}

for i=1, #T do
  res[i] = 1 / (1 + math.exp(22 - 2.7 * (T[i] - 273.15) - 0.2 * RH[i] * RHscale))
end

result:SetParam(param("PRECPHASE-0TO1"))
result:SetValues(res)
luatool:WriteToFile(result)
