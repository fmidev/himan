--[[

FORECAST-RELIABILITY

Classify forecast reliability from the standard deviation of
2 meter temperature (T-STDDEV-K, MEPSMTA) for simplicity.

Result is an integer index:

  3 = good        (sigma < 0.8 K)
  2 = fairly good (0.8 <= sigma < 1.5 K)
  1 = poor        (1.5 <= sigma < 2.5 K)
  0 = very poor   (sigma >= 2.5 K)

]]

-- classification limits (K)
local good = 0.8
local fairly_good = 1.5
local poor = 2.5

local stddev_param = param("T-STDDEV-K")
local l = level(HPLevelType.kHeight, 2)

-- MEPSMTA data is stored with forecast type "statistical"
local stddev = luatool:Fetch(current_time, l, stddev_param, forecast_type(HPForecastType.kStatisticalProcessing))

if not stddev then
  logger:Error("Standard deviation of 2m temperature not found")
  return
end

local reliability = {}

for i = 1, #stddev do
  local sd = stddev[i]

  if IsMissing(sd) then
    reliability[i] = missing
  elseif sd < good then
    reliability[i] = 3
  elseif sd < fairly_good then
    reliability[i] = 2
  elseif sd < poor then
    reliability[i] = 1
  else
    reliability[i] = 0
  end
end

result:SetParam(param("FORECAST-RELIABILITY-N"))
result:SetValues(reliability)

logger:Info("Writing result data to file")
luatool:WriteToFile(result)
