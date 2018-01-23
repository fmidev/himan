-- Calculate frost sum from 2 meter daily mean temperature
-- Positive values do not increase the value of frost sum
-- 
-- For example:
-- Daily (24h) mean t2m are: -4, -6, -1, +2, -5
-- Frost sum is: -16 (sic)
--
-- Script should only be used with ECMWF data

-- Calculate daily (24h) mean temperature by reading all t2m values
-- found and doing simple arithmetic mean.
-- Function reads past 24h from given time.

function WriteMeanToFile(mean, ftime)
  local start = ftime:GetStep() - 24
  local agg = aggregation(HPAggregationType.kAverage, HPTimeResolution.kHourResolution, 24, 999999)
  local par = param("T-MEAN-K")
  par:SetAggregation(agg)

  result:SetParam(par)
  result:SetValues(mean)
  luatool:WriteToFile(result)
end

function DailyMeanTemperature(atime, lastTime)

  local curTime = forecast_time(atime, lastTime)

  if curTime:GetStep() == 0 then
    return
  end

  local mean = luatool:FetchWithType(curTime, current_level, param("T-MEAN-K"), current_forecast_type)

  if mean then
    -- got lucky
    return mean
  end

  -- calculate 24h mean temperature

  local tsum = {}

  local stopStep = math.max(curTime:GetStep() - 24, 0)

  local count = 0


  local step = curTime:GetStep()

  while true do
    local stepAdjustment = -1

    if step >= 150 then
      stepAdjustment = -6
    elseif step >= 93 then
      stepAdjustment = -3
    end

--    lastTime:Adjust(HPTimeResolution.kHourResolution, -configuration:GetForecastStep())
    lastTime:Adjust(HPTimeResolution.kHourResolution, stepAdjustment)
    local time = forecast_time(atime, lastTime)

    if time:GetStep() < stopStep or time:GetStep() < 0 then
      break
    end

    local t2m = luatool:FetchWithType(time, current_level, param("T-K"), current_forecast_type)

    if t2m then
      if #tsum == 0 then
        for i=1, #t2m do
         tsum[i] = 0
        end
      end

      for i=1, #t2m do
        tsum[i] = tsum[i] + t2m[i] 
      end
      count = count + 1
    end

    step = time:GetStep()
  end

  if #tsum == 0 then  
    return
  end

  mean = {}

  for i=1,#tsum do
    mean[i] = tsum[i] / count
  end

  WriteMeanToFile(mean, curTime)

  return mean

end

local atime = current_time:GetOriginDateTime()
local vtime = raw_time(current_time:GetValidDateTime():String("%Y-%m-%d %H:%M:%S"))

local step = current_time:GetStep()

if step % 24 ~= 0 then
  logger:Info(string.format("Step is not a multiple of 24 (%d) -- skipping", step))
  return
end

local frostSum = {}

while true do

  local curtime = raw_time(vtime:String("%Y-%m-%d %H:%M:%S"))
 
  logger:Info(string.format("Fetching mean temperature for a 24h period ending at %s", curtime:String("%Y%m%d%H")))

  local mean = DailyMeanTemperature(atime, curtime)

  if mean then
    -- initialize frost sum array to zero
    if #frostSum == 0 then
      for i=1,#mean do
        frostSum[i] = 0
      end
    end

    for i=1,#mean do
      -- positive values should not affect the value of frost sum
      local m = math.min(mean[i] - 273.15, 0)

      frostSum[i] = frostSum[i] + m
    end
  end

  if tonumber(curtime:String("%Y%m%d%H")) <= tonumber(atime:String("%Y%m%d%H")) then
    break
  end

  vtime:Adjust(HPTimeResolution.kHourResolution, -24)
end

local agg = aggregation(HPAggregationType.kAccumulation, HPTimeResolution.kHourResolution, current_time:GetStep(), 0)
local par = param("FROSTSUM-C")

par:SetAggregation(agg)

result:SetParam(par)
result:SetValues(frostSum)

luatool:WriteToFile(result)
