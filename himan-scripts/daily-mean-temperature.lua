-- Calculate daily mean temperature, simple arithmetic mean.
-- Calculation is done only on 00 hour of forecast, so it's not a moving
-- window mean.
-- Function reads past 24h from given time. Works only with ECMWF data.

function WriteMeanToFile(mean, ftime)
  local start = ftime:GetStep():Hours() - 24
  local agg = aggregation(HPAggregationType.kAverage, time_duration(HPTimeResolution.kHourResolution, 24))
  local par = param("T-MEAN-K")
  par:SetAggregation(agg)

  result:SetParam(par)
  result:SetValues(mean)
  luatool:WriteToFile(result)
end

function DailyMeanTemperature(curTime)

  local step = curTime:GetStep():Hours()

  if step == 0 then
    return
  end

  -- calculate 24h mean temperature

  local tsum = {}

  local stopStep = math.max(curTime:GetStep():Hours() - 24, 0)

  local count = 0

  while true do
    local stepAdjustment = -1

    if step >= 150 then
      stepAdjustment = -6
    elseif step >= 93 then
      stepAdjustment = -3
    end

    curTime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, stepAdjustment)

    if curTime:GetStep():Hours() < stopStep or curTime:GetStep():Hours() < 0 then
      break
    end

    local t2m = luatool:Fetch(curTime, current_level, param("T-K"), current_forecast_type)

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

    step = curTime:GetStep():Hours()
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

local step = current_time:GetStep():Hours()

if step % 24 ~= 0 then
  logger:Info(string.format("Step is not a multiple of 24 (%d) -- skipping", step))
  return
end

DailyMeanTemperature(current_time)
