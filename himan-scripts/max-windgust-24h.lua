-- Calculate windgust duration within 24h
-- Function reads past 24h from given time. Works only with ECMWF data.

function WriteToFile(gsum, ftime)
  local start = ftime:GetStep():Hours()
  local agg = aggregation(HPAggregationType.kMaximum, time_duration(HPTimeResolution.kHourResolution, 24))
  local par = param("FFG24H-MS")
  par:SetAggregation(agg)

  result:SetParam(par)
  result:SetValues(gsum)
  luatool:WriteToFile(result)
end

function GustHours(curTime)

  local step = curTime:GetStep():Hours()

  if step == 0 then
    return
  end

  -- calculate 24h gust

  local gsum = {}

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

    local  gust = luatool:Fetch(curTime, current_level, param("FFG-MS"), current_forecast_type)

    if gust then
      if #gsum == 0 then
        for i=1, #gust do
         gsum[i] = 0
        end
      end

      for i=1, #gust do
        gsum[i] = math.max(gsum[i], gust[i])
      end
    end

    step = curTime:GetStep():Hours()
  end

  if #gsum == 0 then  
    return
  end

  WriteToFile(gsum, curTime)

  return gsum

end

local step = current_time:GetStep():Hours()

if step < 24 then
  logger:Info(string.format("Step is less than first 24h -- skipping", step))
  return
end

GustHours(current_time)
