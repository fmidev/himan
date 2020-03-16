-- Calculate frost sum from 2 meter daily mean temperature
-- Positive values do not increase the value of frost sum
-- 
-- For example:
-- Daily (24h) mean t2m are: -4, -6, -1, +2, -5
-- Frost sum is: -16 (sic)
--

local step = current_time:GetStep():Hours()

if step % 24 ~= 0 then
  logger:Info(string.format("Step is not a multiple of 24 (%d) -- skipping", step))
  return
end

local frostSum = {}

local curtime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())

while true do

  logger:Info(string.format("Fetching mean temperature for a 24h period ending at %s", curtime:GetValidDateTime():String("%Y%m%d%H")))

  local mean = luatool:FetchWithType(curtime, current_level, param("T-MEAN-K"), current_forecast_type)

  if mean then
    -- initialize frost sum array to zero
    if #frostSum == 0 then
      for i=1,#mean do
        frostSum[i] = 0
      end
    end

    for i=1,#mean do
      -- positive values should not affect the value of frost sum
      -- Note: following can be replaced with math.min() but that is ~100% slower (!)
      local m = mean[i] - 273.15
      if m < 0 then
        frostSum[i] = frostSum[i] + m
      end
    end
  end

  curtime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -24)

  if curtime:GetStep():Hours() <= 0 then
    break
  end
end

local agg = aggregation(HPAggregationType.kAccumulation, time_duration(HPTimeResolution.kHourResolution, current_time:GetStep():Hours()))
local par = param("FROSTSUM-C")

par:SetAggregation(agg)

result:SetParam(par)
result:SetValues(frostSum)

luatool:WriteToFile(result)
