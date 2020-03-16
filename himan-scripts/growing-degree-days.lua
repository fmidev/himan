-- Calculate growing degree days.
--
-- https://en.wikipedia.org/wiki/Growing_degree-day
--
-- Base temperature is 5 degrees. 
-- 
-- For example:
-- Daily (24h) mean t2m are: 5, 10, 7, 6, 3
-- Growing degree days value is: 8

local step = current_time:GetStep():Hours()

if step % 24 ~= 0 then
  logger:Info(string.format("Step is not a multiple of 24 (%d) -- skipping", step))
  return
end

local gdd = {}

local curtime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())

while true do

  logger:Info(string.format("Fetching mean temperature for a 24h period ending at %s (+%d)", curtime:GetValidDateTime():String("%Y%m%d%H"), curtime:GetStep():Hours()))

  local mean = luatool:FetchWithType(curtime, current_level, param("T-MEAN-K"), current_forecast_type)

  if mean then
    -- initialize frost sum array to zero
    if #gdd == 0 then
      for i=1,#mean do
        gdd[i] = 0
      end
    end

    for i=1,#mean do
      -- Note: following can be replaced with math.max() but that is ~100% slower (!)
      local m = mean[i] - 273.15 - 5
      if m > 0 then
        gdd[i] = gdd[i] + m
      end
    end
  end

  curtime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -24)

  if curtime:GetStep():Hours() <= 0 then
    break
  end

end

local agg = aggregation(HPAggregationType.kAccumulation, time_duration(HPTimeResolution.kHourResolution, current_time:GetStep():Hours()))
local par = param("GDD-C")

par:SetAggregation(agg)

result:SetParam(par)
result:SetValues(gdd)

luatool:WriteToFile(result)
