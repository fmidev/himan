--
-- Limit the value of POT to max 29% during winter time.
-- 30% is the threshold that will show lightning in the
-- weather symbol and we don't want to predict
-- lightning in the winter (in Scandinavia).
--

local POT = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("POT-PRCNT"), current_forecast_type)

if not POT then
  return
end

local m = tonumber(current_time:GetValidDateTime():String("%m"))

-- winter is november to march
if m >= 11 or m <= 3 then
  for i=1, #POT do
    local pot = POT[i]

    if pot == pot then
      POT[i] = math.min(29, pot)
    end
  end
end

result:SetParam(param("POT-PRCNT"))
result:SetValues(POT)
luatool:WriteToFile(result)
