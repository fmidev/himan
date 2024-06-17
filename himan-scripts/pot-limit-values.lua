-- Modify POT (=Thunderstorm probability) values in two ways:
-- * limit the maximum value to some constant
-- * reduce the value categorically as a function of lead time

local POT = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("POT-PRCNT"), current_forecast_type)

if not POT then
  return
end

function LimitMaximumValueDuringWinter(POT, maximum)

  -- Limit the value of POT to max 29% during winter time.
  -- 30% is the threshold that will show lightning in the
  -- weather symbol and we don't want to predict
  -- lightning in the winter (in Scandinavia).

  local m = tonumber(current_time:GetValidDateTime():String("%m"))

  -- winter is november to march
  if m >= 11 or m <= 3 then
    for i=1, #POT do
      local pot = POT[i]

      if pot == pot then
        POT[i] = math.min(maximum, pot)
      end
    end
  end

  return POT

end

function ReduceValue(POT)

  -- Reduce the value of POT as a function of lead time.
  -- We don't want to predict high probabilities for long
  -- lead times.

  local lead_time = current_time:GetStep():Hours()

  if lead_time <= 120 then
    for i=1,#POT do
      POT[i] = POT[i] - POT[i] * RampUp(24, 120, POT[i]) * 0.5
    end
  else
    for i=1,#POT do
      POT[i] = POT[i] * 0.5 - POT[i] * RampUp(123, 240, POT[i]) * 0.25
    end
  end

  return POT

end

POT = ReduceValue(POT)
POT = LimitMaximumValueDuringWinter(POT, 29)

result:SetParam(param("POT-PRCNT"))
result:SetValues(POT)
luatool:WriteToFile(result)
