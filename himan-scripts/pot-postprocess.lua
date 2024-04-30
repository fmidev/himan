--
-- 1. Add a smoothing function to POT data: sometimes the probability will
--    go from 0 to 100% in to adjacent grid points, which is neither realistic
--    nor visually pleasing. Smoothing is done with a 3x3 grid point filter.
-- 2. Limit the value to max 29% during winter time. 30% is the threshold that 
--    will show lighting in the weather symbol and we don't want to predict 
--    lighting in the winter (in Scandinavia).
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

local POTmat = matrix(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
POTmat:SetValues(POT)

local filter = matrix(3, 3, 1, missing)    
filter:Fill(1/9.)

local POT = Filter2D(POTmat, filter, configuration:GetUseCuda()):GetValues()

result:SetParam(param("POT-PRCNT"))
result:SetValues(POT)
luatool:WriteToFile(result)
