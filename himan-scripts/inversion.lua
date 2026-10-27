--[[

inversion

Find the maximum inversion between 0 and 573m

STU-34529

]]

logger:Info("Calculating inversion")

hitool:SetHeightUnit(HPParameterUnit.kM)

-- Get surface temperature
local t = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), param("T-K"), current_forecast_type)

-- Get vertical maximum temperature in 2m to 573m
local max_t = hitool:VerticalMaximum(param("T-K"),2,573)

local inversion = {}

for i=1, #max_t do
  if IsMissing(t[i]) then
    inversion[i] = missing
  else inversion[i] = max_t[i] - t[i]
  end
end

for i=1, #inversion do
  if inversion[i] < 0 then
    inversion[i] = 0
  end
end

-- Get height of maximum inversion as inversion top

-- Set search height grids
local lower_height = {}
local upper_height = {}

for i=1, #t do
  lower_height[i] = 2
  upper_height[i] = 573
end

local inversion_top = hitool:VerticalHeightGrid(param("T-K"),lower_height,upper_height,max_t,1)

-- Meters to feet
for i=1, #inversion_top do
  if IsMissing(inversion_top[i]) then
    -- Do nothing
  else
    inversion_top[i] = inversion_top[i] * 3.28084
  end
end

-- Set inversion base at 0 ft
local inversion_base = {}

for i=1, #t do
  if IsMissing(t[i]) then
    inversion_base[i] = missing
  else
    inversion_base[i] = 0
  end
end

result:SetParam(param("INVERSION-K"))
result:SetValues(inversion)

logger:Info("Writing INVERSION-K data to file")
luatool:WriteToFile(result)

result:SetParam(param("INVERSION-TOP-FT"))
result:SetValues(inversion_top)

logger:Info("Writing INVERSION-TOP-FT data to file")
luatool:WriteToFile(result)

result:SetParam(param("INVERSION-BASE-FT"))
result:SetValues(inversion_base)

logger:Info("Writing INVERSION-BASE-FT data to file")
luatool:WriteToFile(result)
