-- Cloud ceiling code, smarttool script including comments by Simo Neiglick
-- https://wiki.fmi.fi/display/PROJEKTIT/Ceiling_ft
-- partio 2/2015  / junila 5/2015 / partio 9/2015  / junila 8/2015  / junila 2/2017

logger:Info("Calculating Cloud ceiling in meters")

-- Max height to check for cloud base [m]
local maxH = 15000

-- Threshold for N (cloud amount) to calculate base [%]
local Nthreshold = 0.55

-- findh: 1 = first value (of threshold) from sfc upwards
local Nheight = hitool:VerticalHeightGreaterThan(param("N-0TO1"), 0, maxH, Nthreshold, 1)

if not Nheight then
  Nheight = hitool:VerticalHeightGreaterThan(param("N-PRCNT"), 0, maxH, Nthreshold * 100, 1)
end

if not Nheight then
  return
end

local ceiling = {}

for i = 1, #Nheight do
  local nh = Nheight[i]
  ceiling[i] = nh
end

result:SetParam(param("CEIL-M"))
result:SetValues(ceiling)

luatool:WriteToFile(result)

