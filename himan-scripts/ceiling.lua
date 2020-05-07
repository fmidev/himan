-- Cloud ceiling code, smarttool script including comments by Simo Neiglick
-- https://wiki.fmi.fi/display/PROJEKTIT/Ceiling_ft
-- partio 2/2015  / junila 5/2015 / partio 9/2015  / junila 8/2015  / junila 2/2017

logger:Info("Calculating Cloud ceiling in meters")

local Missing = missing

-- Max height to check for cloud base [m]
local maxH = 15000

-- Threshold for N (cloud amount) to calculate base [%]
local Nthreshold = 0.55

local N = param("N-0TO1")

-- findh: 1 = first value (of threshold) from sfc upwards
local Nheight = hitool:VerticalHeightGreaterThan(N, 0, maxH, Nthreshold, 1)

if not Nheight then
  N = param("N-PRCNT")
  -- N-PRCNT values also 0...1 despite the name
  Nheight = hitool:VerticalHeightGreaterThan(N, 0, maxH, Nthreshold, 1)

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
result:SetMissingValue(Missing)
result:SetValues(ceiling)

luatool:WriteToFile(result)

