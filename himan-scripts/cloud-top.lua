-- Highest (significant) cloud top in feet above the surface.
-- Significant cloudiness is defined as at least 55% in amount.
--  junila 4/2017

logger:Info("Calculating Cloud top in feet")

local Missing = missing

local l = level(HPLevelType.kHeight, 0)
local Ceiling = luatool:Fetch(current_time, l, param("CEIL-2-M"), current_forecast_type)

if not Ceiling then
  return
end

local Base = {}

for i = 1, #Ceiling do

  local ceil = Ceiling[i]
  local base = Missing

  if ceil == ceil then
    base = ceil
  end

  Base[i] = base

end

-- Max height to check for cloud top [m]

local maxH = {}

for i = 1, #Ceiling do
  maxH[i] = 15000
end

-- Threshold for N (cloud amount) to calculate top [%]

local Nthreshold = {}

for i = 1, #Ceiling do
  Nthreshold[i] = 0.55
end

local N = param("N-0TO1")

local CloudTop = hitool:VerticalHeightGreaterThanGrid(N, Base, maxH, Nthreshold, 0)

if not CloudTop then
  N = param("N-PRCNT")
  -- N-PRCNT values also 0...1 despite the name
  CloudTop = hitool:VerticalHeightGreaterThanGrid(N, Base, maxH, Nthreshold, 0)

end

if not CloudTop then
  return
end


local CLDTOP = {}

for i = 1, #CloudTop do

  local TOP = Missing
  local top = CloudTop[i]

  if top == top then
    TOP = top*3.2808    -- meters to feet
  end

  CLDTOP[i] = TOP

end


result:SetParam(param("CLDTOP-FT"))
result:SetValues(CLDTOP)

luatool:WriteToFile(result)
