logger:Info("Calculating Clear air turbulence base,top,max flight level")
-- The script runs a vertical search of the atmosphere to find the base and top of turbulence above a threshold as well as the maximum intensity
-- Base and top thus describe the vertical extend of a patch of turbulence

local MISS = missing
local CATParam = param("TI2-S2")

-- We set the vertical search function to work with pressure based vertical coordinate
-- The reasoning is that the pressures can be directly converted into flight levels (FL)
hitool:SetHeightUnit(HPParameterUnit.kHPa)

-- We define the upper and lower bounds for the vertical search
-- FL070 and FL500 are therefore converted to pressure in hPa
-- pressure [hPa] at bottom flight level 070
local lowlimit = 782
-- pressure [hPa] at top flight level 500
local highlimit = 116
-- threshold for turbulence parameter, note: in smartmet workstation the param is scaled with factor 10^7
local threshold = 15/10000000

-- Find the base of Clear Air Turbulence, i.e. pressure at which TI2 crosses the threshold with an upward slope
-- Data is missing where there is no turbulence of intensity above threshold found in the search range
local basedata = hitool:VerticalHeightGreaterThan(CATParam, lowlimit, highlimit, threshold, 1)

-- Switch to grid based limit values for top
local highlimitdata = {}
local thresholddata = {}

for i=1, #basedata do
  highlimitdata[i] = highlimit
  thresholddata[i] = threshold
end

-- Find the top of the layer of Clear Air Turbulence, i.e. pressure at which TI2 crosses the threshold with a downward slope
-- Search starts from base obtained in previous step. Where no base was found the search for top is masked out
-- Question: shoule we search for next top (next occurence of crossing threshold downwards | 1) or last top (last occurence | 0)?
local topdata = hitool:VerticalHeightLessThanGrid(CATParam, basedata, highlimitdata, thresholddata, 1)

-- Find maximum value of Clear Air Turbulence intensity
local maxdata = hitool:VerticalMaximumGrid(CATParam, basedata, highlimitdata)

-- Convert pressure of base and top to flight level using conversion function based on International Standard Atmonsphere (ISA)
for i=1, #basedata do
  basedata[i] = FlightLevel_(basedata[i] * 100)
  topdata[i] = FlightLevel_(topdata[i] * 100)
end

-- Write output
result:SetParam(param("TI2-BASE-FL"))
result:SetValues(basedata)
luatool:WriteToFile(result)

result:SetParam(param("TI2-TOP-FL"))
result:SetValues(topdata)
luatool:WriteToFile(result)

result:SetParam(param("TI2-MAX-S2"))
result:SetValues(maxdata)
luatool:WriteToFile(result)
