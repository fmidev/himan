--[[
 
Severe icing computation for ILME 

Find the base and top of severe icing index defined by some threshold
Produce it in both flight level and hft coordinates
]]

logger:Info("Calculating base and top for severe icing")
local utils = require("utils")

local IceParam = param("ICING-N")

-- We set the vertical search function to work with pressure based vertical coordinate
-- The reasoning is that the pressures can be directly converted into flight levels (FL)
hitool:SetHeightUnit(HPParameterUnit.kHPa)

-- Get surface pressure
local p = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("P-PA"), current_forecast_type)

function BaseHPa(threshold)
  -- Find the base of icing define as the lowest height / highest pressure at which icing index crosses the threshold value
  -- returns the height as pressure based coordinate

  local zerodata = {}
  local pFL300data = {}
  local thresholddata = {}

  for i = 1, #p do
     -- convert surface pressure to hPa
     zerodata[i] = p[i] / 100
     thresholddata[i] = threshold

     -- FL300 (30000ft=301hPa)
     pFL300data[i] = 301
  end

  local basedata = hitool:VerticalHeightGreaterThanGrid(IceParam, zerodata, pFL300data, thresholddata, 1)

  return basedata
end

function TopHPa(threshold,basedata)
  -- Find the next top above the base of icing define as the height / pressure at which icing index crosses the threshold value
  -- returns the height as pressure

  local pFL300data = {}
  local thresholddata = {}

  for i = 1, #basedata do
     thresholddata[i] = threshold

     -- FL300 (30000ft=301hPa)
     pFL300data[i] = 301
  end

  local topdata = hitool:VerticalHeightLessThanGrid(IceParam, basedata, pFL300data, thresholddata, 1)

  return topdata
end

function AddScalar(arr, scalar)
  local ret = {}
  for i=1,#arr do
    ret[i] = arr[i] + scalar
  end
  return ret
end

local baseHPa = BaseHPa(6) -- severe icing index >= 7
local topHPa = TopHPa(7,AddScalar(baseHPa,-1)) -- severe icing index < 7

-- Convert pressure to FL
local topFL = {}
local baseFL = {}
for i=1, #topHPa do
  topFL[i] = FlightLevel_(topHPa[i] * 100)
  baseFL[i] = FlightLevel_(baseHPa[i] * 100)
end

-- Fetch metric heights of base and top to convert them to hFt
local baseM = hitool:VerticalValueGrid(param("HL-M"), baseHPa)
local topM = hitool:VerticalValueGrid(param("HL-M"), topHPa)

-- Convert top [M] to hFt
local topHFt = {}
local baseHFt = {}
for i=1, #baseM do
  -- If height < 15 m, icing should be classified as freezing rain/drizzle (0 m)
  if baseM[i] < 15 then
    topHFt[i] = 0
    baseHFt[i] = 0
  else
    topHFt[i] = utils.round(topM[i] / 0.3048 / 100)
    baseHFt[i] = utils.round(baseM[i] / 0.3048 / 100)
  end
end


result:SetParam(param("ICING-SEV-TOP-FL"))
result:SetValues(topFL)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

result:SetParam(param("ICING-SEV-BASE-FL"))
result:SetValues(baseFL)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

result:SetParam(param("ICING-SEV-TOP-FT"))
result:SetValues(topHFt)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

result:SetParam(param("ICING-SEV-BASE-FT"))
result:SetValues(baseHFt)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

