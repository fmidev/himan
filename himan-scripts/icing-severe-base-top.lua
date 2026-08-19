--[[
 
Severe icing computation for ILME 

Find the base and top of severe icing index defined by some threshold
Produce it in both flight level and hft coordinates
]]

logger:Info("Calculating base and top for severe icing")

local IceParam = param("ICING-N")

-- We set the vertical search function to work with height based vertical coordinate
-- The reasoning is that the search range and the freezing rain/drizzle limit are given in meters
hitool:SetHeightUnit(HPParameterUnit.kM)

-- Highest searched height
local maxH = 10000

-- Get surface pressure
local p = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("P-PA"), current_forecast_type)

function BaseHeight(threshold)
  -- Find the base of icing define as the lowest height at which icing index crosses the threshold value
  -- returns the height in meters

  local zerodata = {}
  local maxHdata = {}
  local thresholddata = {}

  for i = 1, #p do
     zerodata[i] = 0
     thresholddata[i] = threshold
     maxHdata[i] = maxH
  end

  local basedata = hitool:VerticalHeightGreaterThanGrid(IceParam, zerodata, maxHdata, thresholddata, 1)

  return basedata
end

function TopHeight(threshold,basedata)
  -- Find the next top above the base of icing define as the height at which icing index crosses the threshold value
  -- returns the height in meters

  local maxHdata = {}
  local thresholddata = {}

  for i = 1, #basedata do
     thresholddata[i] = threshold
     maxHdata[i] = maxH
  end

  -- Search is started slightly above the base, otherwise the base itself is returned as the top
  local topdata = hitool:VerticalHeightLessThanGrid(IceParam, AddScalar(basedata,1), maxHdata, thresholddata, 1)

  return topdata
end

function AddScalar(arr, scalar)
  local ret = {}
  for i=1,#arr do
    ret[i] = arr[i] + scalar
  end
  return ret
end

-- Base and top are the heights where the interpolated icing index reaches and leaves
-- the value 7, so that the layer is not padded towards the neighbouring model levels.
-- The base threshold is nudged just below 7 because the search is done with a strict
-- comparison (>) and the index is commonly exactly 7 in freezing rain and drizzle.
local baseM = BaseHeight(7 - 0.001) -- severe icing index >= 7
local topM = TopHeight(7,baseM) -- severe icing index < 7

-- Base and top are always given as a pair
for i=1, #baseM do
  if IsMissing(baseM[i]) then
    topM[i] = missing
  elseif IsMissing(topM[i]) then
    -- Icing continues above the searched range
    topM[i] = maxH
  end
end

-- Fetch pressures of base and top to convert them to FL
local baseP = hitool:VerticalValueGrid(param("P-HPA"), baseM)
local topP = hitool:VerticalValueGrid(param("P-HPA"), topM)

-- Convert base and top to FL and hFt
local topFL = {}
local baseFL = {}
local topHFt = {}
local baseHFt = {}
for i=1, #baseM do
  if IsMissing(baseM[i]) then
    topFL[i] = missing
    baseFL[i] = missing
    topHFt[i] = missing
    baseHFt[i] = missing
  else
    topFL[i] = FlightLevel_(topP[i] * 100)
    topHFt[i] = math.floor(topM[i] / 30.48)

    -- If height < 15 m, icing should be classified as freezing rain/drizzle (0 m)
    if baseM[i] < 15 then
      baseFL[i] = FlightLevel_(p[i])
      baseHFt[i] = 0
    else
      baseFL[i] = FlightLevel_(baseP[i] * 100)
      baseHFt[i] = math.floor(baseM[i] / 30.48)
    end
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

