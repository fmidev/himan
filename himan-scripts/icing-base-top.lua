--[[
 
icing-base-top 

Find the base and top of icing index defined by some threshold
Produce it in both flight level and hft coordinates
]]

logger:Info("Calculating base and top for icing")
local MISS = missing
local IceParam = param("ICING-N")

-- The vertical search is done in the default height based coordinate (HL-M),
-- so the search range and the ground level limit are given in meters

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
-- the value 4, so that the layer is not padded towards the neighbouring model levels.
-- The base threshold is nudged just below 4 because the search is done with a strict
-- comparison (>).
local baseM = BaseHeight(4 - 0.001) -- icing index >= 4
local topM = TopHeight(4,baseM) -- icing index < 4

-- Base and top are always given as a pair
for i=1, #baseM do
  if IsMissing(baseM[i]) then
    topM[i] = MISS
  elseif IsMissing(topM[i]) then
    -- Icing continues above the searched range
    topM[i] = maxH
  end
end

-- Fetch pressures [hPa] of base and top to convert them to FL
local baseP = hitool:VerticalValueGrid(param("P-HPA"), baseM)
local topP = hitool:VerticalValueGrid(param("P-HPA"), topM)

-- Convert base and top to FL and hFt
local topFL = {}
local baseFL = {}
local topHFt = {}
local baseHFt = {}
for i=1, #baseM do
  -- Top is rounded up and base down so that the rounding does not narrow the layer
  topFL[i] = FlightLevel_(topP[i] * 100) -- hPa to Pa
  topHFt[i] = math.ceil(topM[i] / 30.48) -- 0.3048 / 100

  -- If height < 15 m, icing reaches the ground (0 m)
  if baseM[i] < 15 then
    baseFL[i] = 0
    baseHFt[i] = 0
  else
    baseFL[i] = FlightLevel_(baseP[i] * 100) -- hPa to Pa
    baseHFt[i] = math.floor(baseM[i] / 30.48) -- 0.3048 / 100
  end
end

result:SetParam(param("ICING-TOP-FL"))
result:SetValues(topFL)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

result:SetParam(param("ICING-BASE-FL"))
result:SetValues(baseFL)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

result:SetParam(param("ICING-TOP-FT"))
result:SetValues(topHFt)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

result:SetParam(param("ICING-BASE-FT"))
result:SetValues(baseHFt)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)

