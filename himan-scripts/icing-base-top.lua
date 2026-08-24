--[[
 
icing-base-top 

Find the base and top of icing index defined by some threshold
Produce it in both flight level and hft coordinates
]]

logger:Info("Calculating base and top for icing")
local MISS = missing
local IceParam = param("ICING-N")

-- We set the vertical search function to work with pressure based vertical coordinate
-- The reasoning is that the search limits are better defined in the pressure domain and
-- the pressures can be directly converted into flight levels (FL)
hitool:SetHeightUnit(HPParameterUnit.kHPa)

-- Highest searched height, 10000 m in the standard atmosphere
local maxP = 264

-- Get surface pressure
local p = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("P-PA"), current_forecast_type)

function BaseHPa(threshold)
  -- Find the base of icing define as the lowest height / highest pressure at which icing index crosses the threshold value
  -- returns the height as pressure based coordinate

  local zerodata = {}
  local maxPdata = {}
  local thresholddata = {}

  for i = 1, #p do
     -- convert surface pressure to hPa
     zerodata[i] = p[i] / 100
     thresholddata[i] = threshold
     maxPdata[i] = maxP
  end

  local basedata = hitool:VerticalHeightGreaterThanGrid(IceParam, zerodata, maxPdata, thresholddata, 1)

  return basedata
end

function TopHPa(threshold,basedata)
  -- Find the next top above the base of icing define as the height / pressure at which icing index crosses the threshold value
  -- returns the height as pressure

  local maxPdata = {}
  local thresholddata = {}

  for i = 1, #basedata do
     thresholddata[i] = threshold
     maxPdata[i] = maxP
  end

  -- Search is started slightly above the base, otherwise the base itself is returned as the top
  local topdata = hitool:VerticalHeightLessThanGrid(IceParam, AddScalar(basedata,-1), maxPdata, thresholddata, 1)

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
local baseHPa = BaseHPa(4 - 0.001) -- icing index >= 4
local topHPa = TopHPa(4,baseHPa) -- icing index < 4

-- Base and top are always given as a pair
for i=1, #baseHPa do
  if IsMissing(baseHPa[i]) then
    topHPa[i] = MISS
  elseif IsMissing(topHPa[i]) then
    -- Icing continues above the searched range
    topHPa[i] = maxP
  end
end

-- Fetch metric heights of base and top to convert them to hFt
local baseM = hitool:VerticalValueGrid(param("HL-M"), baseHPa)
local topM = hitool:VerticalValueGrid(param("HL-M"), topHPa)

-- Convert base and top to FL and hFt
local topFL = {}
local baseFL = {}
local topHFt = {}
local baseHFt = {}
for i=1, #baseHPa do
  topFL[i] = FlightLevel_(topHPa[i] * 100) -- hPa to Pa
  topHFt[i] = math.ceil(topM[i] / 30.48) -- 0.3048 / 100

  -- If height < 15 m, icing reaches the ground (0 m)
  if baseM[i] < 15 then
    baseFL[i] = FlightLevel_(p[i])
    baseHFt[i] = 0
  else
    baseFL[i] = FlightLevel_(baseHPa[i] * 100) -- hPa to Pa
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
