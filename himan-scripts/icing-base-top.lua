--[[
 
icing-base-top 

Find the base and top of icing index defined by some threshold
Produce it in both flight level and hft coordinates
]]

logger:Info("Calculating base and top for icing")
local MISS = missing
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
     zerodata[i] = p[i] / 100
     thresholddata[i] = threshold

     -- FL300 (30000ft=301hPa)
     pFL300data[i] = 301
  end

  local basedata = hitool:VerticalHeightGreaterThanGrid(IceParam, zerodata, pFL300data, thresholddata, 1)

  return basedata
end

function BaseM(threshold)
  -- Find the base of icing define as the lowest height / highest pressure at which icing index crosses the threshold value
  -- returns the height as metric distance above ground

  local zerodata = {}
  local hFL300data = {}
  local thresholddata = {}

  for i = 1, #p do
     zerodata[i] = 0
     thresholddata[i] = threshold

     -- FL300 (9144m)
     hFL300data[i] = 9144
  end

  local basedata = hitool:VerticalHeightGreaterThanGrid(IceParam, zerodata, hFL300data, thresholddata, 1)

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

function TopM(threshold,basedata)
  -- Find the next top above the base of icing define as the height / pressure at which icing index crosses the threshold value
  -- returns the height as metric distance above ground

  local hFL300data = {}
  local thresholddata = {}

  for i = 1, #basedata do
     thresholddata[i] = threshold

     -- FL300 (9144m)
     hFL300data[i] = 9144
  end

  local topdata = hitool:VerticalHeightLessThanGrid(IceParam, basedata, hFL300data, thresholddata, 1)

  return topdata
end

function AddScalar(arr, scalar)
  local ret = {}
  for i=1,#arr do
    ret[i] = arr[i] + scalar
  end
  return ret
end

local baseHPa = BaseHPa(4)
local topHPa = TopHPa(4,AddScalar(baseHPa,-1))

-- Convert top [hPa] to FL
local topFL = {}
local baseFL = {}
for i=1, #topHPa do
  topFL[i] = FlightLevel_(topHPa[i] * 100)
  baseFL[i] = FlightLevel_(baseHPa[i] * 100)
end

hitool:SetHeightUnit(HPParameterUnit.kM)

local baseM = BaseM(4)
local topM = TopM(4,AddScalar(baseM,1))

-- Convert top [M] to hFt
local topHFt = {}
local baseHFt = {}
for i=1, #baseM do
  topHFt[i] = 5 * math.floor(topM[i] / 152.4)
  baseHFt[i] = 5 * math.floor(baseM[i] / 152.4)
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

