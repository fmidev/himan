-- Cloud ceiling (number 2) code
-- https:--wiki.fmi.fi/display/PROJEKTIT/Ceiling2_ft, smarttool script including comments by Simo Neiglick
-- junila 9/2016 - 10/2016 - 2/2017 - 6/2017
-- 8.6.2017 v1.5 SN
-- - st cloud layer searches now use VERTZ_AVG (instead of VERTZ_MAX)
-- - when there's stratus only on lowest model level, search (other) stratus above it starting from 25m (previously 15m)
-- - the two changes above have no effect in SmartTool, but HIMAN uses vertical interpolation thus not raising ceiling "enough"
-- - (freezing) drizzle no longer prevents raising low ceiling in precipitation

logger:Info("Calculating Cloud ceiling number 2 in meters")

local MISS = missing

-- Max height to check for cloud base [m]
local maxH = 15000

-- Threshold for N (cloud amount) to calculate base [%]
local Nthreshold = 0.55

function GetClouds(N)
  -- 1st model level: Hirlam ~12m, EC ~10m
  -- 2nd model level: Hirlam ~37m, EC ~31m
  -- 3rd model level: Hirlam ~65m, EC ~53m
  -- 4th model level: Hirlam ~87m, EC ~76m
  -- 5th model level: Hirlam ~112m, EC ~102m
  -- 6th model level: Hirlam ~141m, EC ~130m

  -- Based on model level heights, calculate low stratus cloud layers with ~1 model level in each:

  -- 0-15m (~0-50ft), i.e. ~lowest model level
  -- 15-45m (~50-150ft)
  -- 45-70m (~150-230ft)
  -- 70-95m (~230-310ft)
  -- 95-120m (~310-400ft)
  -- 120-150m (~400-500ft)

  -- Low stratus cloud layers below 150m (500ft)

  -- Stratus clouds below 15m (~0-50ft)
  local N15 = hitool:VerticalAverage(N, 0, 15)

  if not N15 then
    return nil, nil, nil, nil, nil, nil
  end

  -- Stratus between 15-45m (~50-150ft)
  local N45 = hitool:VerticalAverage(N, 15, 45)

  -- Stratus between 45-70m (~150-230ft)
  local N70 = hitool:VerticalAverage(N, 45, 70)

  -- Stratus between 70-95m (~230-310ft)
  local N95 = hitool:VerticalAverage(N, 70, 95)

  -- Stratus between 95-120m (~310-400ft)
  local N120 = hitool:VerticalAverage(N, 95, 120)

  -- Stratus between 120-150m (~400-500ft)
  local N150 = hitool:VerticalAverage(N, 120, 150)

  return N15, N45, N70, N95, N120, N150
end

local N = param("N-0TO1")
local N15, N45, N70, N95, N120, N150 = GetClouds(N)

if not N15 or not N45 or not N70 or not N95 or not N120 or not N150 then
  N = param("N-PRCNT")
  N15, N45, N70, N95, N120, N150 = GetClouds(N)
  Nthreshold = Nthreshold * 100
end

if not N15 or not N45 or not N70 or not N95 or not N120 or not N150 then
  return
end

local CEIL_150 = hitool:VerticalHeightGreaterThan(N, 150, maxH, Nthreshold, 1)
local CEIL_120 = hitool:VerticalHeightGreaterThan(N, 120, maxH, Nthreshold, 1)
local CEIL_95 = hitool:VerticalHeightGreaterThan(N, 95, maxH, Nthreshold, 1)
local CEIL_70 = hitool:VerticalHeightGreaterThan(N, 70, maxH, Nthreshold, 1)
local CEIL_45 = hitool:VerticalHeightGreaterThan(N, 45, maxH, Nthreshold, 1)
local CEIL_25 = hitool:VerticalHeightGreaterThan(N, 25, maxH, Nthreshold, 1)
local CEIL_0 = hitool:VerticalHeightGreaterThan(N, 0, maxH, Nthreshold, 1)

if not CEIL_150 or not CEIL_120 or not CEIL_95 or not CEIL_70 or not CEIL_45 or not CEIL_25 or not CEIL_0 then
  return
end

local l = level(HPLevelType.kHeight, 0)

local rr = param("RRR-KGM2")  --rain over the last hour, same parameter name for EC and Hirlam

local PRECR = luatool:Fetch(current_time, l, rr, current_forecast_type)

if not PRECR then

  PRECR = luatool:Fetch(current_time, l, param("RR-1-MM"), current_forecast_type ) -- rain over one hour, no matter

  if not PRECR then
    return
  end

end

local ceiling = {}

for  i=1, #N15 do
  local ceil = MISS

  local n150 = N150[i]
  local n120 = N120[i]
  local n95 = N95[i]
  local n70 = N70[i]
  local n45 = N45[i]
  local n15 = N15[i]

  local ceil_150 = CEIL_150[i]
  local ceil_120 = CEIL_120[i]
  local ceil_95 = CEIL_95[i]
  local ceil_70 = CEIL_70[i]
  local ceil_45 = CEIL_45[i]
  local ceil_25 = CEIL_25[i]
  local ceil_0 = CEIL_0[i]

  local precr = PRECR[i]

  -- Discard lowest model level cloudiness (below 15m~50ft), if there're less than bkn clouds directly above it

  if n15>Nthreshold and n45<Nthreshold then
    ceil = ceil_25
  else
    ceil = ceil_0
  end

  -- Raise very low ceiling in precipitation

  if precr>0 and ceil<150 then
    -- Layer 120-150m (400-500ft) raised to at least 150m (500ft)

    if n150>Nthreshold then
      ceil = ceil_150
    end

    -- Layer 95-120m (310-400ft) raised to at least 120m (400ft)
    if n120>Nthreshold then
      ceil = ceil_120
    end

    -- Layer 70-95m (230-310ft) raised to at least 95m (310ft)
    if n95>Nthreshold then
      ceil = ceil_95
    end

    -- Layer 45-70m (150-230ft) raised to at least 70m (230ft)
    if n70>Nthreshold then
      ceil = ceil_70
    end

    -- Layer 15-45m (50-150ft) raised to at least 45m (150ft)
    if n45>Nthreshold then
      ceil = ceil_45
    end

    -- Layer 0-15m (0-50ft) raised to at least 45m (150ft)
    if n15>Nthreshold then
      ceil = ceil_45
    end
  end

  ceiling[i] = ceil

end

result:SetParam(param("CEIL-2-M"))
result:SetValues(ceiling)

luatool:WriteToFile(result)
