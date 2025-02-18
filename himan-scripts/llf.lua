-- Highest LLF (Low Level Forecast) cloud top:
-- - top shown for cloud amount N>55% (at least ~BKN cloud) or TCu/Cb top (if higher)
-- - max value FL125 (i.e. higher cloud tops are ignored in layered clouds)
-- - in hft above surface below FL050 (at 1hft resolution)
-- - in FL above FL050 (at 5hft resolution)
--
-- - Requirements:
--   pressure (hPa) on (low) model levels (OR parameter station pressure)
--   N (cloud fraction %) on model levels
--   par915 (Cb/TCu top FL)
--
-- partio 20121104, original smarttool macro by simo n
-- Tack 20250218, rewrite to fix error due to use of wrong definition of flight level 
--
-- https://wiki.fmi.fi/pages/viewpage.action?pageId=123387126

logger:Info("Calculating low level forecast cloud top height")
local MISS = missing
local NParam = param("N-0TO1")

-- We set the vertical search function to work with pressure based vertical coordinate
-- The reasoning is that the pressures can be directly converted into flight levels (FL)
hitool:SetHeightUnit(HPParameterUnit.kHPa)

function GetBase(lowlimitdata, highlimitdata, thresholddata, pFL125data)
  local basedata = hitool:VerticalHeightGreaterThanGrid(NParam, lowlimitdata, highlimitdata, thresholddata, 1)

  if not basedata then
    NParam = param("N-PRCNT")
    basedata = hitool:VerticalHeightGreaterThanGrid(NParam, lowlimitdata, highlimitdata, thresholddata, 1)
  end

  for i=1,#basedata do
    if basedata[i] < pFL125data[i] then
      basedata[i] = MISS
    end
  end

  return basedata
end

function GetTop(lowlimitdata, highlimitdata, thresholddata, pFL125data)
  local topdata = hitool:VerticalHeightLessThanGrid(NParam, lowlimitdata, highlimitdata, thresholddata, 1)

  for i=1,#topdata do
    topdata[i] = math.max(topdata[i], pFL125data[i])
  end

  return topdata
end

function AddScalar(arr, scalar)
  local ret = {}
  for i=1,#arr do
    ret[i] = arr[i] + scalar
  end
  return ret
end

function Top()

  -- Cloud amount threshold to consider a cloud (base and) top [0..1]
  local threshold = 0.55

  -- Max height to check for cloud top [hPa] (572 ~ FL150)
  local minP = 572

  -- cb/tcu
  local cbtcudata = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("CBTCU-FL"), current_forecast_type)
  local p = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("P-PA"), current_forecast_type)

  if not cbtcudata then
    return
  end

  local pFL050data = {}
  local pFL125data = {}

  local zerodata = {}
  local minPdata = {}
  local thresholddata = {}

  for i = 1, #p do
     zerodata[i] = p[i] / 100
     minPdata[i] = minP
     thresholddata[i] = threshold

     -- FL050 (5000ft=843hPa) height in meters above the surface
     pFL050data[i] = 843

     -- FL125 (12500ft=632hPa) height in meters above the surface
     pFL125data[i] = 632
  end

  local base1data = GetBase(zerodata, minPdata, thresholddata, pFL125data)
  local top1data = GetTop(AddScalar(base1data, -1), minPdata, thresholddata, pFL125data)

  local base2data = GetBase(AddScalar(top1data, -1), minPdata, thresholddata, pFL125data)
  local top2data = GetTop(AddScalar(base2data, -1), minPdata, thresholddata, pFL125data)

  local base3data = GetBase(AddScalar(top2data, -1), minPdata, thresholddata, pFL125data)
  local top3data = GetTop(AddScalar(base3data, -1), minPdata, thresholddata, pFL125data)

  local base4data = GetBase(AddScalar(top3data, -1), minPdata, thresholddata, pFL125data)
  local top4data = GetTop(AddScalar(base4data, -1), minPdata, thresholddata, pFL125data)

  local ret = {}
  local top = {}
  for i=1,#top1data do
    -- Base found, but top extends above FL125 (>maxH in the calculation)
    if base1data[i] < pFL125data[i] and IsMissing(top1data[i]) then
      top1data[i] = pFL125data[i]
    end
    if base2data[i] < pFL125data[i] and IsMissing(top2data[i]) then
      top2data[i] = pFL125data[i]
    end
    if base3data[i] < pFL125data[i] and IsMissing(top3data[i]) then
      top3data[i] = pFL125data[i]
    end
    if base4data[i] < pFL125data[i] and IsMissing(top4data[i]) then
      top4data[i] = pFL125data[i]
    end

    -- highest top
    top[i] = math.min(top1data[i], top2data[i], top3data[i], top4data[i])
  end

  -- Convert top [hPa] to FL
  local topFL = {}
  for i=1, #top do
    topFL[i] = FlightLevel_(top[i] * 100)
  end

  -- Convert top [hPa] to hft
  -- We set the vertical search function to work with height based vertical coordinate
  -- The reasoning is that below transition altitude top is reported in height above ground
  hitool:SetHeightUnit(HPParameterUnit.kM)

  local zerodata = {}
  local maxHdata = {}

  for i = 1, #p do
     zerodata[i] = 0
     maxHdata[i] = 5000
  end

  local topheight = hitool:VerticalHeightGrid(param("P-HPA"), zerodata, maxHdata, top, 1)

  for i=1, #topFL do
    if topFL[i] < 50 then
      ret[i] = math.floor(0.5 + topheight[i] / 30.48)
    else
      ret[i] = topFL[i]
    end
  

    -- Possible TCu/Cb tops (par915 [FL])
    local cbtcu = math.abs(cbtcudata[i])

    if IsMissing(ret[i]) or ret[i] < cbtcu then
      ret[i] = math.min(cbtcu, 125)
    end
  end
  

  result:SetParam(param("LLF-TOP-FL"))
  result:SetValues(ret)
  luatool:WriteToFile(result)

end


Top()
