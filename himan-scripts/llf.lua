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
--
-- https://wiki.fmi.fi/pages/viewpage.action?pageId=123387126

logger:Info("Calculating low level forecast cloud top height")
local MISS = missing

function GetBase(lowlimitdata, highlimitdata, thresholddata, zFL125data)
  local basedata = hitool:VerticalHeightGreaterThanGrid(param("N-0TO1"), lowlimitdata, highlimitdata, thresholddata, 1)

  for i=1,#basedata do
    if basedata[i] > zFL125data[i] then
      basedata[i] = MISS
    end
  end

  return basedata
end

function GetTop(lowlimitdata, highlimitdata, thresholddata, zFL125data)
  local topdata = hitool:VerticalHeightLessThanGrid(param("N-0TO1"), lowlimitdata, highlimitdata, thresholddata, 1)

  for i=1,#topdata do
    topdata[i] = math.min(topdata[i], zFL125data[i])
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

  -- Max height to check for cloud top [m] (4500m ~ FL150)
  local maxH = 4500

  -- station pressure
  local qfedata = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("P-PA"), current_forecast_type)

  -- cb/tcu
  local cbtcudata = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("CBTCU-FL"), current_forecast_type)

  if not qfedata or not cbtcudata then
    return
  end

  local sfcFLmdata = {}
  local zFL050data = {}
  local zFL125data = {}

  local zerodata = {}
  local maxHdata = {}
  local thresholddata = {}

  for i = 1, #qfedata do
     zerodata[i] = 0
     maxHdata[i] = maxH
     thresholddata[i] = threshold

     -- surface light level in meters (can be negative)
     -- use pressure in hPa
     sfcFLmdata[i] = (3.7314 - (0.01 * qfedata[i]) ^ (1.0 / 5.2559)) / 0.0000256 * 0.3048

     -- FL050 (5000ft=1524m) height in meters above the surface
     zFL050data[i] = 1524 - sfcFLmdata[i]

     -- FL125 (12500ft=3810m) height in meters above the surface
     zFL125data[i] = 3810 - sfcFLmdata[i]
  end

  local base1data = GetBase(zerodata, maxHdata, thresholddata, zFL125data)
  local top1data = GetTop(AddScalar(base1data, 1), maxHdata, thresholddata, zFL125data)

  local base2data = GetBase(AddScalar(top1data, 1), maxHdata, thresholddata, zFL125data)
  local top2data = GetTop(AddScalar(base2data, 1), maxHdata, thresholddata, zFL125data)

  local base3data = GetBase(AddScalar(top2data, 1), maxHdata, thresholddata, zFL125data)
  local top3data = GetTop(AddScalar(base3data, 1), maxHdata, thresholddata, zFL125data)

  local base4data = GetBase(AddScalar(top3data, 1), maxHdata, thresholddata, zFL125data)
  local top4data = GetTop(AddScalar(base4data, 1), maxHdata, thresholddata, zFL125data)

  local ret = {}

  for i=1,#top1data do
    -- Base found, but top extends above FL125 (>maxH in the calculation)
    if base1data[i] < zFL125data[i] and IsMissing(top1data[i]) then
      top1data[i] = zFL125data[i]
    end
    if base2data[i] < zFL125data[i] and IsMissing(top2data[i]) then
      top2data[i] = zFL125data[i]
    end
    if base3data[i] < zFL125data[i] and IsMissing(top3data[i]) then
      top3data[i] = zFL125data[i]
    end
    if base4data[i] < zFL125data[i] and IsMissing(top4data[i]) then
      top4data[i] = zFL125data[i]
    end

    -- highest top
    local top = math.max(top1data[i], top2data[i], top3data[i], top4data[i])
    local topFLm = top + sfcFLmdata[i]

    -- Convert top [m] to hft/FL, rounding based on altitude (1hft = 30.48m)

    if top < zFL050data[i] then
      top = math.floor(0.5 + top / 30.48)
    elseif top >= zFL050data[i] then
      top = math.floor(0.5 + topFLm / 30.48 / 5) * 5
    end

    -- Possible TCu/Cb tops (par915 [FL])

    local cbtcu = math.abs(cbtcudata[i])

    if IsMissing(top) or top < cbtcu then
      top = math.min(cbtcu, 125)
    end

    ret[i] = top

  end

  result:SetParam(param("LLF-TOP-FL"))
  result:SetValues(ret)
  luatool:WriteToFile(result)

end


Top()
