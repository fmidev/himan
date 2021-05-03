--[[
--  Categorize NWCSAF cloud types
--]]

local Missing = missing

function CloudLayers()

  local ct = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("NWCSAF_CLDTYPE-N"), current_forecast_type)
  --local ctqc = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("NWCSAF_CLDTYPE_QC-N"), current_forecast_type)

  if not ct then -- or not ctqc then
    return
  end

  --[[
  CLOUD TYPE

  1 Cloud-free land
  2 Cloud-free sea
  3 Snow over land
  4 Sea ice
  5 Very low clouds
  6 Low clouds
  7 Mid-level clouds
  8 High opaque clouds
  9 Very high opaque clouds
  10 Fractional clouds
  11 High semitransparent thin clouds
  12 High semitransparent meanly thick clouds
  13 High semitransparent thick clouds
  14 High semitransparent above low or medium clouds
  15 High semitransparent above snow/ice

  QUALITY

  1 nodata
  2 internal_consistency
  4 temporal_consistency
  8 good 
  16 questionable 
  24 bad
  32 interpolated

  OUTPUT

  -- 0 No clouds
  -- 1 Low clouds
  -- 2 Middle clouds
  -- 3 High clouds
  --
  -- Most significant cloud is considered to be the lowest cloud, should
  -- many layers exist

  --]]

  local values = {}

  for i = 1, #ct do

    local v = ct[i]
    local nv = Missing
    -- local m = ctml[i]
    local q = ctqc[i]

    if (v <= 4) then
      nv = 0

    -- merge 'very low', 'low', 'fractional' and 'high above low or medium clouds'
    elseif (v == 5 or v == 6 or v == 10 or v == 14) then
      nv = 1

    elseif (v == 7) then
      nv = 2

    -- merge 'very high', 'high' and 'high semitransparents'
    elseif (v == 8 or v == 9 or v == 11 or v == 12 or v == 13 or v == 15) then
      nv = 3
    end

    -- no qc for now
    --if (q == 1 or q == 24) then
    --  nv = Missing
    --end

    values[i] = nv
  end

  result:SetValues(values)
  result:SetParam(param("CLDTYPE-N"))
  luatool:WriteToFile(result)
  luatool:WriteToFile(result)
  luatool:WriteToFile(result)

end

CloudLayers()
