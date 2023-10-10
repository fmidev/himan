-- Toplink products for aviation
-- https://wiki.fmi.fi/display/ASI/TOPLINK
-- aaltom 20150709

logger:Info("Calculating TopLink parameters")
local MISS = missing
local lvl = level(HPLevelType.kHeight, 0)

function Product1()

  local vis = param("VV2-M")
  local rr = param("RRR-KGM2")
  local ptype = param("PRECFORM2-N")
  local wiparam = param("TOPL2-N")

  local visdata = luatool:FetchWithType(current_time, lvl, vis, current_forecast_type)
  local rrdata = luatool:FetchWithType(current_time, lvl, rr, current_forecast_type)
  local ptypedata = luatool:FetchWithType(current_time, lvl, ptype, current_forecast_type)

  if not visdata or not rrdata or not ptypedata then
    logger:Error("Some(or all) of the data is not found")
    return
  end

  local wi = {}

  for i = 1, #visdata do

    local tlindex = 0
    local rrval = rrdata[i]
    local visval = visdata[i]
    local ptypeval = ptypedata[i]
    local sn = 0 -- snow index
    local fz = 0 -- freezing rain index
    local vvi = 0 -- visibility index

    if visval == visval and rrval == rrval then

      if rrval >= 0.1 then

        if ptypeval == 2 or ptypeval == 3 then
          sn = 1
          if rrval >= 2.5 then
            sn = 3
          elseif rrval >= 0.5 then
            sn = 2
          end
        elseif ptypeval == 4 or ptypeval == 5 then
          fz = 2
          if rrval > 0.7 then
            fz = 3
          end
        end

        if visval < 600 then
          vvi = 3
        elseif visval < 1500 then
          vvi = 2
        elseif visval < 5000 then
          vvi = 1
        end

        -- Check snow and vis
        if sn > tlindex then
          tlindex = sn

          if vvi > tlindex then
            tlindex = vvi
          end
        end

        -- Check freezing and vis
        if fz > tlindex then
          tlindex = fz

          if vvi > tlindex then
            tlindex = vvi
          end
        end

      else
        tlindex = 0
      end
    else
      tlindex = MISS
    end

    wi[i] = tlindex

  end

  result:SetParam(wiparam)
  result:SetValues(wi)
  luatool:WriteToFile(result)

end


function Product2()

  local vis = param("VV2-M")
  local rr = param("RRR-KGM2")
  local ptype = param("PRECFORM2-N")
  local ceil = param("CEIL-2-M")
  local wiparam = param("TOPL3-N")

  local visdata = luatool:FetchWithType(current_time, lvl, vis, current_forecast_type)
  local rrdata = luatool:FetchWithType(current_time, lvl, rr, current_forecast_type)
  local ptypedata = luatool:FetchWithType(current_time, lvl, ptype, current_forecast_type)
  local ceildata = luatool:FetchWithType(current_time, lvl, ceil, current_forecast_type)

  if not visdata or not rrdata or not ptypedata or not ceildata then
    logger:Error("Some(or all) of the data is not found")
    return
  end

  local wi = {}

  for i = 1, #visdata do

    local tlindex = 0
    local rrval = rrdata[i]
    local visval = visdata[i]
    local ptypeval = ptypedata[i]
    local ceilval = ceildata[i] * 3.2808
    local sn = 0 -- snow index
    local fz = 0 -- freezing rain index
    local vvi = 0 -- visibility index
    local ci = 0 -- ceiling index

    if visval == visval and rrval == rrval then

      if rrval >= 0.1 then

        if ptypeval == 2 or ptypeval == 3 then
          sn = 1
          if rrval >= 1.5 then
            sn = 3
          elseif rrval >= 0.5 then
            sn = 2
          end
        elseif ptypeval == 4 or ptypeval == 5 then
          fz = 3
        end

        if visval < 2000 then
          vvi = 3
        elseif visval < 5000 then
          vvi = 2
        elseif visval < 8000 then
          vvi = 1
        end

        if ceilval < 600 then
          ci = 3
        elseif ceilval < 1500 then
          ci = 2
        elseif ceilval < 2000 then
          ci = 1
        end

        -- Check snow and vis and ci
        if sn > tlindex then
          tlindex = sn

          if vvi > tlindex then
            tlindex = vvi
          end

          if ci > tlindex then
            tlindex = ci
          end

        end
        -- Check freezing and vis and ci
        if fz > tlindex then
          tlindex = fz

          if vvi > tlindex then
            tlindex = vvi
          end

          if ci > tlindex then
            tlindex = ci
          end

        end

      else
        tlindex = 0
      end
    else
      tlindex = MISS
    end

    wi[i] = tlindex

  end

  result:SetParam(wiparam)
  result:SetValues(wi)
  luatool:WriteToFile(result)

end

Product1()
Product2()
