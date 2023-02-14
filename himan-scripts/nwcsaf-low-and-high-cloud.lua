--[[
// Effective cloudinessin fiksausta
// Leila & Anniina Versio 31/3/22
// Korjaa satelliitin karkean resoluution aiheuttamia aukkoja ylä- ja alapilvialueiden rajoilla
//
// This is also the so called "winter fix"
]]


function Write(res)
  result:SetParam(param("NWCSAF_EFFCLD-0TO1"))
  result:SetValues(res)
  luatool:WriteToFile(result)
end

function LowAndHighCloudGapFix() 

  local effc = luatool:FetchWithType(current_time, current_level, param("NWCSAF_EFFCLD-0TO1"), current_forecast_type)
  local ctt = luatool:FetchWithType(current_time, current_level, param("CTBT-K"), current_forecast_type)

  local mnwc_prod = producer(7, "MNWC")
  mnwc_prod:SetCentre(251)
  mnwc_prod:SetProcess(7)

  local mnwc_origintime = raw_time(radon:GetLatestTime(mnwc_prod, "", 0))
  local mnwc_time = forecast_time(mnwc_origintime, current_time:GetValidDateTime())

  local o = {forecast_time = mnwc_time,
           level = current_level,
           param = param("NH-0TO1"),
           forecast_type = current_forecast_type,
           producer = mnwc_prod,
           geom_name = "",
           read_previous_forecast_if_not_found = true
  }

  local nh = luatool:FetchWithArgs(o)

  o["param"] = param("NM-0TO1")
  local nm = luatool:FetchWithArgs(o)

  o["param"] = param("NL-0TO1")
  local nl = luatool:FetchWithArgs(o)

  if not effc or not ctt or not nh or not nm or not nl then
    logger:Error("Some data not found")
    Write(effc)
    return
  end

  local filter = matrix(3, 3, 1, missing)
  filter:Fill(1/9.)
  local Nmat = matrix(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
  Nmat:SetValues(effc)

  local effc_p1 = Filter2D(Nmat, filter, configuration:GetUseCuda()):GetValues()
  filter = matrix(7, 7, 1, missing)
  filter:Fill(1/49.)
  local effc_p2 = Filter2D(Nmat, filter, configuration:GetUseCuda()):GetValues()

  local res = {}

  for i=1,#effc do
    local effc_ = effc[i]
    local ctt_ = ctt[i]
    local nl_ = nl[i]
    local nm_ = nm[i]
    local nh_ = nh[i]
    local effc_p1_ = effc_p1[i]
    local effc_p2_ = effc_p2[i]

    -- Luodaan alkuun näennäinen cloudmask cloudtop brightness temperaturen avulla.
    -- Jos arvo -> on pilvi
    if IsValid(ctt_) and effc_ == 0.0 then
      effc_ = 0.8
    end

    -- Keskiarvostetaan pienimpiä aukkoja ja reunoja.
    if effc_ <= 0.9 and effc_p1_ > 0.65 then
      effc_ = 0.9
    end
    if effc_ <= 0.9 and effc_p2_ > 0.55 then
      effc_ = 0.9
    end

    -- Vähennetään pilveä, jos vain yläpilveä
    if effc_ > 0.8 and nh_ > 0.5 and nl_ < 0.2 and nm_ < 0.2 then
      effc_ = 0.8
    end

    res[i] = effc_

  end

  return res
end

function MissingCloudFixWithRH(effc)

  local cmqc = luatool:FetchWithType(current_time, current_level, param("NWCSAF_CLDMASK_QC-N"), current_forecast_type)

  local snwc_prod = producer(281, "SMARTMETNWC")
  snwc_prod:SetCentre(86)
  snwc_prod:SetProcess(207)

  local latest_origintime = raw_time(radon:GetLatestTime(snwc_prod, "", 0))
  local ftime = forecast_time(latest_origintime, current_time:GetValidDateTime())
  logger:Info(string.format("SmartMet NWC origintime: %s validtime: %s", ftime:GetOriginDateTime():String("%Y-%m-%d %H:%M:%S"), ftime:GetValidDateTime():String("%Y-%m-%d %H:%M:%S")))

  local o = {forecast_time = ftime,
       level = level(HPLevelType.kHeight, 2),
       param = param("RH-PRCNT"),
       forecast_type = current_forecast_type,
       producer = snwc_prod,
       geom_name = "",
       read_previous_forecast_if_not_found = false,
       time_interpolation = true,
       time_interpolation_search_step = time_duration("00:15:00")
  }

  local snwc_rh = luatool:FetchWithArgs(o)

  local mnwc_prod = producer(7, "MNWC")
  mnwc_prod:SetCentre(251)
  mnwc_prod:SetProcess(7)

  local mnwc_origintime = raw_time(radon:GetLatestTime(mnwc_prod, "", 0))
  ftime = forecast_time(mnwc_origintime, current_time:GetValidDateTime())
  logger:Info(string.format("MNWC origintime: %s validtime: %s", ftime:GetOriginDateTime():String("%Y-%m-%d %H:%M:%S"), ftime:GetValidDateTime():String("%Y-%m-%d %H:%M:%S")))


  o = {forecast_time = ftime,
       level = current_level,
       param = param("NL-0TO1"),
       forecast_type = current_forecast_type,
       producer = mnwc_prod,
       geom_name = "",
       read_previous_forecast_if_not_found = true
  }

  local mnwc_cl = luatool:FetchWithArgs(o)

  if not cmqc or not snwc_rh or not mnwc_cl then
    logger:Warning("Unable to perform RH based fix")
    return effc
  end

  local num_changed = 0
  local changesum = 0

  for i=1,#snwc_rh do
    local old_effc = effc[i]

    if effc[i] < 0.1 and snwc_rh[i] >= 86 and (mnwc_cl[i] > 0.8 or cmqc[i] == 24) then
      effc[i] = 0.6
    end
    if effc[i] <= 0.6 and snwc_rh[i] >= 98 and (mnwc_cl[i] > 0.2 or cmqc[i] == 24) then
      effc[i] = 0.8
    end

    if effc[i] ~= old_effc then
      num_changed = num_changed + 1
      changesum = changesum + (effc[i] - old_effc)
    end
  end
  logger:Info(string.format("Changed %d values (%.1f%% of grid values), average change was %.3f", num_changed, 100*num_changed/#snwc_rh, changesum / num_changed))
  return effc
end

local mon = tonumber(current_time:GetValidDateTime():String("%m"))

-- Fix is valid for winter months September ... March
if mon >= 10 or mon <= 3 then
  effc = LowAndHighCloudGapFix()
  effc = MissingCloudFixWithRH(effc)
  Write(effc)
end
