--[[
// Effective cloudinessin fiksausta
// Leila & Anniina Versio 31/3/22
// Korjaa satelliitin karkean resoluution aiheuttamia aukkoja ylä- ja alapilvialueiden rajoilla
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

  local morning = 7 -- utc, cumulus process is starting
  local evening = 17 -- utc, cumulus process is ending
  local hour = tonumber(current_time:GetValidDateTime():String("%H"))

  local is_day = false

  if hour >= morning and hour < evening then
    is_day = true
  end

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
    if IsValid(ctt_) then
      effc_ = math.max(effc_, 0.8)
    end

    -- Keskiarvostetaan pienimpiä aukkoja ja reunoja.
    if effc_ <= 0.9 and effc_p1_ > 0.65 then
      effc_ = 0.8
    end
    if effc_ <= 0.9 and effc_p2_ > 0.55 then
      effc_ = 0.8
    end

    -- Vähennetään pilveä, jos vain ylä
    if effc_ > 0.4 and nh_ > 0.5 and nl_ < 0.2 and nm_ < 0.2 then
      effc_ = 0.4
    end

    res[i] = effc_

  end

  Write(res)
end

local mon = tonumber(current_time:GetValidDateTime():String("%m"))

-- Fix is valid for winter months September ... March
if mon >= 9 or mon <= 3 then
  LowAndHighCloudGapFix()
end
