--[[
// Effective cloudinessin fiksausta
// Leila&Anniina versio 09/06/22
// Korjaa NWCSAF effective cloudiness pilvisyyttä huomioiden erityisesti:
// 1: Talvikaudelle ala/keskipilven raja-alueille muodostuvat virheelliset selkeät alueet
// --> hyödynnetään Cloudtop brightness temperature-arvoa "pilvimaskina"
// 2: pienet selkeät alueet ns. "kohinaa" tms.
// --> tutkitaan onko ympäröivät hilapisteet pilvisiä ja lisätään pilveä jos näin on
// 3:  konvektion aiheuttamat pilvet on 100% vaikka pitäisi olla vähemmän
// --> ECWMF mallin lyhytaaltosäteilyn voimakkuuden perusteella vähentää pilvisyyttä, jos konvektiota
// --> Lasketaan SWR max ("clearsky") arvon avulla "SWR kerroin" ja korjaus huomioidaan vain päviällä
]]

local effc = luatool:FetchWithType(current_time, current_level, param("NWCSAF_EFFCLD-0TO1"), current_forecast_type)
local top = luatool:FetchWithType(current_time, current_level, param("CTBT-K"), current_forecast_type)

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

local ecg_prod = producer(240, "ECGMTA")
ecg_prod:SetCentre(86)
ecg_prod:SetProcess(240)

local ecg_origintime = raw_time(radon:GetLatestTime(ecg_prod, "", 0))
local ecg_validtime = current_time:GetValidDateTime()
local minutes = ecg_validtime:String("%M")
ecg_validtime:Adjust(HPTimeResolution.kMinuteResolution,-minutes)

local ecg_time = forecast_time(ecg_origintime, ecg_validtime)

o = {forecast_time = ecg_time,
     level = current_level,
     param = param("RADGLO-WM2"),
     forecast_type = current_forecast_type,
     producer = ecg_prod,
     geom_name = "",
     read_previous_forecast_if_not_found = true
}

local swr = luatool:FetchWithArgs(o)

o["param"] = param("RADGLOC-WM2")
local swrc = luatool:FetchWithArgs(o)

if not effc or not top or not nh or not nm or not nl or not swr or not swrc then
  logger:Error("Some data not found")
  return
end

local filter = matrix(3, 3, 1, missing)
filter:Fill(1/9)
local Nmat = matrix(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
Nmat:SetValues(effc)

local effc_p1 = Filter2D(Nmat, filter, configuration:GetUseCuda()):GetValues()
local filter = matrix(5, 5, 1, missing)
filter:Fill(1/25)
local effc_p2 = Filter2D(Nmat, filter, configuration:GetUseCuda()):GetValues()

local morning = 6 -- utc
local evening = 18 -- utc
local hour = tonumber(current_time:GetValidDateTime():String("%H"))

local is_day = false

if hour >= morning and hour < evening then
  is_day = true
end

local res = {}

for i=1,#effc do
  local effc_ = effc[i]
  local top_ = top[i]
  local nl_ = nl[i]
  local nm_ = nm[i]
  local nh_ = nh[i]
  local swr_ = swr[i]
  local swrc_ = swrc[i]
  local effc_p1_ = effc_p1[i]
  local effc_p2_ = effc_p2[i]

  -- Luodaan alkuun näennäinen cloudmask cloudtop brightness temperaturen (CBT) avulla.
  -- Jos CBT:llä on arvo -> on pilvi 80%
  if not IsMissing(top_) and effc_ < 0.8 then
    effc_ = 0.8
  end
 
  -- Vähennetään pilveä jos pääosin yläpilveä
  if effc_ > 0.4 and nh_ > 0.4 and nl_ < 0.2 and nm_ < 0.2 then
    effc_ = 0.4
  end
 
  -- N=100% kumpupilvitilanteiden pilvisyyden vähennys-testaus
  -- Lasketaan pilvettömän taivaan max lyhytaaltosäteilyn ja EC:n lyhytaaltosäteilynsuhde
  local swr_rela = swr_ / swrc_

  if is_day then 
    if swr_rela >= 0.65 and effc_ > 0.8 then
      effc_ = 0.8
    end
 
    if swr_rela >= 0.75 and effc_ > 0.5 then
      effc_ = 0.5
    end
  end

  -- Keskiarvoistetaan mahdolliset pienet väärät aukot pois
  if effc_ <= 0.8 and effc_ > 0.5 and effc_p2_ > 0.5 then
    effc_ = 0.8
  end

  if effc_ <= 0.7 and effc_ > 0.4 and effc_p1_ > 0.4 then
    effc_ = 0.7
  end

  res[i] = effc_

end

result:SetParam(param("NWCSAF_EFFCLD-0TO1"))
result:SetValues(res)
luatool:WriteToFile(result)
