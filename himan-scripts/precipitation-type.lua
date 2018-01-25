local function CheckedFetch(parm_name, lvl, time)
  local p = luatool:FetchWithType(time, lvl, param(parm_name), current_forecast_type)
  if not p then
    msg = string.format("Failed to find parameter '%s'", parm_name)
    logger:Error(msg)
    error("luatool:Fetch failed")
  else
    return p
  end
end

local MISS = missing
local currentProducer = configuration:GetSourceProducer(1)
local currentProducerName = currentProducer.GetName(currentProducer)

msg = string.format("Calculating potential precipitation type for producer: %s", currentProducerName)
logger:Info(msg)

local rh925 = nil
local rh850 = nil
local rh700 = nil

if currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  rh925 = CheckedFetch("RH-0TO1", level(HPLevelType.kPressure, 925), current_time)
  rh850 = CheckedFetch("RH-0TO1", level(HPLevelType.kPressure, 850), current_time)
  rh700 = CheckedFetch("RH-0TO1", level(HPLevelType.kPressure, 700), current_time)
else
  rh925 = CheckedFetch("RH-PRCNT", level(HPLevelType.kPressure, 925), current_time)
  rh850 = CheckedFetch("RH-PRCNT", level(HPLevelType.kPressure, 850), current_time)
  rh700 = CheckedFetch("RH-PRCNT", level(HPLevelType.kPressure, 700), current_time)
end

local pref = CheckedFetch("POTPRECF-N", level(HPLevelType.kHeight, 0), current_time)
local rrr = CheckedFetch("RRR-KGM2", level(HPLevelType.kHeight, 0), current_time)

-- Limit for relative humidity
-- HARMONIE and HIRLAM seem to have PRCNT in the range [0,1] but EC
-- seems to have it in [0,100]
local Limit
if currentProducerName == "ECG" or currentProducerName == "ECGMTA" then
  Limit = 80
elseif currentProducerName == "AROME" or currentProducerName == "AROMTA" or
  currentProducerName == "HL2" or currentProducerName == "HL2MTA" or
  currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  Limit = 0.8
end

local pret = {}
local potpret = {}

for i=1, #rh700 do
  local _rh925 = rh925[i]
  local _rh850 = rh850[i]
  local _rh700 = rh700[i]
  local _pref = pref[i]
  local _rrr = rrr[i]

  potpret[i] = 2 -- initially 2
  pret[i] = MISS

  if _rh700 > Limit and _rh850 > Limit and _rh925 > Limit then
    potpret[i] = 1
  end

  if _pref == 0 or _pref == 4 or _pref == 5 then
    potpret[i] = 1
  end

  if _rrr > 0 then
    pret[i] = potpret[i]
  end

end

result:SetParam(param("PRECTYPE-N"))
result:SetValues(pret)

result:SetParam(param("POTPRECT-N"))
result:SetValues(potpret)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)
