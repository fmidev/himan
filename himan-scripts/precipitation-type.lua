local function CheckedFetch(parm_name, lvl, time)
  local p = luatool:Fetch(time, lvl, param(parm_name), current_forecast_type)
  if not p then
    msg = string.format("Failed to find parameter '%s'", parm_name)
    logger:Error(msg)
  else
    return p
  end
end

local function Interpolate(parm_name, height)

  hitool:SetHeightUnit(HPParameterUnit.kHPa)
  local data = hitool:VerticalValue(param(parm_name), height)

  if not data then
    error("Data not found")
  end
 
  return data
end

local MISS = missing
local currentProducer = configuration:GetSourceProducer(1)
local currentProducerName = currentProducer.GetName(currentProducer)

msg = string.format("Calculating potential precipitation type for producer: %s", currentProducerName)
logger:Info(msg)

local rh925 = nil
local rh850 = nil
local rh700 = nil

local RHParam = "RH-PRCNT"

if currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" or currentProducerName == "HL2MTA"
  or currentProducerName == "MNWCMTA" then
  RHParam = "RH-0TO1"
end

local rh925 = CheckedFetch(RHParam, level(HPLevelType.kPressure, 925), current_time)
local rh850 = CheckedFetch(RHParam, level(HPLevelType.kPressure, 850), current_time)
local rh700 = CheckedFetch(RHParam, level(HPLevelType.kPressure, 700), current_time)

if not rh925 or not rh850 or not rh700 then
  logger:Info("Trying to interpolate from model levels")
  rh925 = Interpolate(RHParam, 925)
  rh850 = Interpolate(RHParam, 850)
  rh700 = Interpolate(RHParam, 700)
end

local pref = CheckedFetch("POTPRECF-N", level(HPLevelType.kHeight, 0), current_time)
local rrr = CheckedFetch("RRR-KGM2", level(HPLevelType.kHeight, 0), current_time)

if not pref or not rrr or not rh925 or not rh850 or not rh700 then
  error("Data not found")
end

-- Limit for relative humidity
-- HARMONIE and HIRLAM seem to have PRCNT in the range [0,1] but EC
-- seems to have it in [0,100]

local Limit = 80

if RHParam == "RH-0TO1" then
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

luatool:WriteToFile(result)

result:SetParam(param("POTPRECT-N"))
result:SetValues(potpret)

luatool:WriteToFile(result)
