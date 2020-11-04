--
-- SmartMet NWC parameters
--
-- Copy certain set of parameters from Smartmet data so that they
-- look like created from NWC (same analysis time etc)
-- 

local MISS = missing
local editor_prod = producer(181, "SMARTMET")

editor_prod:SetCentre(86)
editor_prod:SetProcess(181)

local editor_origintime = raw_time(radon:GetLatestTime(editor_prod, "", 0))
local editor_time = forecast_time(editor_origintime, current_time:GetValidDateTime())

function split (inputstr, sep)
  local t={}
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    table.insert(t, str)
  end
  return t
end

function FetchAndWrite(par)
  logger:Info("Processing param " .. par:GetName())
  local data = luatool:FetchWithProducer(editor_time, current_level, par, current_forecast_type, editor_prod, "")

  if not data then
    logger:Error("Data not found")
    return
  end

  result:SetParam(par)
  result:SetValues(data)
  luatool:WriteToFile(result)
end

local parlist = split(configuration:GetValue("params"), ",")

for k, parname in pairs(parlist) do
  FetchAndWrite(param(parname))
end
