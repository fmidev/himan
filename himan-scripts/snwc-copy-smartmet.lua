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

logger:Info(string.format("Latest Smartmet data origin time is: %s", editor_origintime:String("%Y-%m-%d %H:%M:%S")))

function split (inputstr, sep)
  local t={}
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    table.insert(t, str)
  end
  return t
end

function FetchAndWrite(par)
  logger:Info("Processing param " .. par:GetName())

  local data = luatool:FetchWithArgs{
    forecast_time=editor_time,
    level=current_level,
    param=par,
    forecast_type=current_forecast_type,
    producer=editor_prod,
    geom_name="",
    read_previous_forecast_if_not_found=true
  }

  if not data then
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
