--
-- SmartMet NWC parameters
--
-- Create maximum wind speed and maximum wind gust parameters to SNWC.
-- Source data can be only SNWC (current preop) or a combination of SNWC
-- and smartmet (current prod).
-- 

local FFG = {}

local use_smartmet = true

if configuration:GetValue("use_smartmet") == "false" then
  use_smartmet = false
end

if use_smartmet == true then
  local editor_prod = producer(181, "SMARTMET")
  editor_prod:SetCentre(86)
  editor_prod:SetProcess(181)

  local editor_origintime = raw_time(radon:GetLatestTime(editor_prod, "", 0))
  local editor_time = forecast_time(editor_origintime, current_time:GetValidDateTime())

  FFG = luatool:FetchWithProducer(editor_time, current_level, param("FFG-MS"), current_forecast_type, editor_prod, "")

else
  FFG = luatool:FetchWithType(current_time, current_level, param("FFG-MS"), current_forecast_type)

end

-- maximum wind hourly factors are:
--   0    1    2    3    4    5    6    7    8    9
-- 1.03 1.03 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.10

local maxWindFactor = 1.03
local step = current_time:GetStep():Hours()

if step > 2 then
  maxWindFactor = 1.03 + 0.01 * (step - 2)
end

local FF = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 10), param("FF-MS"), current_forecast_type)

if not FF or not FFG then
  return
end

local _FF = {}
local _FFG = {}

for i=1, #FF do
  local ff = FF[i] * maxWindFactor
  local ffg = FFG[i]

  if ff * 1.135 > ffg then
    ffg = ff * 1.16
  end

  _FF[i] = ff
  _FFG[i] = ffg
end

result:SetParam(param("FFG-MS"))
result:SetValues(_FFG)
luatool:WriteToFile(result)
result:SetParam(param("FF-MS"))
result:SetValues(_FF)
luatool:WriteToFile(result)
