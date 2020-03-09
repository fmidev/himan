--[[

snow percentage from ECMWF EPS

]]

local kFloatMissing = missing
local currentProducer = configuration:GetSourceProducer(1)
local currentProducerName = currentProducer.GetName(currentProducer)

msg = string.format("Calculating snow percentage for producer: %s", currentProducerName)
logger:Info(msg)

local ground_level = level(HPLevelType.kGround, 0)

local PrecAv = luatool:Fetch(current_time, ground_level, param("F50-RR-6-MM"))
local SnowAv = luatool:Fetch(current_time, ground_level, param("F50-SN-6-MM"))

if not PrecAv or not SnowAv then
  return
end

local Prcnt = {}

for i=1, #PrecAv do
  local _p = PrecAv[i]
  local _s = SnowAv[i]

  Prcnt[i] = Missing

  if SnowAv[i] == 0.0 or PrecAv[i] == 0.0 then
    Prcnt[i] = 0.0
  elseif SnowAv[i] > PrecAv[i] then
    Prcnt[i] = 100.0
  else
    Prcnt[i] = 100.0 * (SnowAv[i] / PrecAv[i])
  end
end

result:SetParam(param("FSNO-PRCNT"))
result:SetValues(Prcnt)

logger:Info("Writing result data to file")
luatool:WriteToFile(result)
