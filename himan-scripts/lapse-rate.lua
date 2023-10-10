-- Calculating lapse rate K/km from hybrid level information.
--
-- Original algorithm by Anna-Kaisa Sarkanen & Juha Kilpinen.
--
-- Comments in Finnish are copied from the original version.
--
-- STU-2721

logger:Info("Calculating lapse rate")

local Temp = param("T-K")

-- Muutostarkastelun korkeus [m] mallin maanpinnasta lukien
local Layer = 150

-- Get number of lowest hybrid level
local Producer = configuration:GetTargetProducer()

local LowestNumber = tonumber(radon:GetProducerMetaData(Producer, "last hybrid level number"))
local LowestLevel = level(HPLevelType.kHybrid, LowestNumber)

local T_top = hitool:VerticalValue(Temp, Layer)

if not T_top then
  return
end

local T_bot = luatool:Fetch(current_time, LowestLevel, Temp, current_forecast_type)

if not T_bot then
  return
end

local LapseRate = {}

for i=1,#T_top do
  local top = T_top[i]
  local bottom = T_bot[i]

  LapseRate[i] = 1000 * (top - bottom) / Layer

end

result:SetParam(param("LR-KM"))
result:SetValues(LapseRate)
luatool:WriteToFile(result)
