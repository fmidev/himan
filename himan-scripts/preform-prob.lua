logger:Info("Calculating preform probability")

local Missing = missing

local probRain = {}
local probDrizzle = {}
local probSnow = {}
local probSleet = {}
local probFreezingRain = {}
local probFreezingDrizzle = {}
local probFreezingPrecipitation = {}

local producer = configuration:GetSourceProducer(0)
local ensSize = tonumber(radon:GetProducerMetaData(producer, "ensemble size"))

if not ensSize then
  logger.Error("Ensemble size not found from database for producer " .. producer:GetId())
  return
end

local ens = ensemble(param("PRECFORM2-N"), ensSize)

ens:Fetch(configuration, current_time, current_level)

local ensSize = ens:Size()

ens:ResetLocation()

local i = 0
while ens:NextLocation() do
  i = i+1

  local vals = ens:Values()

  local numRain = 0
  local numFreezingRain = 0
  local numFreezingDrizzle = 0
  local numSnow = 0
  local numSleet = 0
  local numDrizzle = 0
  local numFreezingPrecipitation = 0

  probRain[i] = Missing
  probDrizzle[i] = Missing
  probSnow[i] = Missing
  probSleet[i] = Missing
  probFreezingRain[i] = Missing
  probFreezingDrizzle[i] = Missing
  probFreezingPrecipitation[i] = Missing

  for j = 1, #vals do
    local val = vals[j]

    if val ~= Missing then
      if val == 0 then
        numDrizzle = numDrizzle + 1
      elseif val == 1 then
        numRain = numRain + 1
      elseif val == 2 then
        numSleet = numSleet + 1
      elseif val == 3 then
        numSnow = numSnow + 1
      elseif val == 4 then
        numFreezingDrizzle = numFreezingDrizzle + 1
        numFreezingPrecipitation = numFreezingPrecipitation + 1
      elseif val == 5 then
        numFreezingRain = numFreezingRain + 1
        numFreezingPrecipitation = numFreezingPrecipitation + 1
      end
    end
  end

  probRain[i] = numRain / ensSize
  probSnow[i] = numSnow / ensSize
  probSleet[i] = numSleet / ensSize
  probDrizzle[i] = numDrizzle / ensSize
  probFreezingDrizzle[i] = numFreezingDrizzle / ensSize
  probFreezingRain[i] = numFreezingRain / ensSize
  probFreezingPrecipitation[i] = numFreezingPrecipitation / ensSize
end

local probParam = param("PROB-RAIN")
result:SetParam(probParam)
result:SetValues(probRain)
luatool:WriteToFile(result)

probParam = param("PROB-SNOW")
result:SetParam(probParam)
result:SetValues(probSnow)
luatool:WriteToFile(result)

probParam = param("PROB-SLEET")
result:SetParam(probParam)
result:SetValues(probSleet)
luatool:WriteToFile(result)

probParam = param("PROB-DRIZZLE")
result:SetParam(probParam)
result:SetValues(probDrizzle)
luatool:WriteToFile(result)

probParam = param("PROB-FRDRZZL")
result:SetParam(probParam)
result:SetValues(probFreezingDrizzle)
luatool:WriteToFile(result)

probParam = param("PROB-FRRAIN")
result:SetParam(probParam)
result:SetValues(probFreezingRain)
luatool:WriteToFile(result)

probParam = param("PROB-FRPREC")
result:SetParam(probParam)
result:SetValues(probFreezingPrecipitation)
luatool:WriteToFile(result)
