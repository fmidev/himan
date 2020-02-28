logger:Info("Calculating LVP (Low Visibility Procedures) probability")

local Missing = missingf

local probLVP = {}

local producer = configuration:GetSourceProducer(0)
local ensSize = tonumber(radon:GetProducerMetaData(producer, "ensemble size"))

if not ensSize then
  logger.Error("Ensemble size not found from database for producer " .. producer:GetId())
  return
end

local ens1 = lagged_ensemble(param("VV2-M"), "MEPS_LAGGED_ENSEMBLE")
local ens2 = lagged_ensemble(param("CL-2-FT"), "MEPS_LAGGED_ENSEMBLE")

ens1:SetMaximumMissingForecasts(ensSize)
ens2:SetMaximumMissingForecasts(ensSize)

ens1:Fetch(configuration, current_time, current_level)
ens2:Fetch(configuration, current_time, current_level)

local lagEnsSize = ens1:Size()

ens1:ResetLocation()
ens2:ResetLocation()

local i = 0
while ens1:NextLocation() and ens2:NextLocation() do
  i = i+1

  local vals1 = ens1:Values()
  local vals2 = ens2:Values()

  local numLVP = 0

  probLVP[i] = Missing

  for j = 1, #vals1 do
    local val1 = vals1[j]
    local val2 = vals2[j]
   
    if val1 < 600 or val2 <= 200 then
        numLVP = numLVP + 1
    end
    
             
  end

  probLVP[i] = numLVP / lagEnsSize
  
end

local probParam = param("PROB-LVP-1")
result:SetForecastType(forecast_type(HPForecastType.kStatisticalProcessing))
result:SetParam(probParam)
result:SetValues(probLVP)
luatool:WriteToFile(result)
