--- ensemble member count

local currentProducer = configuration:GetSourceProducer(0)
local currentProducerName = currentProducer.GetName(currentProducer)

msg = string.format("Calculating ensemble member count for for producer: %s", currentProducerName)
logger:Info(msg)

local params = {}
params["T-K"] = { current_level, 0, 0 }

local ensemble_size = 0
if configuration:Exists("ensemble_size") then
  ensemble_size = tonumber(configuration:GetValue("ensemble_size"))
end

msg = string.format("Ensemble size: %d", ensemble_size)
logger:Info(msg)

for key, value in pairs(params) do
  ens = nil
  -- For MEPS we use lagged ensemble
  if currentProducerName == "MEPS" then
    ens = lagged_ensemble(param(key), "MEPS_LAGGED_ENSEMBLE")
  else
    ens = ensemble(param(key), ensemble_size)
  end

  ens:SetMaximumMissingForecasts(250)
  ens:Fetch(configuration, current_time, value[1])
  local sz = ens:Size()
  local missing_members = ensemble_size - sz

  params[key][3] = sz
  if missing_members > 0 then
    params[key][2] = value[2] + missing_members
  end
end

for key, value in pairs(params) do
  msg = string.format("Parameter '%s' missing %d members", key, value[2])
  logger:Info(msg)
end

local values = {}
for i = 1, result:SizeLocations() do
  values[i] = params["T-K"][3]
end

result:SetValues(values)
result:SetParam(param("ENSMEMB-N"))
result:SetForecastType(forecast_type(HPForecastType.kStatisticalProcessing))

logger:Info("Writing source data to file")
luatool:WriteToFile(result)
