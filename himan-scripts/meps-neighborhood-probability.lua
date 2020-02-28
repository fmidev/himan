-- Calculate probabilities from meps with area and time averaging
--

local MISS = missingf

local mask = matrixf(33, 33, 1, MISS)
mask:Fill(1)

function produceProbabilities(sourceparam, targetparam, op, limit)

  local datas = {}
  local producer = configuration:GetSourceProducer(0)
  local ensemble_size = tonumber(radon:GetProducerMetaData(producer, "ensemble size"))

  logger:Info("Producing area probabilities for " .. sourceparam:GetName())

  local ens = lagged_ensemble(sourceparam, "MEPS_LAGGED_ENSEMBLE")
  ens:SetMaximumMissingForecasts(ensemble_size)

  local curtime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())

  for j=0,3 do -- Look for the past 3 hours

    ens:Fetch(configuration, curtime, current_level)

    local actual_size = ens:Size()
    for i=0,actual_size-1 do
      local data = ens:GetForecast(i)

      if data then
        local reduced = nil
        if op == ">" then
          reduced = ProbLimitGt2D(data:GetData(), mask, limit):GetValues()
        elseif op == "==" then
          reduced = ProbLimitEq2D(data:GetData(), mask, limit):GetValues()
        end
        mvals = 0
        for k,v in pairs(reduced) do
          if v == v then
            reduced[k] = math.ceil(v)
          end
        end
        datas[#datas+1] = reduced
      end
    end

    curtime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)

  end
  logger:Info(string.format("Read %d grids", #datas))

  if #datas == 0 then
    return
  end

  local prob = {}

  for i=1,#datas[1] do
    prob[i] = MISS

    local tmp = 0
    local cnt = 0

    for j=1,#datas do
      local v = datas[j][i]
      if v == v then
        tmp = tmp + datas[j][i]
        cnt = cnt + 1
      end
    end

    prob[i] = tmp / cnt
  end

  proctype = nil

  if op == ">" then
    proctype = processing_type(HPProcessingType.kProbabilityGreaterThan, limit, kHPMissingValue)
  elseif op == "==" then
    proctype = processing_type(HPProcessingType.kProbabilityEq, limit, kHPMissingValue)
  end

  targetparam:SetProcessingType(proctype)

  if sourceparam:GetName() == "FFG-MS" then
    targetparam:SetAggregation(aggregation(HPAggregationType.kMaximum, configuration:GetForecastStep()))
  elseif sourceparam:GetName() == "RRR-KGM2" then
    targetparam:SetAggregation(aggregation(HPAggregationType.kAccumulation, configuration:GetForecastStep()))
  end

  result:SetForecastType(forecast_type(HPForecastType.kStatisticalProcessing))
  result:SetParam(targetparam)
  result:SetValues(prob)

  luatool:WriteToFile(result)

end

local sourceparam = configuration:GetValue("source_param")
local targetparam = configuration:GetValue("target_param")
local op = configuration:GetValue("comparison_op")
local limit = tonumber(configuration:GetValue("limit"))

if sourceparam == "" or targetparam == "" or op == "" or limit == nil then
  logger:Error("Source or target param or limit not specified")
  return
end

logger:Info(string.format("Source: %s Target: %s Operator: %s Limit: %d", sourceparam, targetparam, op, limit))
produceProbabilities(param(sourceparam), param(targetparam), op, limit)
