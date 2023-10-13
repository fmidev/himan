-- Calculate windgust duration within 24h
-- Function reads past 24h from given time. Works only with ECMWF data.

local potpar = param("PROB-POT-3", aggregation(), processing_type(HPProcessingType.kProbabilityGreaterThan, 60))

function WriteToFile(gsum, ftime)
  local start = ftime:GetStep():Hours()
  local agg = aggregation(HPAggregationType.kAccumulation, time_duration(HPTimeResolution.kHourResolution, 24))
  local par = param("RAJUILMA-N")
  par:SetAggregation(agg)

  result:SetParam(par)
  result:SetValues(gsum)
  luatool:WriteToFile(result)
end

function TimeMax(curTime,param_csi)

  local myTime = forecast_time(curTime:GetOriginDateTime(),curTime:GetValidDateTime())

  local step = myTime:GetStep():Hours()

  if step == 0 then
    return
  end

  -- calculate 24h gust

  local csimax= {}

  local stopStep = math.max(myTime:GetStep():Hours() - 24, 0)

  local count = 0

  while true do
    local stepAdjustment = -1

    if step >= 150 then
      stepAdjustment = -6
    elseif step >= 93 then
      stepAdjustment = -3
    end

    myTime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, stepAdjustment)

    if myTime:GetStep():Hours() < stopStep or myTime:GetStep():Hours() < 0 then
      break
    end

    local pot_3 = luatool:Fetch(myTime, current_level, potpar, forecast_type(HPForecastType.kStatisticalProcessing))
    local csi = luatool:Fetch(myTime, current_level, param_csi, forecast_type(HPForecastType.kStatisticalProcessing))

    if pot_3 then
      if #csimax == 0 then
        for i=1, #pot_3 do
         csimax[i] = 0
        end
      end

      for i=1, #pot_3 do
        if pot_3[i] >= 0.10 then
          csimax[i] = math.max(csimax[i], csi[i])
        end
      end
    end

    step = myTime:GetStep():Hours()
  end

  if #csimax == 0 then
    return
  end

  return csimax
end

function Rajuilma(curTime)

  local step = curTime:GetStep():Hours()

  if step == 0 then
    return
  end

  local amber = {}
  local red = {}

  local warning = {}

  local yellow = TimeMax(curTime,param("PROB-CSI-1", aggregation(), processing_type(HPProcessingType.kProbabilityGreaterThan, 30)))
  local amber = TimeMax(curTime,param("PROB-CSI-2", aggregation(), processing_type(HPProcessingType.kProbabilityGreaterThan, 60)))
  local red = TimeMax(curTime,param("PROB-CSI-3", aggregation(), processing_type(HPProcessingType.kProbabilityGreaterThan, 100)))

  local ukkonen = TimeMax(curTime, potpar)

  if #warning == 0 then
    for i=1, #yellow do
      warning[i] = 0
    end
  end


  for i=1, #warning do
    if yellow[i] >= 0.440 then
      warning[i] = 1
    end
    if amber[i] >= 0.30 then
      warning[i] = 2
    end
    if red[i] >= 0.20 then
      warning[i] = 3
    end

    if ukkonen[i] < 0.10 then
      warning[i] = missing
    end

    if ukkonen[i] >=0.10 and yellow[i] < 0.40 and amber[i] < 0.30 and red[i] < 0.20 then
      warning[i] = 0
    end
  end

  if #warning == 0 then
    return
  end

  WriteToFile(warning, curTime)

end

local step = current_time:GetStep():Hours()

if step < 24 then
  logger:Info(string.format("Step is less than first 24h -- skipping", step))
  return
end

Rajuilma(current_time)
