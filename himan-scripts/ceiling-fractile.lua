
function fills_to_nan(vals)
  local temp = {}

  for k,v in pairs(vals) do
    if v==19999 then
      table.insert( temp, missing )
    else 
      table.insert( temp, v )
    end
  end

  return temp
end

function fill_sort(vals)
  local temp = {}

  for k,v in pairs(vals) do
    if type(v) == 'number' and v==v then
      table.insert( temp, v )
    else 
      table.insert( temp, 19999 )
    end
  end

  table.sort( temp )
  return temp
end


function percentile(vals, p)
  local index = p * #vals
  local int_index = math.floor(index)
  local percentile = nil

  local F0 = vals[1]


  if p == 0 then
    percentile = vals[1]
  elseif p == 1 then
    percentile =  vals[#vals]
  else
    percentile = (vals[int_index + 1] - vals[int_index]) * (index - int_index) + vals[int_index]
  end

  return percentile
end

function write_fractile(fractile, par)
  result:SetValues(fractile)
  result:SetParam(par)
  result:SetForecastType(forecast_type(HPForecastType.kStatisticalProcessing))

  luatool:WriteToFile(result)
end

producer = configuration:GetTargetProducer()
local ens = nil
local ensSize = nil

if producer:GetId() == 242 then 
  ens = ensemble(param("CEIL-2-M"), 51, 250)
else 
  ensSize = tonumber(radon:GetProducerMetaData(producer, "ensemble size"))

  if not ensSize then
    logger.Error("Ensemble size not found from database for producer " .. producer:GetId())
    return
  end

  ens = lagged_ensemble(param("CEIL-2-M"), "MEPS_LAGGED_ENSEMBLE", ensSize)
end

ens:Fetch(configuration, current_time, current_level)
local lagEnsSize = ens:Size()

local F0, F10, F25, F50, F75, F90, F100 = {}, {}, {}, {}, {}, {}, {}

ens:FirstLocation()
local values = ens:Values()
values = fill_sort(values)
values = fills_to_nan(values)

F0[1] = percentile(values, 0)
F10[1] = percentile(values, 0.1)
F25[1] = percentile(values, 0.25)
F50[1] = percentile(values, 0.5)
F75[1] = percentile(values, 0.75)
F90[1] = percentile(values, 0.9)
F100[1] = percentile(values, 1)

local i = 2
while ens:NextLocation() do
  local values = ens:Values()
  values = fill_sort(values)
  values = fills_to_nan(values)

  F0[i] = percentile(values, 0)
  F10[i] = percentile(values, 0.1)
  F25[i] = percentile(values, 0.25)
  F50[i] = percentile(values, 0.5)
  F75[i] = percentile(values, 0.75)
  F90[i] = percentile(values, 0.9)
  F100[i] = percentile(values, 1)
  i = i + 1
end

write_fractile(F0, param("F0-CEIL-2-M"))
write_fractile(F10, param("F10-CEIL-2-M"))
write_fractile(F25, param("F25-CEIL-2-M"))
write_fractile(F50, param("F50-CEIL-2-M"))
write_fractile(F75, param("F75-CEIL-2-M"))
write_fractile(F90, param("F90-CEIL-2-M"))
write_fractile(F100, param("F100-CEIL-2-M"))
