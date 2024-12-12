logger:Info("Calculating ceiling fractiles")

function is_nan(val)
  return val ~= val
end 

function nans_to_9999(vals)
  for i = 1, #vals do
    if is_nan(vals[i]) then
      vals[i] = 9999
    end
  end
  return vals
end

function fills_to_nan(vals)
  local with_nan = {}
  for i = 1, #vals do
    if vals[i] == 9999 then
      with_nan[i] = 0/0
    else
      with_nan[i] = vals[i]
    end
  end
  return with_nan
end

function get_percentile(vals, percent)
  local index = math.ceil(percent * #vals)
  if index < 1 then index = 1 end
  if index > #vals then index = #vals end
  return vals[index]
end

function write_fractile(fractile, par)
  result:SetValues(fractile)
  result:SetParam(par)
  luatool:WriteToFile(result)
end

local producer = configuration:GetSourceProducer(0)
local ensSize = tonumber(radon:GetProducerMetaData(producer, "ensemble size"))

print('enssize: '.. ensSize)

if not ensSize then
  logger.Error("Ensemble size not found from database for producer " .. producer:GetId())
  return
end

local ens = lagged_ensemble(param("CEIL-2-M"), "MEPS_LAGGED_ENSEMBLE", ensSize)

ens:Fetch(configuration, current_time, current_level)

local lagEnsSize = ens:Size()

local F0, F10, F25, F50, F75, F90, F100 = {}, {}, {}, {}, {}, {}, {}
while ens:NextLocation() do
    local vals = ens:Values()

    vals = nans_to_9999(vals)

    table.sort(vals)

    vals = fills_to_nan(vals)

    F0[#F0 + 1] = get_percentile(vals, 0)
    F10[#F10 + 1] = get_percentile(vals, 0.1)
    F25[#F25 + 1] = get_percentile(vals, 0.25)
    F50[#F50 + 1] = get_percentile(vals, 0.5)
    F75[#F75 + 1] = get_percentile(vals, 0.75)
    F90[#F90 + 1] = get_percentile(vals, 0.9)
    F100[#F100 + 1] = get_percentile(vals, 1)

end

-- set results to a parameter
write_fractile(F0, param("F0-CEIL-2-M"))
write_fractile(F10, param("F10-CEIL-2-M"))
write_fractile(F25, param("F25-CEIL-2-M"))
write_fractile(F50, param("F50-CEIL-2-M"))
write_fractile(F75, param("F75-CEIL-2-M"))
write_fractile(F90, param("F90-CEIL-2-M"))
write_fractile(F100, param("F100-CEIL-2-M"))