function compute_std_mean_ec(N_EC, N_MEAN_EC, N_STD_EC, wt_ec, wt_ens_ec, wt_sum)
  local mean = (N_EC * wt_ec + N_MEAN_EC * wt_ens_ec) / wt_sum
  local stDev = math.sqrt(math.max(((wt_ens_ec * (N_STD_EC^2 + N_MEAN_EC^2) + wt_ec * N_EC^2) / wt_sum) - mean^2, 0))
  return mean, stDev
end

function compute_std_mean_all(N_EC, N_MEPS, N_MEAN_EC, N_MEAN_MEPS, N_STD_EC, N_STD_MEPS, wt_ec, wt_ens_ec, wt_meps, wt_ens_meps, wt_sum)
  local mean = (N_EC * wt_ec + N_MEAN_EC * wt_ens_ec + N_MEPS * wt_meps + N_MEAN_MEPS * wt_ens_meps) / wt_sum
  local stDev = math.sqrt(math.max(((wt_ens_ec * (N_STD_EC^2 + N_MEAN_EC^2) + wt_ec * N_EC^2 + wt_meps * N_MEPS^2 + wt_ens_meps * (N_STD_MEPS^2 + N_MEAN_MEPS^2)) / wt_sum) - mean^2, 0))
  return mean, stDev
end

function convert_to_100(array)
  if not array then return nil end 
  local t = {}
  for i = 1, #array do
      t[i] = array[i] * 100
  end
  return t
end

function process_params(producer, ftype, ...)
  local params = { ... }
  local results = { get_param(producer, ftype, table.unpack(params)) }
  
  -- Convert all results to 100-scale
  for i = 1, #results do
      results[i] = convert_to_100(results[i])
  end
  
  return table.unpack(results)  
end

function get_data(producer1, producer2, ftype)
  local mean_params = {
      param("NL-MEAN-0TO1", aggregation(), processing_type(HPProcessingType.kMean)),
      param("NM-MEAN-0TO1", aggregation(), processing_type(HPProcessingType.kMean)),
      param("NH-MEAN-0TO1", aggregation(), processing_type(HPProcessingType.kMean))
  }

  local std_params = {
      param("NL-STDDEV-0TO1"),
      param("NM-STDDEV-0TO1"),
      param("NH-STDDEV-0TO1")
  }

  local main_params = {
      param("NL-0TO1"),
      param("NM-0TO1"),
      param("NH-0TO1")
  }

  local NL_MEAN, NM_MEAN, NH_MEAN = process_params(producer2, forecast_type(HPForecastType.kStatisticalProcessing), table.unpack(mean_params))
  local NL_STD, NM_STD, NH_STD = process_params(producer2, forecast_type(HPForecastType.kStatisticalProcessing), table.unpack(std_params))
  local NL, NM, NH = process_params(producer1, ftype, table.unpack(main_params))

  return NL, NM, NH, NL_MEAN, NM_MEAN, NH_MEAN, NL_STD, NM_STD, NH_STD
end

-- Get origin times for MEPS and EC
function get_time(producer)
  
  local test = configuration:Exists("origin_time_test")

  if test then
    local test_time = configuration:GetValue("origin_time_test")
    ftime = forecast_time(raw_time(test_time), raw_time("2025-04-04 06:00:00"))
    return ftime
  end

  local vire_hour = current_time:GetOriginDateTime():String('%H')
  local producer_id = producer:GetId()

  if producer_id == 4 or producer_id == 260 then
    adjust_hours = -4
  elseif (vire_hour == '07' or vire_hour == '19') and (producer_id == 131 or producer_id == 242) then
      adjust_hours = -7
  elseif (vire_hour == '13' or vire_hour == '01') and (producer_id == 131 or producer_id == 242) then
      adjust_hours = -13
  end

  if adjust_hours then
    current_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjust_hours)
  end
  
  ftime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())
  current_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, -adjust_hours) -- reset to original time
  return ftime
end



function get_param(producer, ftype, param1, param2, param3)

  local ftime = get_time(producer)

  
  local o = {forecast_time = ftime,
  level = level(HPLevelType.kHeight, 0),
  param = param1,
  forecast_type = ftype,
  time_interpolation = true,
  time_interpolation_search_step = time_duration("01:00:00"),
  producer = producer}
  local param1 = luatool:FetchWithArgs(o)

  o.param = param2
  local param2 = luatool:FetchWithArgs(o)

  o.param = param3
  local param3 = luatool:FetchWithArgs(o)
  return param1, param2, param3
end

function compute_levels(step)
  local mid_level = 50 + step^(1.5) / 500
  local dev_factor_low = 1.2 + step ^(1.5) / 10000
  local dev_factor_mid = 1.1 + step^ (1.5) / 12500
  local dev_factor_high = 1 + step^ (1.5) / 15000
  return mid_level, dev_factor_low, dev_factor_mid, dev_factor_high
end

function compute_weights(step)
  local ratio = math.min(0.0001175 * step ^ 2 + 0.35, 1) 
  local wt_meps = (1 - ratio) * (1/4)
  local wt_ens_meps = (1 - ratio) * (3/4)
  local wt_ec = ratio * (1 - 0.1818 * step^(0.25867))
  local wt_ens_ec = ratio - wt_ec
  return wt_meps, wt_ens_meps, wt_ec, wt_ens_ec
end


local step = current_time:GetStep():Hours()

local wt_meps, wt_ens_meps, wt_ec, wt_ens_ec = compute_weights(step)
local mid_level, dev_factor_low, dev_factor_mid, dev_factor_high = compute_levels(step)

local mean, stDev = nil, nil
local cl, cm, ch = {}, {}, {} 


local disable_meps = nil
if configuration:Exists("disable_meps") then 
  disable_meps = ParseBoolean(configuration:GetValue("disable_meps"))
end

meps_time = get_time(producer(4, "MEPS"))
meps_step = tonumber(meps_time:GetStep():Hours())

if disable_meps or meps_step > 66 then
  logger:Info("Only using EC data")
  
  NL_EC, NM_EC, NH_EC, NL_MEAN_EC, NM_MEAN_EC, NH_MEAN_EC, NL_STD_EC, NM_STD_EC, NH_STD_EC = get_data(producer(131, "ECG"), producer(242, "ECM_PROB"), forecast_type(HPForecastType.kDeterministic))
  
  if not NL_EC or not NM_EC or not NH_EC then
    logger:Warning("Some data not found")
  else
    logger:Info("EC Data fetched")
  end

  local wt_sum = wt_ens_ec + wt_ec
  for i = 1, #NL_EC do

    mean, stDev = compute_std_mean_ec(NL_EC[i], NL_MEAN_EC[i], NL_STD_EC[i], wt_ec, wt_ens_ec, wt_sum)
    cl[i] = mean + stDev ^ dev_factor_low * (mean - mid_level) / 50

    mean, stDev = compute_std_mean_ec(NM_EC[i], NM_MEAN_EC[i], NM_STD_EC[i], wt_ec, wt_ens_ec, wt_sum)
    cm[i] = mean + stDev ^ dev_factor_mid * (mean - mid_level) / 50

    mean, stDev = compute_std_mean_ec(NH_EC[i], NH_MEAN_EC[i], NH_STD_EC[i], wt_ec, wt_ens_ec, wt_sum)
    ch[i] = mean + stDev ^ dev_factor_high * (mean - mid_level) / 50
  end
else
  logger:Info("Using MEPS and EC data")
  
  local prod1 = producer(131, "ECG")
  local prod2 = producer(242, "ECM_PROB")
  local prod3 = producer(4,"MEPS")
  local prod4 = producer(260, "MEPSMTA")

  NL_EC, NM_EC, NH_EC, NL_MEAN_EC, NM_MEAN_EC, NH_MEAN_EC, NL_STD_EC, NM_STD_EC, NH_STD_EC = get_data(prod1, prod2, forecast_type(HPForecastType.kDeterministic))
  NL_MEPS, NM_MEPS, NH_MEPS, NL_MEAN_MEPS, NM_MEAN_MEPS, NH_MEAN_MEPS, NL_STD_MEPS, NM_STD_MEPS, NH_STD_MEPS = get_data(prod3, prod4,forecast_type(HPForecastType.kEpsControl, 0))
  
  if not NL_EC or not NM_EC or not NH_EC or not NL_MEPS or not NM_MEPS or not NH_MEPS then
    logger:Warning("Some data not found")
  else
    logger:Info("EC and MEPS Data fetched")
  end

  local wt_sum = wt_ens_ec + wt_ec + wt_meps + wt_ens_meps
  for i = 1, #NL_EC do

    mean, stDev = compute_std_mean_all(NL_EC[i], NL_MEPS[i], NL_MEAN_EC[i], NL_MEAN_MEPS[i], NL_STD_EC[i], NL_STD_MEPS[i], wt_ec, wt_ens_ec, wt_meps, wt_ens_meps, wt_sum)
    cl[i] = mean + stDev ^ dev_factor_low * (mean - mid_level) / 50

    mean, stDev = compute_std_mean_all(NM_EC[i], NM_MEPS[i], NM_MEAN_EC[i], NM_MEAN_MEPS[i], NM_STD_EC[i], NM_STD_MEPS[i], wt_ec, wt_ens_ec, wt_meps, wt_ens_meps, wt_sum)
    cm[i] = mean + stDev ^ dev_factor_mid * (mean - mid_level) / 50

    mean, stDev = compute_std_mean_all(NH_EC[i], NH_MEPS[i], NH_MEAN_EC[i], NH_MEAN_MEPS[i], NH_STD_EC[i], NH_STD_MEPS[i], wt_ec, wt_ens_ec, wt_meps, wt_ens_meps, wt_sum)
    ch[i] = mean + stDev ^ dev_factor_high * (mean - mid_level) / 50
  end
end

local n = {}

for i=1, #cl do 
  cl[i] = 0.5 * math.sqrt(cl[i] ^2) - 0.5 * math.sqrt((cl[i] - 100) ^2) + 50
  cm[i] = 0.5 * math.sqrt(cm[i] ^2) - 0.5 * math.sqrt((cm[i] - 100) ^2) + 50
  ch[i] = 0.5 * math.sqrt(ch[i] ^2) - 0.5 * math.sqrt((ch[i] - 100) ^2) + 50

  n[i] = 100 - (1 - cl[i] * 1/100) * (1 - cm[i] * 0.75/100) * (1 - ch[i] * 0.25/100) * (1 - cl[i] * cm[i] * (1/3 - (1 - ((50 - cl[i]) ^2 + (50 - cm[i]) ^2) / 5000))/10000) * (1 - cm[i] * ch[i] * (2/3 - (1 - ((50 - cm[i]) ^2 + (50 - ch[i]) ^2) / 5000))/10000) * 100
  n[i] = n[i] * 0.01
  
  if n[i] > 1 then
    n[i] = 1
  end
end


result:SetParam(param('N-0TO1'))
result:SetValues(n)

luatool:WriteToFile(result)