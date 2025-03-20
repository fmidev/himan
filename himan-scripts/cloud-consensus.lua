
print("origin time: " .. current_time:GetOriginDateTime():String("%Y-%m-%d %H:%M:%S"))

function convert_to_100(array)
  local t = {}
  for i = 1, #array do
      t[i] = array[i] * 100
  end
  return t
end

function get_data(producer1, producer2, ftype)
  local mean1 = param("NL-MEAN-0TO1", aggregation(), processing_type(HPProcessingType.kMean))
  local mean2 = param("NM-MEAN-0TO1", aggregation(), processing_type(HPProcessingType.kMean))
  local mean3 = param("NH-MEAN-0TO1", aggregation(), processing_type(HPProcessingType.kMean))
  NL_MEAN, NM_MEAN, NH_MEAN = get_param(producer2, forecast_type(HPForecastType.kStatisticalProcessing), mean1, mean2, mean3)
  NL_STD, NM_STD, NH_STD = get_param(producer2, forecast_type(HPForecastType.kStatisticalProcessing), param("NL-STDDEV-0TO1"), param("NM-STDDEV-0TO1"), param("NH-STDDEV-0TO1"))
  NL, NM, NH = get_param(producer1, ftype, param("NL-0TO1"), param("NM-0TO1"), param("NH-0TO1"))

  NL_MEAN = convert_to_100(NL_MEAN)
  NM_MEAN = convert_to_100(NM_MEAN)
  NH_MEAN = convert_to_100(NH_MEAN)
  NL_STD = convert_to_100(NL_STD)
  NM_STD = convert_to_100(NM_STD)
  NH_STD = convert_to_100(NH_STD)
  NL = convert_to_100(NL)
  NM = convert_to_100(NM)
  NH = convert_to_100(NH)

  return NL, NM, NH, NL_MEAN, NM_MEAN, NH_MEAN, NL_STD, NM_STD, NH_STD
end

function get_param(producer, ftype, param1, param2, param3)
  local o = {forecast_time = current_time,
  level = level(HPLevelType.kHeight, 0),
  param = param1,
  forecast_type = ftype,
  producer = producer}
  local param1 = luatool:FetchWithArgs(o)

  o.param = param2
  local param2 = luatool:FetchWithArgs(o)

  o.param = param3
  local param3 = luatool:FetchWithArgs(o)
  return param1, param2, param3
end


local step = current_time:GetStep():Hours()

local ratio = 0.0001175 * step ^ 2 + 0.35
if ratio > 1 then
  ratio = 1
end

local wt_meps = (1 - ratio) * (1/4)
local wt_ens_meps = (1 - ratio) * (3/4)

local wt_ec = ratio * (1 - 0.1818 * step^(0.25867))
local wt_ens_ec = ratio - wt_ec

local mid_level = 50 + step^(1.5) / 500

local dev_factor_low = 1.2 + step ^(1.5) / 10000
local dev_factor_mid = 1.1 + step^ (1.5) / 12500
local dev_factor_high = 1 + step^ (1.5) / 15000

local mean = {}
local stDev = {}

local cl = {}
local cm = {}
local ch = {}


if configuration:GetValue("disable_meps") == "false" then
  print("MEPS is enabled")
  
  local prod1 = producer(131, "ECG")
  local prod2 = producer(242, "ECM_PROB")
  local prod3 = producer(4,"MEPS")
  local prod4 = producer(260, "MEPSMTA")
  NL_EC, NM_EC, NH_EC, NL_MEAN_EC, NM_MEAN_EC, NH_MEAN_EC, NL_STD_EC, NM_STD_EC, NH_STD_EC = get_data(prod1, prod2, forecast_type(HPForecastType.kDeterministic))
  NL_MEPS, NM_MEPS, NH_MEPS, NL_MEAN_MEPS, NM_MEAN_MEPS, NH_MEAN_MEPS, NL_STD_MEPS, NM_STD_MEPS, NH_STD_MEPS = get_data(prod3, prod4,forecast_type(HPForecastType.kEpsControl, 0))
  print("EC and MEPS data fetched")

  local wt_sum = wt_ens_ec + wt_ec + wt_meps + wt_ens_meps
  for i = 1, #NL_EC do

    mean[i] = (NL_EC[i] * wt_ec + NL_MEAN_EC[i] * wt_ens_ec + NL_MEPS[i] * wt_meps + NL_MEAN_MEPS[i] * wt_ens_meps) / wt_sum
    stDev[i] = math.sqrt(((wt_ens_ec * (NL_STD_EC[i]^2 + NL_MEAN_EC[i]^2) + wt_ec * NL_EC[i]^2 + wt_meps * NL_MEPS[i]^2 + wt_ens_meps * (NL_STD_MEPS[i]^2 + NL_MEAN_MEPS[i]^2)) / wt_sum) - mean[i]^2)
    cl[i] = mean[i] + stDev[i] ^ dev_factor_low * (mean[i] - mid_level) / 50

    mean[i] = (NM_EC[i] * wt_ec + NM_MEAN_EC[i] * wt_ens_ec + NM_MEPS[i] * wt_meps + NM_MEAN_MEPS[i] * wt_ens_meps) / wt_sum
    stDev[i] = math.sqrt(((wt_ens_ec * (NM_STD_EC[i]^2 + NM_MEAN_EC[i]^2) + wt_ec * NM_EC[i]^2 + wt_meps * NM_MEPS[i]^2 + wt_ens_meps * (NM_STD_MEPS[i]^2 + NM_MEAN_MEPS[i]^2)) / wt_sum) - mean[i]^2)
    cm[i] = mean[i] + stDev[i] ^ dev_factor_mid * (mean[i] - mid_level) / 50

    mean[i] = (NH_EC[i] * wt_ec + NH_MEAN_EC[i] * wt_ens_ec + NH_MEPS[i] * wt_meps + NH_MEAN_MEPS[i] * wt_ens_meps) / wt_sum
    stDev[i] = math.sqrt(((wt_ens_ec * (NH_STD_EC[i]^2 + NH_MEAN_EC[i]^2) + wt_ec * NH_EC[i]^2 + wt_meps * NH_MEPS[i]^2 + wt_ens_meps * (NH_STD_MEPS[i]^2 + NH_MEAN_MEPS[i]^2)) / wt_sum) - mean[i]^2)
    ch[i] = mean[i] + stDev[i] ^ dev_factor_high * (mean[i] - mid_level) / 50
  end
else
  print("MEPS is disabled")

  NL_EC, NM_EC, NH_EC, NL_MEAN_EC, NM_MEAN_EC, NH_MEAN_EC, NL_STD_EC, NM_STD_EC, NH_STD_EC = get_data(producer(131, "ECG"), producer(242, "ECM_PROB"), forecast_type(HPForecastType.kDeterministic))
  print("EC data fetched")

  local wt_sum = wt_ens_ec + wt_ec
  for i = 1, #NL_EC do

    mean[i] = (NL_EC[i] * wt_ec + NL_MEAN_EC[i] * wt_ens_ec) / wt_sum
    stDev[i] = math.sqrt(((wt_ens_ec * (NL_STD_EC[i]^2 + NL_MEAN_EC[i]^2) + wt_ec * NL_EC[i]^2) / wt_sum) - mean[i]^2)
    cl[i] = mean[i] + stDev[i] ^ dev_factor_low * (mean[i] - mid_level) / 50

    mean[i] = (NM_EC[i] * wt_ec + NM_MEAN_EC[i] * wt_ens_ec) / wt_sum
    stDev[i] = math.sqrt(((wt_ens_ec * (NM_STD_EC[i]^2 + NM_MEAN_EC[i]^2) + wt_ec * NM_EC[i]^2) / wt_sum) - mean[i]^2)
    cm[i] = mean[i] + stDev[i] ^ dev_factor_mid * (mean[i] - mid_level) / 50

    mean[i] = (NH_EC[i] * wt_ec + NH_MEAN_EC[i] * wt_ens_ec) / wt_sum
    stDev[i] = math.sqrt(((wt_ens_ec * (NH_STD_EC[i]^2 + NH_MEAN_EC[i]^2) + wt_ec * NH_EC[i]^2) / wt_sum) - mean[i]^2)
    ch[i] = mean[i] + stDev[i] ^ dev_factor_high * (mean[i] - mid_level) / 50
  end
end

local n = {}
for i=1, #cl do 
  cl[i] = 0.5 * math.sqrt(cl[i] ^2) - 0.5 * math.sqrt((cl[i] - 100) ^2) + 50
  cm[i] = 0.5 * math.sqrt(cm[i] ^2) - 0.5 * math.sqrt((cm[i] - 100) ^2) + 50
  ch[i] = 0.5 * math.sqrt(ch[i] ^2) - 0.5 * math.sqrt((ch[i] - 100) ^2) + 50

  n[i] = 100 - (1 - cl[i] * 1/100) * (1 - cm[i] * 0.75/100) * (1 - ch[i] * 0.25/100) * (1 - cl[i] * cm[i] * (1/3 - (1 - ((50 - cl[i]) ^2 + (50 - cm[i]) ^2) / 5000))/10000) * (1 - cm[i] * ch[i] * (2/3 - (1 - ((50 - cm[i]) ^2 + (50 - ch[i]) ^2) / 5000))/10000) * 100
  n[i] = n[i] * 0.01
end

local max_value = nil
for _, value in pairs(n) do
    if max_value == nil or value > max_value then
        max_value = value
    end
end

local max_value = nil
local min_value = nil
local sum = 0
local count = 0

for _, value in pairs(n) do
    -- Update max value
    if max_value == nil or value > max_value then
        max_value = value
    end

    -- Update min value
    if min_value == nil or value < min_value then
        min_value = value
    end

    -- Compute sum and count for mean calculation
    sum = sum + value
    count = count + 1
end

-- Calculate mean
local mean_value = count > 0 and (sum / count) or 0

-- Print results
print("Minimum value:", min_value)
print("Sum value:", sum)
print("Maximum value:", max_value)


-- result:SetParam(param('N-0TO1'))
-- result:SetValues(n)

-- luatool:WriteToFile(result)