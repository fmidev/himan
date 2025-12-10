-- Cloud consensus for cloud fraction from MEPS and ECMWF deterministic and ensemble forecasts.
-- Original algorithm by Jani Sorsa.
-- Comments copied and translated from original SmartTool code.
-- Rain and cloud layer corrections are also included. 
-- Original code for corrections from snwc-cloudiness-and-precipitation.lua and snwc-cloudlayers.lua.


-- Get julian date in radians from the valid time
function julian_day_radians()
  local year = tonumber(current_time:GetValidDateTime():String('%Y'))
  local month = tonumber(current_time:GetValidDateTime():String('%m'))
  local day = tonumber(current_time:GetValidDateTime():String('%d'))

  local days_in_month = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
  local days_in_year = 365

  local function is_leap(y)
    return (y % 4 == 0 and y % 100 ~= 0) or (y % 400 == 0)
  end

  if is_leap(year) then
      days_in_month[2] = 29
      days_in_year = 366
  end

  local doy = day
  for m = 1, month - 1 do
      doy = doy + days_in_month[m]
  end

  return 2 * math.pi * (doy / days_in_year)
end

-- Weighted mean and standard deviation are calculated.
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

-- Rain correction:
-- Check consistency between total cloudiness and precipitation; first one is modified accordingly.
function rain_correction(RR, N)
  local _N = {}

  for i=1,#N do
    _N[i] = N[i]
    -- If there is even light precipitation, there should also be clouds
    if RR[i] > 0.01 then
      _N[i] = math.max(_N[i], 0.5)
    end
  end

  return _N
end

-- Cloud layers are corrected:
-- Add low, mid, or high clouds so that one layer matches the total cloud cover.
-- The layer is chosen based on which has the highest initial amount.
function correct_cloudlayers(N, NL, NM, NH)
  local _NH = {}
  local _NM = {}
  local _NL = {}

  for i=1, #N do
    local cc = N[i]
    local ch = NH[i]
    local cm = NM[i]
    local cl = NL[i]

    _NH[i] = ch
    _NM[i] = cm
    _NL[i] = cl

    if (cl < cc and cl >= cm and cl >= ch) then
      _NL[i] = cc
    end

    if (cm < cc and cl < cm and ch < cm) then
      _NM[i] = cc
    end

    if (cm < cc and cl < ch and cm < ch) then
      _NH[i] = cc
    end

    _NL[i] = math.min(cl, cc)
    _NM[i] = math.min(cm, cc)

  end

  return _NL, _NM, _NH
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

  --local RR = process_params(producer2, forecast_type(HPForecastType.kEpsControl, 0), table.unpack(rain_param))
  local NL_MEAN, NM_MEAN, NH_MEAN = process_params(producer2, forecast_type(HPForecastType.kStatisticalProcessing), table.unpack(mean_params))
  local NL_STD, NM_STD, NH_STD = process_params(producer2, forecast_type(HPForecastType.kStatisticalProcessing), table.unpack(std_params))
  local NL, NM, NH = process_params(producer1, ftype, table.unpack(main_params))

  return NL, NM, NH, NL_MEAN, NM_MEAN, NH_MEAN, NL_STD, NM_STD, NH_STD
end

-- Get origin times for MEPS and EC
function get_time(producer, get_earlier)
  
  local test = configuration:Exists("origin_time_test")

  if test then
    local test_time = configuration:GetValue("origin_time_test")
    ftime = forecast_time(raw_time(test_time), raw_time("2025-05-20 15:00:00"))
    return ftime
  end

  local vire_hour = current_time:GetOriginDateTime():String('%H')
  local producer_id = producer:GetId()

  if producer_id == 4 or producer_id == 260 then
    adjust_hours = -4
  elseif (vire_hour == '07' or vire_hour == '19') and (producer_id == 131 or producer_id == 242 or producer_id == 240) then
    adjust_hours = -7
  elseif (vire_hour == '13' or vire_hour == '01') and (producer_id == 131 or producer_id == 242 or producer_id == 240) then
    adjust_hours = -13
  end

  if get_earlier and producer_id == 242 then
    adjust_hours = -19
  end

  local ftime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())
  ftime:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjust_hours)

  return ftime
end



function get_param(producer, ftype, ...)
  local param_list = { ... }
  local ftime = get_time(producer, false)
  local results = {}

  local o = {
    forecast_time = ftime,
    level = level(HPLevelType.kHeight, 0),
    forecast_type = ftype,
    time_interpolation = true,
    time_interpolation_search_step = time_duration("01:00:00"),
    producer = producer
  }

  for i, param in ipairs(param_list) do
    o.param = param
    results[i] = luatool:FetchWithArgs(o)

    if not results[i] then
      ftime = get_time(producer, true)
      o.forecast_time = ftime
      results[i] = luatool:FetchWithArgs(o)
    end
  end

  return table.unpack(results)
end


-- Other weights:
-- mid_level: determines the cloud cover midpoint threshold above which cloudiness is increased and below which it is decreased.
-- Values below 50 increase cloud cover, values above 50 decrease it. The effect increases with forecast lead time, i.e. cloudiness is reduced more for later forecast days than for near-term ones.
-- dev_factor_low, dev_factor_mid, dev_factor_high: control the strength of extremization. The goal is to reduce mid-range values, which are common in averaging. 
-- Each cloud layer has its own adjustment. Low clouds are pushed more aggressively toward 0% or 100%, while mid/high clouds are allowed to stay in the 30–70% range more often.
function compute_levels(step)
  local jday_rad = julian_day_radians()

  local mid_level = 49.5 + 1.5 * math.cos(jday_rad) + step^(1.5) / 1000
  local dev_factor_low = 1.2 + step ^(1.5) / 10000
  local dev_factor_mid = 1.1 + step^ (1.5) / 12500
  local dev_factor_high = 1 + step^ (1.5) / 15000
  return mid_level, dev_factor_low, dev_factor_mid, dev_factor_high
end

-- Model and ensemble weights:
-- Calculate the ratio of EC to MEPS. EC's share increases with forecast lead time from 35% to 100%.
-- Calculate the ratio between the ensemble and deterministic runs. MEPS ratio is a fixed 3:1, while for EC the ensemble share increases with forecast lead time.
function compute_weights(step)
  local ratio = math.min(0.0001175 * step ^ 2 + 0.35, 1) 
  local wt_meps = (1 - ratio) * (1/4)
  local wt_ens_meps = (1 - ratio) * (3/4)
  local wt_ec = ratio * (1 - 0.1818 * step^(0.25867))
  local wt_ens_ec = ratio - wt_ec
  return wt_meps, wt_ens_meps, wt_ec, wt_ens_ec
end

function limit_values(array)
  
  for i=1, #array do
    if array[i] < 0 then
      array[i] = 0
    elseif array[i] > 1 then
      array[i] = 1
    end
  end

  return array
end 

function write_results_to_file(param, data)
  result:SetParam(param)
  result:SetValues(data)

  luatool:WriteToFile(result)
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

local ec = producer(131, "ECG")
local ec_prob = producer(242, "ECM_PROB")
local ec_mta = producer(240, "ECMMTA")
local meps = producer(4,"MEPS")
local meps_mta = producer(260, "MEPSMTA")

local meps_time = get_time(meps, false)
local meps_step = tonumber(meps_time:GetStep():Hours())


-- Get all cloud data from MEPS and EC
local NL_EC, NM_EC, NH_EC, NL_MEAN_EC, NM_MEAN_EC, NH_MEAN_EC, NL_STD_EC, NM_STD_EC, NH_STD_EC = get_data(ec, ec_prob, forecast_type(HPForecastType.kDeterministic))
local NL_MEPS, NM_MEPS, NH_MEPS, NL_MEAN_MEPS, NM_MEAN_MEPS, NH_MEAN_MEPS, NL_STD_MEPS, NM_STD_MEPS, NH_STD_MEPS = get_data(meps, meps_mta,forecast_type(HPForecastType.kEpsControl, 0))

-- Get cloud data from VIRE
local NL_VIRE = luatool:Fetch(current_time, current_level, param("NL-0TO1"), current_forecast_type)
local NM_VIRE = luatool:Fetch(current_time, current_level, param("NM-0TO1"), current_forecast_type)
local NH_VIRE = luatool:Fetch(current_time, current_level, param("NH-0TO1"), current_forecast_type)

if not NL_VIRE or not NM_VIRE or not NH_VIRE then
  logger:Error("VIRE cloud layer data not found, aborting")
  return
end

-- Get rain data from VIRE
rr_param = param("RRR-KGM2", aggregation(HPAggregationType.kAccumulation, time_duration("01:00")), processing_type())
local RR_VIRE = luatool:Fetch(current_time, current_level, rr_param, current_forecast_type)


-- By default uses MEPS and EC data before time step 66. After that, only EC data is used. MEPS can be disabled by setting the configuration parameter "disable_meps" to true.
if disable_meps or meps_step > 66 then
  logger:Info("Only using EC data")
  
  if not NL_EC or not NM_EC or not NH_EC or not NL_MEAN_EC or not NM_MEAN_EC or not NH_MEAN_EC or not NL_STD_EC or not NM_STD_EC or not NH_STD_EC then
    logger:Error("Some EC data not found, aborting")
    return
  else
    logger:Info("EC Data fetched")
  end

  local wt_sum = wt_ens_ec + wt_ec
  for i = 1, #NL_EC do

    mean, stDev = compute_std_mean_ec(NL_EC[i], NL_MEAN_EC[i], NL_STD_EC[i], wt_ec, wt_ens_ec, wt_sum)
    cl[i] = mean + stDev ^ dev_factor_low * (mean - mid_level) / 50  -- start from the weighted mean and push values away from the midpoint based on the standard deviation. dev_factor controls the volume of the push.

    mean, stDev = compute_std_mean_ec(NM_EC[i], NM_MEAN_EC[i], NM_STD_EC[i], wt_ec, wt_ens_ec, wt_sum)
    cm[i] = mean + stDev ^ dev_factor_mid * (mean - mid_level) / 50

    mean, stDev = compute_std_mean_ec(NH_EC[i], NH_MEAN_EC[i], NH_STD_EC[i], wt_ec, wt_ens_ec, wt_sum)
    ch[i] = mean + stDev ^ dev_factor_high * (mean - mid_level) / 50
  end
else
  logger:Info("Using MEPS and EC data")

  if not NL_EC or not NM_EC or not NH_EC or not NL_MEPS or not NM_MEPS or not NH_MEPS or not NL_MEAN_EC or not NM_MEAN_EC or not NH_MEAN_EC or not NL_MEAN_MEPS or not NM_MEAN_MEPS or not NH_MEAN_MEPS or not NL_STD_EC or not NM_STD_EC or not NH_STD_EC or not NL_STD_MEPS or not NM_STD_MEPS or not NH_STD_MEPS then
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

-- Limit cloud layers are between 0 and 100.
-- Harmonize cloud layers with total cloud cover. The idea is that at each grid point, total cloudiness can be directly computed as a function of the cl, cm, and ch parameters.
if cl and cm and ch then
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

  -- Correction using the cloud consensus data
  NL_VIRE, NM_VIRE, NH_VIRE = correct_cloudlayers(n, NL_VIRE, NM_VIRE, NH_VIRE)
  N_VIRE = rain_correction(RR_VIRE, n)

  -- Limit cloud values to 0-1
  NL_VIRE = limit_values(NL_VIRE)
  NM_VIRE = limit_values(NM_VIRE)
  NH_VIRE = limit_values(NH_VIRE)
  N_VIRE = limit_values(N_VIRE)

  -- Write all parameters to file
  write_options.replace_cache = true
  write_results_to_file(param("NL-0TO1"), NL_VIRE)
  write_results_to_file(param("NM-0TO1"), NM_VIRE)
  write_results_to_file(param("NH-0TO1"), NH_VIRE)
  
  write_results_to_file(param("N-0TO1"), N_VIRE)
end

