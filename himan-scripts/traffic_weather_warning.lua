logger:Info("Alku: traffic warning calculation")

function min_max_mean(tbl)

  local min = tbl[1]
  local max = tbl[1]
  local sum = 0

  for i = 1, #tbl do
      if tbl[i] > max then
          max = tbl[i]
      end
      if tbl[i] < min then
          min = tbl[i]
      end
      sum = sum + tbl[i]
  end
  local mean = sum / #tbl
  return tostring(min).. ' ' .. tostring(max).. ' '.. tostring(mean)
end

function warning_reason(warning, ice_warning, frost)
  local reason = {}
  for i=1, #warning do
    reason[i] = 0 
    reason[i] = warning[i] >= frost[i] and warning[i] or reason[i]
    reason[i] = warning[i] >= ice_warning[i] and warning[i] or reason[i]
    reason[i] = frost[i] >= warning[i] and frost[i] or reason[i]
    reason[i] = frost[i] >= ice_warning[i] and frost[i] or reason[i]
    reason[i] = ice_warning[i] >= warning[i] and ice_warning[i] or reason[i]
    reason[i] = ice_warning[i] >= frost[i] and ice_warning[i] or reason[i]
  end
  return reason
end


function combine_warning_factors(warning, thawing, freezing, wind, radglo_w)
  warning = {}
  for i=1, #thawing do
    warning[i] = thawing[i] * freezing[i] * wind[i] * radglo_w[i]
  end
  return warning
end

function frost_effect(nlm, tk, tdk, ff, radglo3)
  local frost, c_f, t_f, tdk_f, w_f, r_f, tc, tdc = {}, {}, {}, {}, {}, {}, {}, {}

  for i = 1, #nlm do
    tc[i] = tk[i] - 273.15
    tdc[i] = tdk[i] - 273.15

    c_f[i] = (125 - nlm[i])/100

    t_f[i] = (-0.25)*tc[i] + 1.25 < 0 and 0 or (-0.25)*tc[i] + 1.25
    t_f[i] = t_f[i] > 1 and 1 or t_f[i]
    
    tdk_f[i] = (-0.5) * (tc[i] - tdc[i]) + 2 < 0 and 0 or (-0.5) * (tc[i] - tdc[i]) + 2
    tdk_f[i] = tdk_f[i] > 2 and 2 or tdk_f[i]

    w_f[i] = 0.125 * ff[i] + 1 < 0.5 and 0.5 or 0.125 * ff[i] + 1
    w_f[i] = w_f[i] > 2 and 2 or w_f[i]

    r_f[i] = (-0.01) * radglo3[i] + 1 < 0 and 0 or (-0.01) * radglo3[i] + 1
    r_f[i] = r_f[i] > 1 and 1 or r_f[i]

    frost[i] = (r_f[i] * c_f[i] * t_f[i] * tdk_f[i] * w_f[i]) - 1
    frost[i] = frost[i] < 0 and 0 or frost[i]
    frost[i] = frost[i] > 2 and 2 or frost[i]

  end
  return frost
end
 
  
function freezing_rain_effect(rr, prec, tk)
  local freezing_rr_effect = {}
  for i = 1, #rr do
    if prec[i] == 4 or prec[i] == 5 then
      rr[i] = rr[i]
    else
      rr[i] = 0
    end
    celcius = tk[i] - 273.15
    freezing_rr_effect[i] = (0.189869351*rr[i]^3-1.151769668*rr[i]^2+2.193555856*rr[i]+0.002354682)*(-0.0045696099*celcius^3-0.0556120065*celcius^2-0.2579134548*celcius+0.7711723896)
    freezing_rr_effect[i] = freezing_rr_effect[i] < 0 and 0 or freezing_rr_effect[i]
    freezing_rr_effect[i] = freezing_rr_effect[i] > 2 and 2 or freezing_rr_effect[i]
  end
  return freezing_rr_effect
end 

function rad_effect(radglo)
  local rad_effect = {}
  for i=1, #radglo do
    rad_effect[i] = (radglo[i] - 275)/-250
    rad_effect[i] = rad_effect[i] < 0 and 0 or rad_effect[i]
    rad_effect[i] = rad_effect[i] > 1 and 1 or rad_effect[i]
  end
  return rad_effect
end

function wind_effect(ff, snr3, snr6)
  local wind = {}
  for i=1, #ff do
    wind[i] = (ff[i] + 10) / 16
    wind[i] = wind[i] > 1.5 and 1.5 or wind[i]
    wind[i] = wind[i] < 1 and 1 or wind[i]
    if snr3[i] < 0.5 or snr6[i] < 0.5 then
      wind[i] = 1
    end
  end
  return wind
end

function thawing_effect(tk)
  local thawing = {}
  for i = 1, #tk do
    celcius = tk[i] - 273.15
    thawing[i] =  -0.016314901*celcius^5 + 0.062931804*celcius^4 + 0.085471305*celcius^3 - 0.416382309*celcius^2 - 0.312297359*celcius + 1.238933192
    thawing[i] = celcius > 3 and 0 or thawing[i] 
    thawing[i] = celcius < -1.3 and 1 or thawing[i] 
  end
  return thawing
end

function freezing_effect(tk)
  local freezing = {}
  for i = 1, #tk do
    celcius = tk[i] - 273.15
    freezing[i] = (celcius - 1)/(-8)
    freezing[i] = freezing[i] > 2 and 2 or freezing[i]
    freezing[i] = freezing[i] < 1 and 1 or freezing[i]
  end
  return freezing
end

function init_snow_acc_values(snr3, snr6, snr12)
  snr3_init, snr6_init, snr12_init = {}, {}, {}
  for i = 1, #snr3 do
    snr3_init[i] = 0.66 * snr3[i] > 2 and 2 or 0.66 * snr3[i]
    snr6_init[i] = 0.4 * snr6[i] > 2 and 2 or 0.4 * snr6[i]
    snr12_init[i] = 0.12 * snr12[i] 
  end
  return snr3_init, snr6_init, snr12_init
end

function set_warning(snr3, snr6, snr12)
  local warning = {}
  for i = 1, #snr3 do
    math.max(snr3[i], snr6[i], snr12[i])
  end 
  return warning
end

function zero_array(size)
  local arr = {}
  for i = 1, size do
    arr[i] = 0
  end
  return arr
end

function print_parameters(param_names, param_data)
  for i = 1, #param_names do
    print(param_names[i] .. ': ' .. min_max_mean(param_data[i]))
  end
end

function adjust_time(origin_time)
  local adjust_time = raw_time(origin_time:String("%Y-%m-%d %H:%M%S"))
  hour = tonumber(adjust_time:String("%H"))
  adjust_time:Adjust(HPTimeResolution.kHourResolution, -math.fmod(hour, 3))

  return adjust_time
end 

function get_prod_time(producer_name, producer_id)

  if configuration:Exists("ecmwf_origintime") then
    logger:Debug("Using hard-coded origintime")

    if producer_name == 'ECGMTA' then
      return forecast_time(raw_time(configuration:GetValue("ecmwf_origintime")), current_time:GetValidDateTime())
    elseif producer_name == 'MEPSMTA' then
      return forecast_time(raw_time(configuration:GetValue("meps_origintime")), current_time:GetValidDateTime())
    else
      return forecast_time(raw_time(configuration:GetValue("vire_origintime")), current_time:GetValidDateTime())
    end
  else 
    local prod = producer(producer_id, producer_name)
    local prod_origin = raw_time(radon:GetLatestTime(prod, "",0))

    if producer_name == 'MEPSMTA' then
      prod_origin = adjust_time(prod_origin)
    end 

    local prod_time = forecast_time(prod_origin, current_time:GetValidDateTime())
    return prod_time
  end
end

function move_valid_time(time, hours)
  local adjusted_time = forecast_time(time)
  adjusted_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, hours)
  return adjusted_time
end

function fetch_parameter(param_name, level_height, prod_time, ftype)
  local o = {forecast_time = prod_time,
          level = level(HPLevelType.kHeight, level_height),
          param = param(param_name),
          forecast_type = ftype,
          read_previous_forecast_if_not_found = false
  }

  data = luatool:FetchWithArgs(o)
  return data
end

function fetch_radiation_data(step, level_height, prod_time, ftype, is_sum)
  local o = {
    forecast_time = prod_time,
    level = level(HPLevelType.kHeight, level_height),
    forecast_type = ftype
  }

  local agg_duration, multiplier
  if step < 83 then
    agg_duration = time_duration("01:00")
    multiplier = 1
  elseif step < 137 then
    agg_duration = time_duration("03:00")
    multiplier = 3
  else
    agg_duration = time_duration("06:00")
    multiplier = 3
  end

  o.param = param('RADGLO-WM2', aggregation(HPAggregationType.kAverage, agg_duration), processing_type())

  -- Fetch radiation data
  if is_sum and step < 83 then
    -- Fetch for the current and two prior time steps for summing
    local radglo_sum = fetch_multiple_radiation(o, prod_time)
    return radglo_sum
  else
    -- Fetch single time step data and apply multiplier if needed
    local radglo_data = luatool:FetchWithArgs(o)
    if is_sum then
      for i = 1, #radglo_data do
        radglo_data[i] = multiplier * radglo_data[i]
      end
    end
    return radglo_data
  end
end

-- Helper function to fetch data for multiple time steps and sum them
function fetch_multiple_radiation(o, prod_time)
  local my_time = forecast_time(prod_time)  

  local radglo_sum = {}
  for j = 0, 2 do
    if j > 0 then
      my_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)
    end
    o.forecast_time = my_time
    local radglo_data = luatool:FetchWithArgs(o)
    for i = 1, #radglo_data do
      radglo_sum[i] = (radglo_sum[i] or 0) + radglo_data[i]
    end
  end
  return radglo_sum
end


ftype_d = forecast_type(HPForecastType.kDeterministic)
ftype_c = forecast_type(HPForecastType.kEpsControl, 0)

vire_time = get_prod_time('VIRE_PREOP', 287)
ecmwf_time = get_prod_time('ECGMTA', 240)
meps_time = get_prod_time('MEPSMTA', 260)

-- if past 66 hours, priority 2 data is ec, not meps
meps_step = tonumber(meps_time:GetStep():Hours())
pri2_time = nil
pri2_ftype = nil

if meps_step < 66 then 
  pri2_time = meps_time
  pri2_ftype = ftype_c
else 
  pri2_time = ecmwf_time
  pri2_ftype = ftype_d
end

tk = fetch_parameter('T-K', 2, vire_time, ftype_d)
ff = fetch_parameter('FF-MS', 10, vire_time, ftype_d)
nlm = fetch_parameter('NLM-PRCNT', 0, vire_time, ftype_d)
prec = fetch_parameter('PRECFORM2-N', 0, vire_time, ftype_d)
tdk = fetch_parameter('TD-K', 2, vire_time, ftype_d)

radglo1 = fetch_radiation_data(meps_step, 0, pri2_time, pri2_ftype, false)
radglo3 = fetch_radiation_data(meps_step, 0, pri2_time, pri2_ftype, true)

ec_step = tonumber(ecmwf_time:GetStep():Hours())

-- if ec step over 85, the steps increases and validation time can't be shifted
if ec_step < 85 then
  rr = fetch_parameter('RR-3-MM', 0, move_valid_time(pri2_time, 1), pri2_ftype)
  snr3 = fetch_parameter('SNR-KGM2', 0, move_valid_time(pri2_time, 1), pri2_ftype)
  snr6 = fetch_parameter('SN-6-MM', 0, move_valid_time(pri2_time, 2), pri2_ftype)
  snr12 = fetch_parameter('SN-12-MM', 0, move_valid_time(pri2_time, 5), pri2_ftype)
else 
  rr = fetch_parameter('RR-3-MM', 0, pri2_time, pri2_ftype)
  snr3 = fetch_parameter('SNR-KGM2', 0, pri2_time, pri2_ftype)
  snr6 = zero_array(#snr3)
  snr12 = zero_array(#snr3)
end

--set initial values for warning calculation
snr3, snr6, snr12 = init_snow_acc_values(snr3, snr6, snr12)

-- set biggest value as warning
warning = set_warning(snr3, snr6, snr12)

-- the effect of extreme freezing temperatures to the warnings
freezing = freezing_effect(tk)

-- the effect of thawing to the warnings
thawing = thawing_effect(tk)

-- wind effect to the warnings with snow
wind = wind_effect(ff, snr3, snr6)

-- -- radiation affect to the warnings
radglo_w = rad_effect(radglo1)

-- -- freezing rain and rain combination effect to the warnings
ice_warning = freezing_rain_effect(rr, prec, tk)

-- frost affect to warnings
frost = frost_effect(nlm, tk, tdk, ff, radglo3)

-- checking the most effective factor for the warning
warning = combine_warning_factors(warning, thawing, freezing, wind, radglo_w)
reason = warning_reason(warning, ice_warning, frost)


result:SetParam(param('TWW-N'))
result:SetValues(reason)

luatool:WriteToFile(result)


