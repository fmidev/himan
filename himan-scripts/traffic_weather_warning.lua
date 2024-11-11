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
    if warning[i] >= frost[i] then
      reason[i] = warning[i]
    end
    if warning[i] >= ice_warning[i] then
      reason[i] = warning[i]
    end
    if frost[i] >= warning[i] then
      reason[i] = frost[i]
    end 
    if frost[i] >= ice_warning[i] then
      reason[i] = frost[i]
    end
    if ice_warning[i] >= warning[i] then
      reason[i] = ice_warning[i]
    end
    if ice_warning[i] >= frost[i] then
      reason[i] = ice_warning[i]
    end
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

function frost_factors(rad_f, cloudiness_f, temperature_f, td_t_dif, wind_f)
  local frost = {}
  for i = 1, #rad_f do
    frost[i] = (rad_f[i] * cloudiness_f[i] * temperature_f[i] * td_t_dif[i] * wind_f[i]) - 1
  end
  return frost
end

function rad_effect_frost(radglo3)
  local rad_f = {}
  for i = 1, #radglo3 do
    rad_f[i] = (-0.01) * radglo3[i] + 1
  end
  return rad_f
end

function wind_effect_frost(ff)
  local wind_f = {}
  for i = 1, #ff do
    wind_f[i] = 0.125 * ff[i] + 1
  end
  return wind_f
end 

function difference_in_temperature(tdk, tk)
  local td_t_dif = {}
  for i = 1, #tdk do
    td_t_dif[i] = (-0.5) * (tdk[i] - tk[i]) + 2
  end
  return td_t_dif
end

function temperatue_effect_frost(tc)
  local temperature = {}
  for i = 1, #tc do
    temperature[i] = (-0.25)*tc[i] + 1.25
  end
  return temperature
end

function cloud_effect_frost(nlm)
  local cloudiness = {}
  for i = 1, #nlm do
    cloudiness[i] = (125 - nlm[i])/100
  end
  return cloudiness
end

function freezing_rain_effect(freezing_rr, tc)
  local freezing_rr_effect = {}
  for i = 1, #freezing_rr do
    freezing_rr_effect[i] = (0.189869351*freezing_rr[i]^3-1.151769668*freezing_rr[i]^2+2.193555856*freezing_rr[i]+0.002354682)*(-0.0045696099*tc[i]^3-0.0556120065*tc[i]^2-0.2579134548*tc[i]+0.7711723896)
  end
  return freezing_rr_effect
end 

function freezing_rain(rr, prec)
  local freezing_rr = {}
  for i = 1, #rr do
    if prec[i] == 4 or prec[i] == 5 then
      freezing_rr[i] = rr[i]
    else
      freezing_rr[i] = 0
    end
  end
  return freezing_rr
end

function rad_effect(radglo)
  local rad_effect = {}
  for i=1, #radglo do
    rad_effect[i] = (radglo[i] - 275)/-250
  end
  return rad_effect
end

function wind_snow(snr3, snr6, wind)
  for i = 1, #snr3 do
    if snr3[i] < 0.5 or snr6[i] < 0.5 then
      wind[i] = 1
    end
  end
  return wind
end

function wind_effect(ff)
  local wind = {}
  for i=1, #ff do
    wind[i] = (ff[i] + 10) / 16
  end
  return wind
end

function thawing_effect(celcius)
  local thawing = {}
  for i = 1, #celcius do
    thawing[i] =  -0.016314901*celcius[i]^5 + 0.062931804*celcius[i]^4 + 0.085471305*celcius[i]^3 - 0.416382309*celcius[i]^2 - 0.312297359*celcius[i] + 1.238933192
  end
  return thawing
end

function freezing_effect(celcius)
  local freezing = {}
  for i = 1, #celcius do
    freezing[i] = (celcius[i])/(-8)
  end
  return freezing
end

function kelvin_to_celsius(kelvin)
  local celcius = {}
  for i = 1, #kelvin do
    celcius[i] = kelvin[i] - 273.15
  end
  return celcius
end

function set_factor_to_array(factor, array)
  for i = 1, #array do
    array[i] = array[i] * factor
  end
  return array
end

function set_warning(snr3, snr6, snr12)
  local warning = {}
  for i = 1, #snr3 do
    if snr3[i] >= snr6[i] then 
      warning[i] = snr3[i]
    elseif snr6[i] >= snr3[i] then 
      warning[i] = snr6[i]
    elseif snr12[i] > snr3 and snr12[i] > snr6 then
      warning[i] = snr12[i]
    end
  end 
  return warning
end

function lessthan_threshold(array, threshold, new_value)
  for i = 1, #array do
    if array[i] < threshold then
      array[i] = new_value
    end
  end
  return array
end

function greater_threshold(array, threshold, new_value)

  for i = 1, #array do
    if array[i] > threshold then
      array[i] = new_value
    end
  end
  return array
end

function print_parameters(param_names, param_data)
  for i = 1, #param_names do
    print(param_names[i] .. ': ' .. min_max_mean(param_data[i]))
  end
  print('Loppu')
end

function adjust_time(origin_time)
  hour = tonumber(origin_time:String("%H"))
  print('hour: ' .. hour)
  print('origin time: ' .. origin_time:String("%Y-%m-%d %H:%M"))

  if hour == 1 or hour == 4 or hour == 7 or hour == 10 or hour == 13 or hour == 16 or hour == 19 or hour == 22 then
    origin_time:Adjust(HPTimeResolution.kHourResolution, -1)
  elseif hour == 2 or hour == 5 or hour == 8 or hour == 11 or hour == 14 or hour == 17 or hour == 20 or hour == 23 then
    origin_time:Adjust(HPTimeResolution.kHourResolution, -2)
  end
  return origin_time
end 

function get_prod_time(producer_name, producer_id)
  
  local prod = producer(producer_id, producer_name)
  local prod_origin = raw_time(radon:GetLatestTime(prod, "",0))

  if producer_name == 'MEPSMTA' then
    print('MEPSMTA laskentaa')
    prod_origin = adjust_time(prod_origin)
    print('adjusted time: ' .. prod_origin:String("%Y-%m-%d %H:%M"))
  end 

  local prod_time = forecast_time(prod_origin, current_time:GetValidDateTime())
  return prod_time
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

  print('summaus '.. tostring(is_sum) .. ' ja ' .. step)
  -- Fetch radiation data
  if is_sum and step < 83 then
    print('Siirryttiin summaamiseen')
    -- Fetch for the current and two prior time steps for summing
    local radglo_sum = fetch_multiple_radiation(o, prod_time, 3)
    return radglo_sum
  else
    -- Fetch single time step data and apply multiplier if needed
    local radglo_data = luatool:FetchWithArgs(o)
    -- if radglo_data ~= nil then
    --   print('dataa saatu. MEPSSTEP: ' .. step .. 'Vire-STEP: '.. vire_time:GetStep():Hours() .. 'EC-STEP: '.. ecmwf_time:GetStep():Hours() ..', EC-AIKA: ' .. prod_time:GetOriginDateTime():String("%Y-%m-%d %H:%M") .. ', MEPS: ' .. meps_time:GetOriginDateTime():String("%Y-%m-%d %H:%M") .. ', VIRE: ' .. vire_time:GetOriginDateTime():String("%Y-%m-%d %H:%M") .. 'valid time: ' .. prod_time:GetValidDateTime():String("%Y-%m-%d %H:%M") .. vire_time:GetValidDateTime():String("%Y-%m-%d %H:%M") .. ecmwf_time:GetValidDateTime():String("%Y-%m-%d %H:%M") .. meps_time:GetValidDateTime():String("%Y-%m-%d %H:%M"))
    -- end 
    if is_sum then
      for i = 1, #radglo_data do
        radglo_data[i] = multiplier * radglo_data[i]
      end
    end
    return radglo_data
  end
end

-- Helper function to fetch data for multiple time steps and sum them
function fetch_multiple_radiation(o, prod_time, count)
  local radglo_sum = {}
  for j = 0, count - 1 do
    if j > 0 then
      prod_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)
    end
    o.forecast_time = prod_time
    local radglo_data = luatool:FetchWithArgs(o)
    for i = 1, #radglo_data do
      radglo_sum[i] = (radglo_sum[i] or 0) + radglo_data[i]
    end
  end
  -- Reset the time back to original
  prod_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, count - 1)
  return radglo_sum
end




ftype_d = forecast_type(HPForecastType.kDeterministic)
ftype_c = forecast_type(HPForecastType.kEpsControl, 0)

vire_time = get_prod_time('VIRE_PREOP', 287)
ecmwf_time = get_prod_time('ECGMTA', 240)
meps_time = get_prod_time('MEPSMTA', 260)

-- vire_time = forecast_time(raw_time("2024-11-10 01:00:00"), current_time:GetValidDateTime())
-- ecmwf_time = forecast_time(raw_time("2024-11-10 00:00:00"), current_time:GetValidDateTime()) 
-- meps_time = forecast_time(raw_time("2024-11-08 03:00:00"), current_time:GetValidDateTime())

print('vire time: ' .. vire_time:GetOriginDateTime():String("%Y-%m-%d %H:%M"))
print('ecmwf time: ' .. ecmwf_time:GetOriginDateTime():String("%Y-%m-%d %H:%M"))
print('meps time: ' .. meps_time:GetOriginDateTime():String("%Y-%m-%d %H:%M"))


-- if past 66 hours, priority 2 data is ec, not meps
step = tonumber(meps_time:GetStep():Hours())
print('timestep from config: ' .. step)

pri2_time = nil
pri2_ftype = nil
if step < 66 then 
  print('MEPS data')
  pri2_time = meps_time
  pri2_ftype = ftype_c
else 
  print('EC data')
  pri2_time = ecmwf_time
  pri2_ftype = ftype_d
end

print('pri2time: ' .. pri2_time:GetOriginDateTime():String("%Y-%m-%d %H:%M"))

tk = fetch_parameter('T-K', 2, vire_time, ftype_d)
ff = fetch_parameter('FF-MS', 10, vire_time, ftype_d)
nlm = fetch_parameter('NLM-PRCNT', 0, vire_time, ftype_d)
prec = fetch_parameter('PRECFORM2-N', 0, vire_time, ftype_d)
tdk = fetch_parameter('TD-K', 2, vire_time, ftype_d)

radglo1 = fetch_radiation_data(step, 0, pri2_time, pri2_ftype, false)
radglo3 = fetch_radiation_data(step, 0, pri2_time, pri2_ftype, true)
rr = fetch_parameter('RR-3-MM', 0, pri2_time, pri2_ftype)
snr3 = fetch_parameter('SNR-KGM2', 0, pri2_time, pri2_ftype)
snr6 = fetch_parameter('SN-6-MM', 0, pri2_time, pri2_ftype)
snr12 = fetch_parameter('SN-12-MM', 0, pri2_time, pri2_ftype)

print_parameters({'tk', 'ff', 'nlm', 'prec', 'tdk', 'rr', 'snr3', 'snr6', 'snr12', 'radglo1', 'radglo3'}, 
              {tk, ff, nlm, prec, tdk, rr, snr3, snr6, snr12, radglo1, radglo3})

--set initial values for warning calculation
snr3 = set_factor_to_array(0.66, snr3)
snr6 = set_factor_to_array(0.4, snr6)
snr12 = set_factor_to_array(0.12, snr12)

-- set 2 as a limit for values, so that values won't increase excessively
snr3 = greater_threshold(snr3, 2, 2)
snr6 = greater_threshold(snr6, 2, 2)

-- set biggest value as warning
warning = set_warning(snr3, snr6, snr12)

-- the affect of extreme freezing temperatures to the warnings
tc = kelvin_to_celsius(tk)
freezing = freezing_effect(tc)
freezing = greater_threshold(freezing, 2, 2)
freezing = lessthan_threshold(freezing, 1, 1)

-- the affect of thawing to the warnings
thawing = thawing_effect(tc)
thawing = greater_threshold(thawing, 3, 0)
thawing = lessthan_threshold(thawing, -1.3, 1)

-- wind affect to the warnings
wind = wind_effect(ff)
wind = greater_threshold(wind, 1.5, 1.5)
wind = lessthan_threshold(wind, 1, 1)

-- wind doesn't have affect with out snow
wind = wind_snow(snr3, snr6, wind)

-- radiation affect to the warnings
radglo_w = rad_effect(radglo1)
radglo_w = greater_threshold(radglo_w, 1, 1)
radglo_w = lessthan_threshold(radglo_w, 0, 0)

-- freezing rain and rain combination effect to the warnings
freezing_rr3 = freezing_rain(rr, prec)
ice_warning = freezing_rain_effect(freezing_rr3, tc)
ice_warning = greater_threshold(ice_warning, 2, 2)
ice_warning = lessthan_threshold(ice_warning, 0, 0)

-- frost affect to warnings
cloudiness_f = cloud_effect_frost(nlm)
temperature_f = temperatue_effect_frost(tc)
td_t_dif = difference_in_temperature(kelvin_to_celsius(tdk), tk)
wind_f = wind_effect_frost(ff)
rad_f = rad_effect_frost(radglo3)

temperature_f = lessthan_threshold(temperature_f, 0, 0)
temperature_f = greater_threshold(temperature_f, 1, 1)

td_t_dif = lessthan_threshold(td_t_dif, 0, 0)
td_t_dif = greater_threshold(td_t_dif, 2, 2)

wind_f = lessthan_threshold(wind_f, 0.5, 0.5)
wind_f = greater_threshold(wind_f, 2, 2)

rad_f = lessthan_threshold(rad_f, 0, 0)
rad_f = greater_threshold(rad_f, 1, 1)

-- combine frost factors, if over 1, then bad weather starts from 1
frost = frost_factors(rad_f, cloudiness_f, temperature_f, td_t_dif, wind_f)
frost = lessthan_threshold(frost, 0, 0)
frost = greater_threshold(frost, 2, 2)

-- checking the most effective factor for the warning
warning = combine_warning_factors(warning, thawing, freezing, wind, radglo_w)
reason = warning_reason(warning, ice_warning, frost)

print_parameters({'snr3', 'snr6', 'snr12', 'warning', 'tc', 'freezing', 'thawing', 'wind', 'radglo_w', 'ice_warning', 'cloudiness', 'temperature', 'td_t_dif', 'wind_f', 'rad_f', 'reason'}, 
              {snr3, snr6, snr12, warning, tc, freezing, thawing, wind, radglo_w, ice_warning, cloudiness_f, temperature_f, td_t_dif, wind_f, rad_f, reason})


-- result:SetParam(param('TRAFFIC-WEATHER-WARNING'))
-- result:SetValues(reason)

-- luatool:WriteToFile(result)
              
print('Loppu')


