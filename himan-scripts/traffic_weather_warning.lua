logger:Info("Traffic warning calculation")


function warning_reason(warning, ice_warning, frost)
  local reason = {}
  for i=1, #warning do
    reason[i] = 0 
    reason[i] = warning[i] >= frost[i] and 2 or reason[i]
    reason[i] = warning[i] >= ice_warning[i] and 2 or reason[i]
    reason[i] = frost[i] >= warning[i] and 3 or reason[i]
    reason[i] = frost[i] >= ice_warning[i] and 3 or reason[i]
    reason[i] = ice_warning[i] >= warning[i] and 1 or reason[i]
    reason[i] = ice_warning[i] >= frost[i] and 1 or reason[i]
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
  local frost = {}
  local c_f, t_f, tdk_f, w_f, r_f, tc, tdc

  for i = 1, #nlm do
    tc = tk[i] - 273.15
    tdc = tdk[i] - 273.15

    c_f = (125 - nlm[i])/100

    t_f = (-0.25)*tc + 1.25 
    t_f = math.max(math.min(t_f, 1), 0)
    
    tdk_f = (-0.5) * (tc - tdc) + 2 
    tdk_f = math.max(math.min(tdk_f, 2), 0)

    w_f = 0.125 * ff[i] + 1
    w_f = math.max(math.min(w_f, 2), 0.5)

    r_f = (-0.01) * radglo3[i] + 1 
    r_f = math.max(math.min(r_f, 1), 0)

    frost[i] = (r_f * c_f * t_f * tdk_f * w_f) - 1
    frost[i] = math.max(math.min(frost[i], 2), 0)
  end
  return frost
end
 
  
function freezing_rain_effect(rr, prec, tk)
  local freezing_rr_effect = {}
  for i = 1, #rr do
    if prec[i] == 4 or prec[i] == 5 then
      celcius = tk[i] - 273.15
      freezing_rr_effect[i] = (0.189869351*rr[i]^3-1.151769668*rr[i]^2+2.193555856*rr[i]+0.002354682)*(-0.0045696099*celcius^3-0.0556120065*celcius^2-0.2579134548*celcius+0.7711723896)
      freezing_rr_effect[i] = freezing_rr_effect[i] < 0 and 0 or freezing_rr_effect[i]
      freezing_rr_effect[i] = freezing_rr_effect[i] > 2 and 2 or freezing_rr_effect[i]
    else
      freezing_rr_effect[i] = 0
    end
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

function adjust_time(origin_time)
  local adjust_time = raw_time(origin_time:String("%Y-%m-%d %H:%M%S"))
  hour = tonumber(adjust_time:String("%H"))
  adjust_time:Adjust(HPTimeResolution.kHourResolution, -math.fmod(hour, 3))

  return adjust_time
end 

function get_prod_time(producer_name, producer_id)

  if configuration:Exists("ecmwf_origintime") and configuration:Exists("meps_origintime") then
    logger:Debug("Using hard-coded origintime")

    if producer_name == 'ECGMTA' then
      return forecast_time(raw_time(configuration:GetValue("ecmwf_origintime")), current_time:GetValidDateTime())
    elseif producer_name == 'MEPSMTA' then
      return forecast_time(raw_time(configuration:GetValue("meps_origintime")), current_time:GetValidDateTime())
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
  local adjusted_time = forecast_time(time:GetOriginDateTime(), time:GetValidDateTime())
  return forecast_time(adjusted_time:GetOriginDateTime(), adjusted_time:GetValidDateTime() + time_duration(hours .. ":00:00"))
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

function fetch_radiation_interpolated(step, pri2_time, pri2_ftype)
  local origin_time = pri2_time:GetOriginDateTime()
  local valid_time = current_time:GetValidDateTime()


  if step < 83 then
    duration = time_duration("01:00")
  elseif step < 137 then
    duration = time_duration("03:00")
  else
    duration = time_duration("06:00")
  end

  local interpolated_radglo = {}
  for i=0,2 do
    if i > 0 then
      valid_time = valid_time - time_duration("01:00:00")
    end

    local p = param("RADGLO-WM2")
    p:SetAggregation(aggregation(HPAggregationType.kAverage, duration))
    local o = {
      forecast_time = forecast_time(origin_time, valid_time),
            level = current_level,
            param = p,
            forecast_type = pri2_ftype,
            read_previous_forecast_if_not_found = false,
            time_interpolation = true
    }
  
    local data = luatool:FetchWithArgs(o)

    interpolated_radglo[#interpolated_radglo + 1] = data
  end

  local sum_radglo = {}
  for i = 1, #interpolated_radglo[1] do
    sum_radglo[i] = (interpolated_radglo[1][i] + interpolated_radglo[2][i] + interpolated_radglo[3][i])
  end

  return interpolated_radglo[1], sum_radglo
end 


local ftype_d = forecast_type(HPForecastType.kDeterministic)
local ftype_c = forecast_type(HPForecastType.kEpsControl, 0)

local vire_time = current_time
local ecmwf_time = get_prod_time('ECGMTA', 240)
local meps_time = get_prod_time('MEPSMTA', 260)

-- if past 66 hours, priority 2 data is ec, not meps
local meps_step = tonumber(meps_time:GetStep():Hours())
local pri2_time = nil
local pri2_ftype = nil

if meps_step < 66 then 
  pri2_time = meps_time
  pri2_ftype = ftype_c
else 
  pri2_time = ecmwf_time
  pri2_ftype = ftype_d
end

local tk = fetch_parameter('T-K', 2, vire_time, ftype_d)
local ff = fetch_parameter('FF-MS', 10, vire_time, ftype_d)
local nlm = fetch_parameter('NLM-PRCNT', 0, vire_time, ftype_d)
local prec = fetch_parameter('PRECFORM2-N', 0, vire_time, ftype_d)
local tdk = fetch_parameter('TD-K', 2, vire_time, ftype_d)


local radglo1, radglo3 = fetch_radiation_interpolated(meps_step, pri2_time, pri2_ftype)

local ec_step = tonumber(ecmwf_time:GetStep():Hours())

-- if ec step over 85, the steps increases and validation time can't be shifted
local rr, snr3, snr6, snr12
if ec_step < 85 then
  rr = fetch_parameter('RR-3-MM', 0, move_valid_time(pri2_time, 1), pri2_ftype)
  snr3 = fetch_parameter('SNR-KGM2', 0, move_valid_time(pri2_time, 1), pri2_ftype)
  snr6 = fetch_parameter('SN-6-MM', 0, move_valid_time(pri2_time, 2), pri2_ftype)
  snr12 = fetch_parameter('SN-12-MM', 0, move_valid_time(pri2_time, 5), pri2_ftype)
elseif ec_step < 137 then 
  rr = fetch_parameter('RR-3-MM', 0, pri2_time, pri2_ftype)
  snr3 = fetch_parameter('SNR-KGM2', 0, pri2_time, pri2_ftype)
  snr6 = fetch_parameter('SN-6-MM', 0, move_valid_time(pri2_time, 3), pri2_ftype)
  snr12 = fetch_parameter('SN-12-MM', 0, move_valid_time(pri2_time, 9), pri2_ftype)
else
  rr = fetch_parameter('RR-3-MM', 0, pri2_time, pri2_ftype)
  snr3 = fetch_parameter('SNR-KGM2', 0, pri2_time, pri2_ftype)
  snr6 = fetch_parameter('SN-6-MM', 0, move_valid_time(pri2_time, 6), pri2_ftype)
  snr12 = fetch_parameter('SN-12-MM', 0, move_valid_time(pri2_time, 6), pri2_ftype)
end

--set initial values for warning calculation
snr3, snr6, snr12 = init_snow_acc_values(snr3, snr6, snr12)

-- set biggest value as warning
local warning = set_warning(snr3, snr6, snr12)

-- the effect of extreme freezing temperatures to the warnings
local freezing = freezing_effect(tk)

-- the effect of thawing to the warnings
local thawing = thawing_effect(tk)

-- wind effect to the warnings with snow
local wind = wind_effect(ff, snr3, snr6)

-- radiation affect to the warnings
local radglo_w = rad_effect(radglo1)

-- freezing rain and rain combination effect to the warnings
local ice_warning = freezing_rain_effect(rr, prec, tk)

-- frost affect to warnings
local frost = frost_effect(nlm, tk, tdk, ff, radglo3)

-- checking the most effective factor for the warning
warning = combine_warning_factors(warning, thawing, freezing, wind, radglo_w)
local reason = warning_reason(warning, ice_warning, frost)

result:SetParam(param('TRAFFIC-N'))
result:SetValues(reason)

luatool:WriteToFile(result)



