function adjust_precip(prec_value, t_value)
    if prec_value == 0 then
        if t_value > 0 then return 3 end
    elseif prec_value == 4 then
        if t_value > 0 then return 0 end
        if t_value < -10 then return 3 end
    elseif prec_value == 5 then
        if t_value > 0 then return 1 end
        if t_value < -10 then return 3 end
    end
    return prec_value
end

function compute_tw(temp, rh)
    local rad = math.pi / 180
    return temp * math.atan(0.151977 * math.sqrt(rh + 8.313659)) * rad
         + math.atan(temp + rh) * rad
         - math.atan(rh - 1.676331) * rad
         + 0.00391838 * (rh^(3/2)) * math.atan(0.023101 * rh) * rad
         - 4.686035
end

function ramp_up(value, lower, upper)
    if value < lower then
        return 0
    elseif value > upper then
        return 1
    else
        return 1 - (value - lower) / (upper - lower)
    end
end

function convert_to_100(array)
    if not array then return nil end 
    local t = {}
    for i = 1, #array do
        t[i] = array[i] * 100
    end
    return t
  end

function convert_to_celsius(array)
    if not array then return nil end 
    local t = {}
    for i = 1, #array do
        t[i] = array[i] - 273.15
    end
    return t
end

function get_time(producer)  

      
    local test = configuration:Exists("origin_time_test")

    if test then
        local test_time = configuration:GetValue("origin_time_test")
        ftime = forecast_time(raw_time(test_time), raw_time("2025-06-16 22:00:00"))
        return ftime
    end
    
    local vire_hour = current_time:GetOriginDateTime():String('%H')
    local producer_id = producer:GetId()
  
    if producer_id == 4 or producer_id == 260 then
      adjust_hours = -4
    elseif (vire_hour == '07' or vire_hour == '19') and (producer_id == 131 or producer_id == 240) then
        adjust_hours = -7
    elseif (vire_hour == '13' or vire_hour == '01') and (producer_id == 131 or producer_id == 240) then
        adjust_hours = -13
    end
  
    if adjust_hours then
      current_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjust_hours)
    end
    
    ftime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())
    current_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, -adjust_hours) 
    return ftime
  end

function get_param(producer, ftype, param, level)

    local ftime = get_time(producer)
  
    local o = {forecast_time = ftime,
    level = level,
    param = param,
    forecast_type = ftype,
    producer = producer}

    local param1 = luatool:FetchWithArgs(o)

    return param1
  end


local par_t = param('T-K')
local par_rh = param('RH-PRCNT')
local par_prec = param('PRECFORM2-N')

local l2 = level(HPLevelType.kHeight, 2)
local l0 = level(HPLevelType.kHeight, 0)

local t = luatool:Fetch(current_time, l2, par_t, current_forecast_type)
local rh = luatool:Fetch(current_time, l2, par_rh, current_forecast_type)
local prec = luatool:Fetch(current_time, l0, par_prec, current_forecast_type)

local meps = producer(4, "MEPS")
local meps_mta = producer(260, "MEPSMTA")
local meps_ftype = forecast_type(HPForecastType.kEpsControl, 0)

local t_meps = get_param(meps, meps_ftype, par_t, l2)
local rh_meps = get_param(meps, meps_ftype, param("RH-0TO1"), l2)
local prec_meps = get_param(meps_mta, meps_ftype, par_prec, l0)

local ec_mta = producer(240, "ECGMTA")
local ec = producer(131, "ECG")
local ec_ftype = forecast_type(HPForecastType.kDeterministic)

local t_ec = get_param(ec, ec_ftype, par_t, l2)
local rh_ec = get_param(ec_mta, ec_ftype, par_rh, l2)
local prec_ec = get_param(ec_mta, ec_ftype, par_prec, l0)

t = convert_to_celsius(t)
t_meps = convert_to_celsius(t_meps)
t_ec = convert_to_celsius(t_ec)

rh_meps = convert_to_100(rh_meps) --variable RH-PRCNT not found, using RH-0TO1

local tw = {}
local tw_meps = {}
local tw_ec = {}

local ec_diff = {}
local meps_diff = {}

local tw_pref = {}

local rainbound = 0.4
local snowbound = 0.1

local disable_meps = nil
if configuration:Exists("disable_meps") then 
    disable_meps = ParseBoolean(configuration:GetValue("disable_meps"))
end

meps_time = get_time(meps)
meps_step = tonumber(meps_time:GetStep():Hours())

local use_meps = (disable_meps == false and meps_step < 66)

for i = 1, #t do

    tw[i] = compute_tw(t[i], rh[i])
    tw_ec[i] = compute_tw(t_ec[i], rh_ec[i])
    
    if use_meps then
        tw_meps[i] = compute_tw(t_meps[i], rh_meps[i])
    end

    ec_diff[i] = math.abs(tw_ec[i] - tw[i])

    if use_meps then
        meps_diff[i] = math.abs(tw_meps[i] - tw[i])
    end

    tw_pref[i] = missing

    if tw[i] > rainbound then
        tw_pref[i] = 1
    end
    if ((tw[i] > snowbound) and (tw[i] <= rainbound)) then
        tw_pref[i] = 2
    end
    if tw[i] <= snowbound then
        tw_pref[i] = 3
    end

    if prec[i] > 0 and prec[i] < 4 then
        local diff = use_meps and meps_diff[i] or ec_diff[i]
        local prec_ref = use_meps and prec_meps[i] or prec_ec[i]
    
        if math.random() < ramp_up(diff, 0.25, 0.5) then
            prec[i] = tw_pref[i]
        else
            if (tw_pref[i] == 3 and prec_ref == 1) or (tw_pref[i] == 1 and prec_ref == 3) then
                prec[i] = 2
            else
                prec[i] = prec_ref
            end
        end
    end    

    prec[i] = adjust_precip(prec[i], t[i])
end

result:SetParam(param('PRECFORM2-N'))
result:SetValues(prec)

luatool:WriteToFile(result)


