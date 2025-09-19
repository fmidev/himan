-- Precipitation form determination, that takes the Vire temperature better into consideration.
-- Using the 2 meter wet bulb temperature, which already a good approximation for determining precipitation form, 
-- but this script combines it with model level based determination.
--
-- Original code from SmartTool by Jani Sorsa and Sini Jääskeläinen.
-- Some comments copied and translated from original SmartTool code.



-- Adjustments for origin times
local HOUR_TO_ADJUST = {
    MEPS = {  -- hours 0..23
        [0]=-3,[1]=-4,[2]=-2,[3]=-3,[4]=-4,[5]=-2,[6]=-3,[7]=-4,[8]=-2,[9]=-3,[10]=-4,[11]=-2,
        [12]=-3,[13]=-4,[14]=-2,[15]=-3,[16]=-4,[17]=-2,[18]=-3,[19]=-4,[20]=-2,[21]=-3,[22]=-4,[23]=-2,
    },
    EC = {
        [0]  = -12, [1]  = -13, [2]  = -14, [3]  = -15, [4]  = -16, [5]  = -17,
        [6]  = -18, [7]  = -7,  [8]  = -8,  [9]  = -9,  [10] = -10, [11] = -11,
        [12] = -12, [13] = -13, [14] = -14, [15] = -15, [16] = -16, [17] = -17,
        [18] = -18, [19] = -7,  [20] = -8,  [21] = -9,  [22] = -10, [23] = -11,
    }    
}

local PRODUCER_GROUP = {
    [4]   = "MEPS",
    [260] = "MEPS",
    [131] = "EC",
    [240] = "EC",
}

function adjust_precip(prec_value, t_value)
    -- Drizzle is left as determined by the model, except when the temperature is below freezing 
    -- (in which case it is snowfall).
    if prec_value == 0 then
        if t_value > 0 then return 3 end
    -- Freezing drizzle is left as determined by the model, except if the temperature is above freezing 
    -- (in which case it is liquid drizzle) or below –10 °C (in which case it is snowfall)
    elseif prec_value == 4 then
        if t_value > 0 then return 0 end
        if t_value < -10 then return 3 end
    --Freezing rain is left as determined by the model, except if the temperature is above freezing 
    -- (in which case it is rain) or below –10 °C (in which case it is snowfall).
    elseif prec_value == 5 then
        if t_value > 0 then return 1 end
        if t_value < -10 then return 3 end
    end
    return prec_value
end

function convert_to_100(array)
    if not array then return nil end 
    local t = {}
    for i = 1, #array do
        t[i] = array[i] * 100
    end
    return t
  end

function get_test_time()
    local test_time = configuration:GetValue("origin_time_test")
    local step_time = configuration:GetValue("step_time_test")
    ftime = forecast_time(raw_time(test_time), raw_time("2025-06-16 22:00:00"))
    return ftime
end

function get_time(producer) 
    test = configuration:Exists("origin_time_test")
    if test then
        return get_test_time()
    end

    local origin_dt = current_time:GetOriginDateTime()
    local hour      = tonumber(origin_dt:String('%H'))
    local pid       = producer:GetId()

    local group = PRODUCER_GROUP[pid]
    if not group then return end

    local adjust_hours = HOUR_TO_ADJUST[group][hour]
    if not adjust_hours then return end

    local ftime = forecast_time(origin_dt, current_time:GetValidDateTime())
    ftime:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjust_hours)
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

rh_meps = convert_to_100(rh_meps) --variable RH-PRCNT not found, using RH-0TO1

local tw = {}
local tw_meps = {}
local tw_ec = {}

local ec_diff = {}
local meps_diff = {}

-- A variable for temperature determination based on the wet bulb temperature.
local tw_pref = {}

-- Computational temperature thresholds used by the wet bulb–based form determination.
-- The values are in principle adjustable, but these have seemed to work reasonably well.
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

    -- Wet bulb temperatures according to the empirical formula (Stull) for VIRE data, EC, and MEPS. 
    -- The calculation requires T and RH. The Stull formula combines computational simplicity
    -- with relatively good accuracy under natural temperature conditions
    tw[i] = TwStull_(t[i], rh[i])
    tw[i] = tw[i] - 273.15

    tw_ec[i] = TwStull_(t_ec[i], rh_ec[i])
    tw_ec[i] = tw_ec[i] - 273.15

    if use_meps then
        tw_meps[i] = TwStull_(t_meps[i], rh_meps[i])
        tw_meps[i] = tw_meps[i] - 273.15
    end

    -- The absolute difference between the VIRE temperature and the raw temperature (EC, MEPS) is calculated. 
    -- This difference is later used to determine which form determination will be applied.
    ec_diff[i] = math.abs(tw_ec[i] - tw[i])

    if use_meps then
        meps_diff[i] = math.abs(tw_meps[i] - tw[i])
    end

    tw_pref[i] = missing

    -- Above the rainbound, the form according to wet bulb determination is rain;
    -- below the snowbound, it is snow; and between the two, it is sleet.
    if tw[i] > rainbound then
        tw_pref[i] = 1
    end
    if ((tw[i] > snowbound) and (tw[i] <= rainbound)) then
        tw_pref[i] = 2
    end
    if tw[i] <= snowbound then
        tw_pref[i] = 3
    end

    -- The actual logic of the form determination (rain, sleet, snow) is implemented here.
    -- Uses MEPS, if available and not disabled, otherwise EC.
    -- The wet bulb–based form is chosen when the model’s TW deviates significantly from the Vire data TW.
    -- With a difference greater than 0.5 °C, the TW based form is always chosen;
    -- with a difference smaller than 0.25 °C, the model’s form is chosen. Between these, a (gradually) random form is selected.
    -- If the model’s form and TW based form are contradictory (rain and snow), sleet is chosen.
    if prec[i] > 0 and prec[i] < 4 then
        local diff = use_meps and meps_diff[i] or ec_diff[i]
        local prec_ref = use_meps and prec_meps[i] or prec_ec[i]
    
        if math.random() < RampUp(0.25, 0.5, diff) then
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


