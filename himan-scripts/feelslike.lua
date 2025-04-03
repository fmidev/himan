-- FeelsLike temperature by Mika Heiskanen and Lea Saukkonen
-- Original c++ code:
-- https://github.com/fmidev/smartmet-library-newbase/blob/master/newbase/NFmiMetMath.cpp#L535
-- Comments copied from c++ code

local t_param = param('T-K')
local rh_param = param('RH-PRCNT')
local ws_param = param('FF-MS')
local l0 = level(HPLevelType.kHeight, 0)
local l2 = level(HPLevelType.kHeight, 2)
local l10 = level(HPLevelType.kHeight, 10)

function FmiSummerSimmerIndex(rh_arr, t_arr)
  local simmer_index = {}
  local simmer_limit = 14.5

  -- When in Finland and when > 14.5 degrees, 60% is approximately
  -- the minimum mean monthly humidity. However, Google wisdom
  -- claims most humans feel most comfortable either at 45%, or
  -- alternatively somewhere between 50-60%. Hence we choose
  -- the middle ground 50%

  local rh_ref = 0.5

  for i=1, #t_arr do
 
    local rh = rh_arr[i] / 100
    local t = t_arr[i] - kKelvin

    if t < simmer_limit then
      simmer_index[i] = t
    else
      simmer_index[i] = (1.8 * t - 0.55 * (1 - rh) * (1.8 * t - 26) - 0.55 * (1 - rh_ref) * 26) /
           (1.8 * (1 - 0.55 * (1 - rh_ref)))

    end
  end

  return simmer_index
end

function FetchRadiation(rad_prod, rad_arr_len)
  -- radiation is an optional, but important component of the feelslike temperature
  -- it is not used in the calculation if it is not available; in this case we create
  -- an array of missing values
  --
  -- we can specify that the radiation producer is ECMWF in the configuration file
  -- by setting the radiation_producer key to "ecmwf"
  -- otherwise we use the default producer ie the same as for the other parameters

  if rad_prod == "ecmwf" then
    rad_prod = producer(240, "ECGMTA")

    local latest_origintime = ""

    latest_origintime = raw_time(radon:GetLatestTime(rad_prod, "", 0))
    local ahour = tonumber(latest_origintime:String("%H"))

    -- we don't want to use ECMWF 06 or 18 UTC forecasts
    if ahour == 6 or ahour == 18 then
      latest_origintime:Adjust(HPTimeResolution.kHourResolution, -6)
    end

    logger:Info("Using ECMWF forecast from: " .. latest_origintime:String("%Y-%m-%d %H:%M:%S"))

    local latest_time = forecast_time(latest_origintime, current_time:GetValidDateTime())

    local step = latest_time:GetStep():Hours()

    -- need to get correct step so that aggregation period can be defined
    if step > 144 then
      step = 6
    elseif step > 90 then
      step = 3
    else
      step = 1
    end

    step = time_duration(HPTimeResolution.kHourResolution, step)

    local ftype = forecast_type(HPForecastType.kDeterministic)
    local radparam = param("RADGLO-WM2", aggregation(HPAggregationType.kAverage, step), processing_type())

    rad_arr = luatool:FetchWithProducer(latest_time, l0, radparam, ftype, rad_prod, "")
  else
    -- hard coding aggregation period to 1 hour; might need to changed later (possibly)
    local radparam = param("RADGLO-WM2", aggregation(HPAggregationType.kAverage, time_duration(HPTimeResolution.kHourResolution, 1)), processing_type())
    rad_arr = luatool:Fetch(current_time, l0, radparam, current_forecast_type)
  end

  -- if we don't have radiation data, we create an array of missing values
  if not rad_arr then
    rad_arr = {}
    for i=1, rad_arr_len do
	rad_arr[i] = missing
    end
  end

  return rad_arr
end

local rh_arr = luatool:Fetch(current_time, l2, param('RH-PRCNT'), current_forecast_type)
local t_arr = luatool:Fetch(current_time, l2, param('T-K'), current_forecast_type)
local ws_arr = luatool:Fetch(current_time, l10, param('FF-MS'), current_forecast_type)

if not ws_arr or not rh_arr or not t_arr then
    return
end

local rad_arr = FetchRadiation(configuration:GetValue("radiation_producer"), #t_arr)
local simmer_index = FmiSummerSimmerIndex(rh_arr, t_arr)

local feelslike = {}
local a = 15.0 -- using this the two wind chills are good match at T=0
local t0 = 37.0 -- wind chill is horizontal at this T

-- Chosen so that at wind=0 and rad=800 the effect is 4 degrees
-- At rad=50 the effect is then zero degrees
local absorption = 0.07

for i=1, #t_arr do
    local t = t_arr[i] - kKelvin
    local rh = rh_arr[i]
    local ws = ws_arr[i]
    local rad = rad_arr[i]

    -- summer simmer index, called 'heat' in the c++ code

    local simmer = simmer_index[i]

    -- Calculate adjusted wind chill portion. Note that even though
    -- the Canadian formula uses km/h, we use m/s and have fitted
    -- the coefficients accordingly. Note that (a*w)^0.16 = c*w^16,
    -- i.e. just get another coefficient c for the wind reduced to 1.5 meters.

    local chill = a + (1 - a / t0) * t + a / t0 * (ws + 1)^0.16 * (t - t0)

    -- Add the two corrections together

    local feels = t + (chill - t) + (simmer - t)

    -- Radiation correction done only when radiation is available
    -- Based on the Steadman formula for Apparent temperature,
    -- we just ignore the water vapour pressure adjustment

    if rad == rad then
      feels = feels + (0.7 * absorption * rad / (ws + 10) - 0.25)
    end

    feelslike[i] = kKelvin + feels

end

result:SetParam(param("FEELSLIKE-K"))
result:SetValues(feelslike)
luatool:WriteToFile(result)
