-- Vire lämppäritesti EC F0 avulla
-- kylmennetään Vireä kohti EC F0:a jos heikkotuulista ja selkeää
-- Original Leila Hieta / 2025-11-19
-- Converted to lua Mikko Partio / 2025-11-20

function summarize_changes(temp, adjusted_temp, dewp, adjusted_dewp, MISSING, nbins)
  nbins = nbins or 10
  
  -- Initialize histogram bins and counters
  local temp_hist = {}
  local dewp_hist = {}
  for i = 1, nbins do
    temp_hist[i] = 0
    dewp_hist[i] = 0
  end
  
  local temp_changed = 0
  local dewp_changed = 0
  local temp_total = 0
  local dewp_total = 0

  -- Find min/max changes for binning
  local min_temp_change, max_temp_change = math.huge, -math.huge
  local min_dewp_change, max_dewp_change = math.huge, -math.huge

  for i = 1, #temp do
    local temp_change = adjusted_temp[i] - temp[i]
    local dewp_change = adjusted_dewp[i] - dewp[i]
    
    if temp_change ~= MISSING then
      temp_total = temp_total + 1
      if temp_change ~= 0 then
        temp_changed = temp_changed + 1
      end
      min_temp_change = math.min(min_temp_change, temp_change)
      max_temp_change = math.max(max_temp_change, temp_change)
    end
    
    if dewp_change ~= MISSING then
      dewp_total = dewp_total + 1
      if dewp_change ~= 0 then
        dewp_changed = dewp_changed + 1
      end
      min_dewp_change = math.min(min_dewp_change, dewp_change)
      max_dewp_change = math.max(max_dewp_change, dewp_change)
    end
  end

  -- Bin the changes
  for i = 1, #temp do
    local temp_change = adjusted_temp[i] - temp[i]
    local dewp_change = adjusted_dewp[i] - dewp[i]
    
    if temp_change ~= MISSING then
      local bin = math.min(nbins, math.max(1, 
        math.floor((temp_change - min_temp_change) / (max_temp_change - min_temp_change + 1e-10) * nbins) + 1))
      temp_hist[bin] = temp_hist[bin] + 1
    end
    
    if dewp_change ~= MISSING then
      local bin = math.min(nbins, math.max(1,
        math.floor((dewp_change - min_dewp_change) / (max_dewp_change - min_dewp_change + 1e-10) * nbins) + 1))
      dewp_hist[bin] = dewp_hist[bin] + 1
    end
  end

  -- Print summary
  logger:Info(string.format("Temperature: %d/%d points changed (%.1f%%)", 
    temp_changed, temp_total, temp_changed / temp_total * 100))
  logger:Info("Temperature changes:")
  for i = 1, nbins do
    local bin_start = min_temp_change + (i-1) * (max_temp_change - min_temp_change) / nbins
    local bin_end = min_temp_change + i * (max_temp_change - min_temp_change) / nbins
    logger:Info(string.format("  [%.2f, %.2f): %d", bin_start, bin_end, temp_hist[i]))
  end

  logger:Info(string.format("\nDewpoint: %d/%d points changed (%.1f%%)", 
    dewp_changed, dewp_total, dewp_changed / dewp_total * 100))
  logger:Info("Dewpoint changes:")
  for i = 1, nbins do
    local bin_start = min_dewp_change + (i-1) * (max_dewp_change - min_dewp_change) / nbins
    local bin_end = min_dewp_change + i * (max_dewp_change - min_dewp_change) / nbins
    logger:Info(string.format("  [%.2f, %.2f): %d", bin_start, bin_end, dewp_hist[i]))
  end
end


function FetchFromMEPS(param, level)
  local meps_prod = producer(4, "MEPS")

  local o = {forecast_time = current_time,
           level = level,
           param = param,
           forecast_type = forecast_type(HPForecastType.kEpsControl, 0),
           producer = meps_prod,
           geom_name = "",
           read_previous_forecast_if_not_found = false
  }

  return luatool:FetchWithArgs(o)
end

function FetchFromVire(param, level)
  local vire = producer(285, "VIRE")

  local analysis_time = current_time:GetOriginDateTime()
  local valid_time = current_time:GetValidDateTime()
  local vire_time = forecast_time(analysis_time, valid_time)

  local meps_hour = tonumber(analysis_time:String('%H'))

  -- MEPS VIRE
  -- 00 19
  -- 03 01
  -- 06 01
  -- 09 07
  -- 12 07
  -- 15 13
  -- 18 13
  -- 21 19

  local adjust_hours = meps_hour % 6 == 0 and -5 or -2
  vire_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjust_hours)

  local o = {forecast_time = vire_time,
           level = level,
           param = param,
           forecast_type = current_forecast_type,
           producer = vire,
           geom_name = "",
           read_previous_forecast_if_not_found = false
  }

  return luatool:FetchWithArgs(o)
end

function FetchInfoFromENS(param, level)
  local ens = producer(242, "ECGEPSMTA")

  local analysis_time = current_time:GetOriginDateTime()
  local valid_time = current_time:GetValidDateTime()
  local ecmwf_time = forecast_time(analysis_time, valid_time)

  local meps_hour = tonumber(analysis_time:String('%H'))

  -- MEPS EC
  -- 00 12
  -- 03 12
  -- 06 12
  -- 09 00
  -- 12 00
  -- 15 00
  -- 18 00
  -- 21 12

  if (meps_hour >= 0 and meps_hour <= 6) or meps_hour == 21 then  
    -- use previous 12 UTC analysis
    adjust_hours = -12 - meps_hour                                          
  else                
    -- use current day 00 UTC analysis
    adjust_hours = -meps_hour  
  end

  ecmwf_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjust_hours)
  local o = {forecast_time = ecmwf_time,
           level = level,
           param = param,
           forecast_type = forecast_type(HPForecastType.kStatisticalProcessing),
           producer = ens,
           geom_name = "",
           read_previous_forecast_if_not_found = false,
	   time_interpolation = true
  }

  return luatool:FetchInfoWithArgs(o)
end

function Write(temperature, dewpoint)
  result:SetParam(param("T-K"))
  result:SetValues(temperature)
  luatool:WriteToFile(result)
  result:SetParam(param("TD-K"))
  result:SetValues(dewpoint)
  luatool:WriteToFile(result)
end


function AdjustTemperatures()
  local temp = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), param("T-K"), current_forecast_type)
  local dewp = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), param("TD-K"), current_forecast_type)
  local wind = FetchFromVire(param("FF-MS"), level(HPLevelType.kHeight, 10))
  local cloud = FetchFromVire(param("NL-0TO1"), level(HPLevelType.kHeight, 0))
  local tempF0info = FetchInfoFromENS(param("F0-T-K", aggregation(), processing_type(HPProcessingType.kFractile, 0)), level(HPLevelType.kHeight, 2))
  local lc = FetchFromMEPS(param("LC-0TO1"), level(HPLevelType.kHeight, 0))

  if not wind or not cloud or not temp or not tempF0info or not dewp or not lc then
      logger:Error("Data not found")
      return
  end

  -- Get the minimum value from a 5x5 stencil
  local mask = matrix(5, 5, 1, missing)
  mask:Fill(1)
  local tempF0 = Min2D(tempF0info:GetData(), mask, configuration:GetUseCuda()):GetValues()

  adjusted_temp = {}
  adjusted_dewp = {}
  adjustment = {}

  for i=1, #temp do
    -- Never raise the temperatures
    local temp_diff = math.min(tempF0[i] - temp[i], 0)
    local w = wind[i]
    local c = cloud[i]
    local lsm = lc[i]
    local td = dewp[i]

    local lat = result:GetLatLon(i):GetY()

    local factor = 0
    -- Adjust temperature towards ECMWF ensemble smallest value depending on cloud cover, wind and whether it is a land or sea point
    if w < 5 and lsm == 1 and c == c then
       if c > 0.8 then
         factor = 0.4
       elseif c >= 0.5 and c <= 0.8 then
         factor = 0.8
       else
         factor = 1.1
       end

       -- Do not adjust values below 64N (~Kajaani)
       factor = RampUp(64, 66, lat) * factor
    end

    adjustment[i] = temp_diff * factor

  end

  -- Apply smoothing to the adjustments: dependency to wind may otherwise
  -- create sharp edges in the adjusted fields
  mask = matrixf(3, 3, 1, missing)
  mask:Fill(1/9)

  local mat = matrixf(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
  mat:SetValues(adjustment)
  adjustment_smooth = Filter2D(mat, mask, configuration:GetUseCuda()):GetValues()

  for i=1, #adjustment_smooth do
    adjusted_temp[i] = temp[i] + adjustment_smooth[i]
    adjusted_dewp[i] = dewp[i] + adjustment_smooth[i]
  end

  summarize_changes(temp, adjusted_temp, dewp, adjusted_dewp, MISSING, 10)

  Write(adjusted_temp, adjusted_dewp)
--  result:SetParam(param("TDIFF-K"))
--  result:SetValues(adjustment_smooth)
--  luatool:WriteToFile(result)
--  logger:Info("Wrote temperature adjustment field")

end

local enable_temperature_adjustment = false
if configuration:Exists("enable_temperature_adjustment") then
  enable_temperature_adjustment = ParseBoolean(configuration:GetValue("enable_temperature_adjustment"))
end

if enable_temperature_adjustment then
  logger:Info("Adjusting temperatures based on ECMWF ensemble minimum")
  AdjustTemperatures()
else
  logger:Info("Temperature adjustment is disabled")
  -- Write original values without adjustment
  local temp = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), param("T-K"), current_forecast_type)
  local dewp = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), param("TD-K"), current_forecast_type)
  Write(temp, dewp)
end
