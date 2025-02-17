--
-- Miscellaneous manipulation to ICON europe data
-- 

local MISS = missing

function GetStepLength()

  local step = current_time:GetStep():Hours()

  if step < 81 then
    return time_duration("01:00:00")
  end

  return time_duration("03:00:00")

end

function LSPAndCP()

  -- large scale precipitation and convective precipitation from rain+snow

  local agg = aggregation(HPAggregationType.kAccumulation, current_time:GetStep())
  local pt = processing_type()

  local sconvdata = luatool:Fetch(current_time, current_level, param("SNC-KGM2", agg, pt), current_forecast_type)
  local slsdata = luatool:Fetch(current_time, current_level, param("SNL-KGM2", agg, pt), current_forecast_type)
  local rconvdata = luatool:Fetch(current_time, current_level, param("RAINC-KGM2", agg, pt), current_forecast_type)
  local rlsdata = luatool:Fetch(current_time, current_level, param("RAINL-KGM2", agg, pt), current_forecast_type)

  if not sconvdata or not slsdata or not rconvdata or not rlsdata then
    logger:Error("Some data not found, aborting")
    return
  end

  local cdata = {}
  local ldata = {}

  for i=1, #rlsdata do
    cdata[i] = sconvdata[i] + rconvdata[i]
    ldata[i] = slsdata[i] + rlsdata[i]
  end

  result:SetParam(param("RRC-KGM2", agg, processing_type()))
  result:SetValues(cdata)
  luatool:WriteToFile(result)
  result:SetParam(param("RRL-KGM2", agg, processing_type()))
  result:SetValues(ldata)
  luatool:WriteToFile(result)

end

function SnowFall()

  -- total snow fall rate from convective and large scale rates
  local agg = aggregation(HPAggregationType.kAccumulation, current_time:GetStep())
  local pt = processing_type()

  local convdata = luatool:Fetch(current_time, current_level, param("SNC-KGM2", agg, pt), current_forecast_type)
  local lsdata = luatool:Fetch(current_time, current_level, param("SNL-KGM2", agg, pt), current_forecast_type)

  if not convdata or not lsdata then
    logger:Error("Some data not found, aborting")
    return
  end

  local data = {}

  for i=1, #lsdata do
    data[i] = lsdata[i] + convdata[i]
  end

  result:SetParam(param("SNACC-KGM2", agg, pt))
  result:SetValues(data)
  luatool:WriteToFile(result)

end


function RadGlo()

  local step = GetStepLength()

  -- global radiation from direct and diffuse short wave radiation
  -- fetch the hourly values and sum them up
  -- note: hourly values are produced by himan; the original data is from analysis time

  local difparam = param("RDIFSW-WM2", aggregation(HPAggregationType.kAverage, step), processing_type())
  local dirparam = param("RADSWDIR-WM2", aggregation(HPAggregationType.kAverage, step), processing_type())

  local difdata = luatool:Fetch(current_time, current_level, difparam, current_forecast_type)
  local dirdata = luatool:Fetch(current_time, current_level, dirparam, current_forecast_type)

  if not difdata or not dirdata then
    logger:Error("Some data not found, aborting")
    return
  end

  local data = {}

  for i=1, #difdata do
    data[i] = difdata[i] + dirdata[i]
  end

  resparam = param("RADGLO-WM2", aggregation(HPAggregationType.kAverage, step), processing_type())
  result:SetParam(resparam)
  result:SetValues(data)
  luatool:WriteToFile(result)

end

function H0CFix()

  -- height of 0 degree isotherm is from MSL -- we want it from ground level

  local h0cdata = luatool:Fetch(current_time, current_level, param("H0C-M"), current_forecast_type)
  local srctime = forecast_time(current_time:GetOriginDateTime(), current_time:GetOriginDateTime())
  local topodata = luatool:Fetch(srctime, current_level, param("HL-M"), current_forecast_type)

  if not h0cdata or not topodata then
    logger:Error("Some data not found, aborting")
    return
  end

  local data = {}
  for i=1, #topodata do
    data[i] = h0cdata[i] + topodata[i]
  end

  result:SetParam(param("H0C-M"))
  result:SetValues(data)
  luatool:WriteToFile(result)
end

function LCLFix()

  -- height of LCL is from MSL -- we want it from ground level

  local lcldata = luatool:Fetch(current_time, current_level, param("LCL-M"), current_forecast_type)
  local srctime = forecast_time(current_time:GetOriginDateTime(), current_time:GetOriginDateTime())
  local topodata = luatool:Fetch(srctime, current_level, param("HL-M"), current_forecast_type)

  if not lcldata or not topodata then
    logger:Error("Some data not found, aborting")
    return
  end

  local data = {}
  for i=1, #topodata do
    data[i] = lcldata[i] + topodata[i]
  end

  result:SetParam(param("LCL-M"))
  result:SetValues(data)
  luatool:WriteToFile(result)
end

function TSnowFix()

  -- "Temperature of snow surface. At snow-free points (H_SNOW = 0), 
  -- T_SNOW contains the temperature of the soil surface T_SO(0)."
  --
  -- Use snow depth as a mask to determine when there is no snow and set all
  -- those points to missing
  --
  -- Edit: unfortunately that's not enough as even with this filttering there
  -- are many points that have t ~ 10 degrees K, which seems a bit cold.
  --
  -- How cold can snow temperature get?
  --
  -- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL078133
  --
  -- "Approximately 100 sites have observed minimum surface temperatures of ~−98 °C during the winters of 2004–2016"
  --
  -- (At Antarctica)
  --
  -- It's safe to assume that in Europe we don't get colder weather than in Antarctica, so limit
  -- minimum snow temperature to -100C (=173K).
  --

  local snowtdata = luatool:Fetch(current_time, current_level, param("TSNOW-K"), current_forecast_type)
  local snowhdata = luatool:Fetch(current_time, current_level, param("SD-M"), current_forecast_type)
 
  if not snowtdata or not snowhdata then
    logger:Error("Some data not found, aborting")
    return
  end

  local data = {}
  for i=1, #snowtdata do
    local t = snowtdata[i]
    local h = snowhdata[i]

    if h == 0 or t < 173 then
        t = MISS
    end

    data[i] = t
  end

  result:SetParam(param("TSNOW-K"))
  result:SetValues(data)
  luatool:WriteToFile(result)
end


-- H0CFix()
LCLFix()
TSnowFix()
RadGlo()
SnowFall()
LSPAndCP()
