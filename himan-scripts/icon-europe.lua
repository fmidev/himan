--
-- Miscellaneous manipulation to ICON europe data
-- 

local MISS = missing

function SnowFall()

  -- total snow fall rate from convective and large scale rates

  local convdata = luatool:FetchWithType(current_time, current_level, param("SNRC-KGM2"), current_forecast_type)
  local lsdata = luatool:FetchWithType(current_time, current_level, param("SNRL-KGM2"), current_forecast_type)

  if not convdata or not lsdata then
    print("Some data not found, aborting")
    return
  end

  local data = {}

  for i=1, #lsdata do
    data[i] = lsdata[i] + convdata[i]
  end

  result:SetParam(param("SNR-KGM2"))
  result:SetValues(data)
  luatool:WriteToFile(result)

end


function RadGlo()

  -- global radiation from direct and diffuse short wave radiation

  local difdata = luatool:FetchWithType(current_time, current_level, param("RDIFSW-WM2"), current_forecast_type)
  local dirdata = luatool:FetchWithType(current_time, current_level, param("RADSWDIR-WM2"), current_forecast_type)

  if not difdata or not dirdata then
    print("Some data not found, aborting")
    return
  end

  local data = {}

  for i=1, #difdata do
    data[i] = difdata[i] + dirdata[i]
  end

  result:SetParam(param("RADGLO-WM2"))
  result:SetValues(data)
  luatool:WriteToFile(result)

end

function H0CFix()

  -- height of 0 degree isotherm is from MSL -- we want it from ground level

  local h0cdata = luatool:FetchWithType(current_time, current_level, param("H0C-M"), current_forecast_type)
  local srctime = forecast_time(current_time:GetOriginDateTime(), current_time:GetOriginDateTime())
  local topodata = luatool:FetchWithType(srctime, current_level, param("HL-M"), current_forecast_type)

  if not h0cdata or not topodata then
    print("Some data not found, aborting")
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

  local lcldata = luatool:FetchWithType(current_time, current_level, param("LCL-M"), current_forecast_type)
  local srctime = forecast_time(current_time:GetOriginDateTime(), current_time:GetOriginDateTime())
  local topodata = luatool:FetchWithType(srctime, current_level, param("HL-M"), current_forecast_type)

  if not lcldata or not topodata then
    print("Some data not found, aborting")
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

  local snowtdata = luatool:FetchWithType(current_time, current_level, param("TSNOW-K"), current_forecast_type)
  local snowhdata = luatool:FetchWithType(current_time, current_level, param("SD-M"), current_forecast_type)
 
  if not snowtdata or not snowhdata then
    print("Some data not found, aborting")
    return
  end

  local data = {}
  for i=1, #snowtdata do
    local t = snowtdata[i]
    local h = snowhdata[i]

    if h == 0 then
        t = MISS
    end
    data[i] = t
  end

  result:SetParam(param("TSNOW-K"))
  result:SetValues(data)
  luatool:WriteToFile(result)
end


H0CFix()
LCLFix()
TSnowFix()
RadGlo()
SnowFall()
