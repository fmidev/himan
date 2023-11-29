-- Near-surface/boundary layer aircraft turbulence intensity.
--
-- The parameter gets (integer) values between 0-5:
--   0 = nil turbulence/smooth
--   1 = feeble turbulence *
--   2 = feeble-moderate turbulence
--   3 = moderate turbulence
--   4 = moderate-severe turbulence
--   5 = severe turbulence
--   (* = not in use presently, may be added later)

-- https://wiki.fmi.fi/display/PROJEKTIT/Rajakerroksen+turbulenssi
-- original smarttool script written by Simo Neiglick:
-- junila 11/2015 new version 10/2016
-- tack 06/2021 new version

logger:Info("Calculating boundary layer turbulence")

-- fetch producer information
local currentProducer = configuration:GetTargetProducer()
local currentProducerName = currentProducer.GetName(currentProducer)

local turbparam = param("BLTURB-N")

-- Surface (10m) wind speed in m/s
local ws = param("FF-MS")
local lev = level(HPLevelType.kHeight, 10)
local wsms = luatool:Fetch(current_time, lev, ws, current_forecast_type)

-- Wind gust
local gustlevel = level(HPLevelType.kHeight, 10) 
local timestep = configuration:GetForecastStep():Hours()

local gust = param("FFG-MS", aggregation(HPAggregationType.kMaximum, time_duration(HPTimeResolution.kHourResolution, timestep)), processing_type())
local gustms = luatool:Fetch(current_time, gustlevel, gust, current_forecast_type)

-- land-sea-mask, proportion from 0 to 1 where 1=land, 0=sea
local landseamask = param("LC-0TO1")
local surface = level(HPLevelType.kHeight, 0)
local mytime = forecast_time(current_time:GetOriginDateTime(),time_duration(HPTimeResolution.kHourResolution,0))

local lcmask = luatool:Fetch(mytime, surface, landseamask, current_forecast_type)

if not (wsms and gustms and lcmask) then
  logger:Error("No data found")
  return
end

logger:Info("Calculating wind shear")

-- Low level wind shear 0-1000ft (de facto: 10-304.8m) scaled to kt/1000ft [kt]
-- Possible to use u and v components of wind as they are in model output (rotated latlon)
-- because only change in wind direction and speed is needed, not absolutely correct direction

local dz = 304.8

-- Use lowest model level wind for sfc shear in ***Hirlam/MEPS, 10m level at EC:
-- Wind component differences between Hirlam lowest model level (L65 ~12m) and 1000ft (above it)

local U_HIR = param("U-MS")
local V_HIR = param("V-MS")
local u_0 = luatool:Fetch(current_time, lev, U_HIR, current_forecast_type)
local v_0 = luatool:Fetch(current_time, lev, V_HIR, current_forecast_type)
local u = hitool:VerticalValue(U_HIR,dz+12)
local v = hitool:VerticalValue(V_HIR,dz+12)

-- Wind component at 1000ft
local u1000 = hitool:VerticalValue(U_HIR,dz)
local v1000 = hitool:VerticalValue(V_HIR,dz)

if not (u and v and u1000 and v1000 and u_0 and v_0) then
  return
end

-- Wind component at 2000ft
local u2000 = hitool:VerticalValue(U_HIR,dz*2)
local v2000 = hitool:VerticalValue(V_HIR,dz*2)

if not (u2000 and v2000) then
  return
end

-- Wind component at 3000ft
local u3000 = hitool:VerticalValue(U_HIR,dz*3)
local v3000 = hitool:VerticalValue(V_HIR,dz*3)

if not (u3000 and v3000) then
  return
end

-- Wind component at 4000ft
local u4000 = hitool:VerticalValue(U_HIR,dz*4)
local v4000 = hitool:VerticalValue(V_HIR,dz*4)

if not (u4000 and v4000) then
  return
end

local turbulence = {}
local Missing = missing

for i = 1, #v_0 do

  local du = u[i] - u_0[i]
  local dv = v[i] - v_0[i]
  local du12 = u1000[i] - u2000[i]
  local dv12 = v1000[i] - v2000[i]
  local du23 = u2000[i] - u3000[i]
  local dv23 = v2000[i] - v3000[i]
  local du34 = u3000[i] - u4000[i]
  local dv34 = v3000[i] - v4000[i]

  local shear = math.sqrt(math.pow(du/dz,2) + math.pow(dv/dz,2))*dz*1.946
  local shear12 = math.sqrt(math.pow(du12/dz,2) + math.pow(dv12/dz,2))*dz*1.946
  local shear23 = math.sqrt(math.pow(du23/dz,2) + math.pow(dv23/dz,2))*dz*1.946
  local shear34 = math.sqrt(math.pow(du34/dz,2) + math.pow(dv34/dz,2))*dz*1.946

  local maxshear = math.max(shear, shear12, shear23, shear34)

  -- Turbulence intensity
  local wskt = wsms[i]*1.946
  local lc = lcmask[i]
  local turb = Missing
  local wgust = (gustms[i] - wsms[i])*1.946

  if IsValid(wskt) and IsValid(shear) then

    turb = 0

    -- Turbulence over land areas

    if lc > 0.4 then

      -- FBL-MOD over land

      if wskt>12 and shear >20 then
        turb = 2
      end
      if wskt>14 and (wgust>14 or maxshear>15) then
        turb = 2
      end

      -- MOD over land

      if wskt>16 and (wgust>18 or maxshear>20) then
        turb = 3
      end
      if wskt>18 and (wgust>14 or maxshear>15) then
        turb = 3
      end

      -- MOD-SEV over land

      if wskt>20 and (wgust>18 or maxshear>25) then
        turb = 4
      end
      if wskt>22 and (wgust>16 or maxshear>20) then
        turb = 4
      end

      -- SEV over land

      if wskt>=24 and wgust>20 then
        turb = 5
      end
      if wskt>26 and (wgust>18 or maxshear>25) then
        turb = 5
      end
    end

    -- Turbulence at sea (or over lakes)

    if lc <= 0.4 then

      -- FBL-MOD over sea

      if wskt>25 and wgust>14 then
        turb = 2
      end

      -- MOD over sea

      if wskt>=30 and wgust>15 then  
        turb = 3
      end

      -- MOD-SEV over sea

      if wskt>=35 and wgust>20 then
        turb = 4
      end

      -- SEV over sea
      if wskt>40 and wgust>23 then
        turb = 5
      end
    end
  end
  turbulence[i]= turb
end

result:SetParam(turbparam)
result:SetValues(turbulence)
luatool:WriteToFile(result)
