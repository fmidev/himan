-- Hirlam near-surface/boundary layer aircraft turbulence intensity.
-- Turbulent layer height may then be estimated by boundary layer height.
-- Requirements:
-- - TKE and wind on model levels (+model level heights) in the surface layer (0...500m)
-- - land/sea/lake differentiation
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

logger:Info("Calculating boundary layer turbulence")

local turbparam = param("BLTURB-N")

-- Surface (10m) wind speed in m/s = lowest model level = 65
local ws = param("FF-MS")
local lev = level(HPLevelType.kHybrid, 65)

local wsms = luatool:FetchWithType(current_time, lev, ws, current_forecast_type)

if not wsms then
  logger:Error("No data found")
  return
end

-- Max tke in the layer 0-500m (note: max tke is generally found at the 1st model level above the surface)
local tke = param("TKEN-JKG")
local maxtke = hitool:VerticalMaximum(tke,0,500)

if not maxtke then
  logger:Error("No data found")
  return
end

logger:Info("Calculating wind shear")

-- Low level wind shear 0-1000ft (de facto: 10-304.8m) scaled to kt/1000ft [kt]
-- Possible to use u and v components of wind as they are in model output (rotated latlon)
-- because only change in wind direction and speed is needed, not absolutely correct direction

local dz = 304.8
local U_HIR = param("U-MS")
local V_HIR = param("V-MS")
local u_dz = hitool:VerticalValue(U_HIR,dz)
local u_0 = luatool:FetchWithType(current_time, lev, U_HIR, current_forecast_type)
local v_dz = hitool:VerticalValue(V_HIR,dz)
local v_0 = luatool:FetchWithType(current_time, lev, V_HIR, current_forecast_type)

if not u_dz or not u_0 or not v_dz or not v_0 then
  return
end

local windshear = {}

for i = 1, #v_0 do

  local u2 = u_dz[i]
  local u1 = u_0[i]
  local v2 = v_dz[i]
  local v1 = v_0[i]

  local shear = math.sqrt(math.pow(((u2-u1)/(dz-10)),2)+ math.pow(((v2-v1)/(dz-10)),2))*(dz-10)*1.946

  windshear[i]= shear

end

-- land-sea-mask, proportion from 0 to 1 where 1=land, 0=sea
local landseamask = param("LC-0TO1")
local surface = level(HPLevelType.kHeight, 0)
local lcmask = luatool:FetchWithType(current_time, surface, landseamask, current_forecast_type)

if not lcmask then
  return
end

-- Turbulence intensity

local turbulence = {}
local Missing = missing

for i = 1, #maxtke do

  local wskt = wsms[i]*1.946
  local lc = lcmask[i]
  local maxTKE = maxtke[i]
  local shear = windshear[i]
  local turb = Missing

  if wskt == wskt and maxTKE == maxTKE and shear == shear then

    turb = 0

    -- Turbulence over land areas

    if lc > 0.5 then

      -- FBL-MOD over land

      if wskt>11 and maxTKE>4 then
        turb = 2
      end
      if wskt>14 and maxTKE>3.5 then
        turb = 2
      end

      -- MOD over land

      if wskt>12 and shear*maxTKE >80 then
        turb = 3
      end
      if wskt>14 and (maxTKE>4 or shear>25) then
        turb = 3
      end
      if wskt>17 and maxTKE>2.5 then
        turb = 3
      end

      -- MOD-SEV over land

      if wskt>20 and maxTKE>4.5 then
        turb = 4
      end
      if wskt>22 and maxTKE>3.5 then
        turb = 4
      end

      -- SEV over land

      if wskt>=26 and maxTKE>4 then
        turb = 5
      end

    end

    -- Turbulence at sea (or over lakes)

    if lc <= 0.5 then

      -- MOD over sea

      if wskt>=35 and maxTKE>3 then  -- MOD over sea
        turb = 3
      end

      -- MOD-SEV over sea

      if wskt>=50 and maxTKE>4 then  -- SEV over sea
        turb = 4
      end

    end
  end
  turbulence[i]= turb
end

result:SetParam(turbparam)
result:SetValues(turbulence)
luatool:WriteToFile(result)
