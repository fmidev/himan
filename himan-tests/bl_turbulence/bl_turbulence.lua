-- Hirlam near-surface/boundary layer aircraft turbulence intensity.
-- Turbulent layer height may then be estimated by boundary layer height.
-- Requirements:
-- - surface (10m) wind speed
-- - TKE and wind on model levels (+model level heights) in the surface layer (0...500m)
-- - land/sea/lake differentiation (topography)
-- - note: lakes should be included in "over sea" calculation, but topography probably >0 over lakes (?)
-- 
-- The parameter gets (integer) values between 0-5:
--   0 = nil turbulence/smooth
--   1 = feeble turbulence *
--   2 = feeble-moderate turbulence *
--   3 = moderate turbulence
--   4 = moderate-severe turbulence *
--   5 = severe turbulence
--   (* = not in use presently, may be added later)

-- https://wiki.fmi.fi/display/PROJEKTIT/Rajakerroksen+turbulenssi
-- original smarttool script written by Simo Neiglick:
-- junila 11/2015
 
logger:Info("Calculating boundary layer turbulence")
  
local kFloatMissing = kFloatMissing
 
-- Surface (10m) wind speed in m/s
-- ws = param("FF-MS")
-- wsms = hitool:VerticalValue(ws,10)
local ws = param("FF-MS")
local l = level(HPLevelType.kHeight, 10) 

local wsms = luatool:Fetch(current_time, l, ws)

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

-- Low level wind shear 0-1000ft (0-304.8m) scaled to kt/1000ft [kt]
local dz = 304.8
local U_HIR = param("U-MS")
local V_HIR = param("V-MS")
local u_dz = hitool:VerticalValue(U_HIR,dz)
local u_0 = hitool:VerticalValue(U_HIR,0)
local v_dz = hitool:VerticalValue(V_HIR,dz)
local v_0 = hitool:VerticalValue(U_HIR,0)

if not u_dz or not u_0 or not v_dz or not v_0 then
    logger:Error("No data found")
    return
end

    local windshear = {}

    for i = 1, #v_0 do
    
	local u2 = u_dz[i]
	local u1 = u_0[i]
	local v2 = v_dz[i]
	local v1 = v_0[i]

	local shear = kFloatMissing

	if u2 ~= kFloatMissing and u1 ~= kFloatMissing and v2 ~= kFloatMissing and v1 ~= kFloatMissing then
		shear = math.sqrt(math.pow(((u2-u1)/dz),2)+ math.pow(((v2-v1)/dz),2))*dz/0.514
	end
	
	windshear[i]= shear
	
	
    end


-- TOPO  = topography [m] (TOPO<=0 over sea)
-- [m2/s2] -> [m]

local geopotential = param("Z-M2S2")
local topo = hitool:VerticalValue(geopotential,0)

if not topo then
    logger:Error("No data found")
    return
end

    
-- Turbulence intensity

	local turbulence = {}
	
	for i = 1, #maxtke do
			
		local wskt = wsms[i]/0.514
		local TOPO = topo[i]/9.81
		local maxTKE = maxtke[i] 
		local shear = windshear[i]

		local turb = kFloatMissing

		if wskt ~= kFloatMissing and TOPO ~= kFloatMissing and maxTKE ~= kFloatMissing and shear ~= kFloatMissing then

		turb = 0

			-- Turbulence over land areas

				if TOPO>0 then

  					-- MOD over land
  					if wskt>13 and maxTKE>2.5 then

 				   		if wskt>16 then
				    		  turb = 3 
				    		elseif maxTKE>3.5 or shear>25 then
				    		  turb = 3 
				    		end

 				 	elseif wskt>10 and shear*maxTKE>65 then
 				  		turb = 3
 				 	end
	 
				  	-- SEV over land
				  	if wskt>=20 and maxTKE>3 then

				    		if wskt>22 then
				    		  turb = 5 
				    		elseif maxTKE>4 then
				    		  turb = 5 
				    		end
			
				  	end

			-- Turbulence at sea (or over lakes)

				else

				  	-- MOD over sea
				  	if wskt>=35 and maxTKE>3 then
				  		  turb = 3 
				  	end  
	
				 	 -- SEV over sea
 				 	if wskt>=50 and maxTKE>4 then
 				 		  turb = 5 
 				 	end

				end
				
		end

	turbulence[i]= turb
	
	
 	
	end
  
result:SetParam(param("BLTURB-N"))

result:SetValues(turbulence)
 
logger:Info("Writing results")
 
luatool:WriteToFile(result)
    
