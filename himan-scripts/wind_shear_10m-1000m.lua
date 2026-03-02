-- STU-29741
-- Define the hybrid level above the current level for vertical difference calculations
local  above_level = level(HPLevelType.kHybrid,current_level:GetValue())
above_level:SetValue(above_level:GetValue()-1)

-- Parameters for height and wind components
local heightParam = param("HL-M")
local uParam = param("U-MS")
local vParam = param("V-MS")

-- Fetch wind components and height at the current level
local u = luatool:Fetch(current_time,current_level,uParam,current_forecast_type)
local above_u = luatool:Fetch(current_time,above_level,uParam,current_forecast_type)
local v = luatool:Fetch(current_time,current_level,vParam,current_forecast_type)
local above_v = luatool:Fetch(current_time,above_level,vParam,current_forecast_type)

-- Fetch geopotential heights at both levels for layer thickness (dz)
local height = luatool:Fetch(current_time,current_level,heightParam,current_forecast_type)
local above_height = luatool:Fetch(current_time,above_level,heightParam,current_forecast_type)

-- Output and intermediate arrays
local shear = {}

-- Calculate wind shear only if U and V grids have matching size
if #u == #v then

	for i = 1,#u do
	
		local du = above_u[i] - u[i]
        	local dv = above_v[i] - v[i]
		local dz = above_height[i] - height[i]
		
		-- Wind shear magnitude: vector shear per unit height, converted from (m/s)/m to kt/100ft
		-- math based on: https://wiki.fmi.fi/x/46N7Dw
		shear[i] = (math.sqrt((math.pow(du/dz,2) + math.pow(dv/dz,2)))) * (30.48 / 0.514)
	
	end

else
	print("Grid size mismatch, cannot compute shear")
	
end

-- Write wind shear result to output parameter and file
result:SetParam(param("WSHR-KTHFT"))
result:SetValues(shear)
luatool:WriteToFile(result)
