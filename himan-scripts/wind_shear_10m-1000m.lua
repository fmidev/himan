local  above_level = level(HPLevelType.kHybrid,current_level:GetValue())
above_level:SetValue(above_level:GetValue()-1)

local windParam = param("FF-MS")
local heightParam = param("HL-M")

local ws = luatool:Fetch(current_time,current_level,windParam,current_forecast_type)
local above_ws = luatool:Fetch(current_time,above_level,windParam,current_forecast_type)

local height = luatool:Fetch(current_time,current_level,heightParam,current_forecast_type)
local above_height = luatool:Fetch(current_time,above_level,heightParam,current_forecast_type)

local shear = {}

for i = 1,#ws do
	shear[i] = (above_ws[i] - ws[i]) / (above_height[i] - height[i]) * 59.25
end

result:SetParam(param("W-SHEAR-KTFT"))
result:SetValues(shear)
luatool:WriteToFile(result)
