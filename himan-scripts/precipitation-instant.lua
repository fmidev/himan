--[[
precipitation_instant.lua

To calculate instant precipitation intensity

junila/2015-01-30
]]

logger:Info("Calculating instantaneous precipitation intensity")

par1 = param("RRI-KGM2") -- Instantaneous rain rate in mm/s
par2 = param("GRI-KGM2") -- Instantaneous graupel rate in mm/s
par3 = param("SNRI-KGM2") -- Instantaneous snowfall rate in mm/s
par4 = param("PRI-KGM2") -- Instantaneous precipitation intensity in mm/h, (par1+par2+par3)*3600

local rri = luatool:Fetch(current_time, current_level, par1, current_forecast_type)
local gri = luatool:Fetch(current_time, current_level, par2, current_forecast_type)
local snri = luatool:Fetch(current_time, current_level, par3, current_forecast_type)

if not rri or not gri or not snri then
  return
end

pri = {}

for i=1, #rri do
  pri[i] = 3600*(rri[i] + gri[i] + snri[i])
end

result:SetValues(pri)
result:SetParam(par4)

logger:Info("Writing results")
luatool:WriteToFile(result)
