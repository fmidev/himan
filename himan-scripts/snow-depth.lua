--[[
snow-depth.lua

aaltom/2015-02-12
]]

local Missing = missing
local scale = 1

logger:Info("Calculating Snow depth")

par1 = param("SND-KGM3") -- Snow density
par2 = param("SD-KGM2") -- Snow depth in water equivalent
par3 = param("SD-M") -- Snow depth in m

local lvl = level(HPLevelType.kHeight, 0)
local prod = configuration:GetSourceProducer(0)
local prod_name = prod.GetName(prod)

if prod:GetId() == 131 then
   par2 = param("SD-TM2")
   scale = 1000
end

local sd = luatool:Fetch(current_time, lvl, par1, current_forecast_type)
local sw = luatool:Fetch(current_time, lvl, par2, current_forecast_type)

if not sd or not sw then
  return
end

local res = {}

for i=1, #sw do
  local sw = sw[i]
  local sd = sd[i]
  local sn = Missing

  if IsValid(sw) and IsValid(sd) then
    if sd == 0 then
      sn = 0
    else
      sn = sw * scale / sd
    end
  end

  res[i] = sn

end

result:SetParam(par3)
result:SetValues(res)

logger:Info("Writing results")

luatool:WriteToFile(result)
