-- Calculate full level metric height from half levels.
-- Script assumes DWD ICON model configuration.
-- https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf

local z1 = current_level:GetValue()
local z2 = current_level:GetValue2()
local hlparam = param("HL-M")
local atime = forecast_time(current_time:GetOriginDateTime(), current_time:GetOriginDateTime())

local hhl = luatool:Fetch(atime, level(HPLevelType.kGeneralizedVerticalLayer, z1, 0), hlparam)
local hhl_n = luatool:Fetch(atime, level(HPLevelType.kGeneralizedVerticalLayer, z1+1, 0), hlparam)
local hsurf = luatool:Fetch(atime, level(HPLevelType.kGround, 0), hlparam)

if not hhl or not hhl_n or not hsurf then
  logger:Error("Some (or all) of the data is not found")
  return
end

local hl = {}

for i = 1, #hhl do

  -- geometric height of half level from ground
  -- (original half level height is from mean sea level)
  local zh = hhl[i] - hsurf[i]

  -- and the half level below
  local zh_n = hhl_n[i] - hsurf[i]

  hl[i] = (zh + zh_n) * 0.5
end

result:SetParam(hlparam)
result:SetValues(hl)
luatool:WriteToFile(result)
