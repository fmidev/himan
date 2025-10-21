--[[
fog-intensity.lua

aaltom/2017-03-31

Values are:

0 = no fog
1 = fog
2 = dense fog

]]

local Missing = missing

par1 = param("VV2-M") -- himan processed visibility
par2 = param("VV-M") -- visibility from model
par3 = param("FOGINT-N") -- Fog intensity

local lvl = level(HPLevelType.kHeight, 0)
local hp_vis = luatool:Fetch(current_time, lvl, par1, current_forecast_type)

if not hp_vis then
  logger:Info("Post processed visibility not found, trying raw")
  hp_vis = luatool:Fetch(current_time, lvl, par2, current_forecast_type)

  if not hp_vis then
    logger:Error("No visibility data found")
    return
  end

else
  logger:Info("Found post processed visibility, using that")
end

local res = {}

for i=1, #hp_vis do
  local visibility = hp_vis[i]
  local fog = 0

  if visibility <= 600 then
    fog = 2
  elseif visibility > 600 and visibility <= 1000 then
    fog = 1
  end

  res[i] = fog

end

result:SetParam(par3)
result:SetValues(res)

logger:Info("Writing results")

luatool:WriteToFile(result)
