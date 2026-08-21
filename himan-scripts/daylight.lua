-- Daylight:
-- Binary parameter telling whether the sun is up (1) or down (0) at the valid time
-- of the forecast. The limit is the same one that is used for sunrise and sunset
-- in almanacs: the center of the sun is 50' (0.833 degrees) below the horizon, which
-- accounts for atmospheric refraction (34') and the radius of the solar disk (16').

logger:Debug("Calculating Daylight")

local sunrise_elevation_angle = -0.833

local validtime = current_time:GetValidDateTime()
local Daylight = {}

for i=1,result:SizeLocations() do
  -- Default is night; only points where the sun is up are flipped to daylight.
  Daylight[i] = 0

  local elevation_angle = ElevationAngle_(result:GetLatLon(i), validtime)

  if elevation_angle > sunrise_elevation_angle then
    Daylight[i] = 1
  end
end

result:SetParam(param("DAYLIGHT-0OR1"))
result:SetValues(Daylight)
luatool:WriteToFile(result)
