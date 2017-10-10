-- Calculate a bunch of parameters a certain pressure level from hybrid levels
-- using vertical interpolation
-- partio/20150302

local MISS = missing

function Interpolate(p)

  logger:Info("Calculating " .. p .. " for " .. current_level:GetValue() .. " hPa")

  local nparam = param(p)

  hitool:SetHeightUnit(HPParameterUnit.kHPa)
  local data = hitool:VerticalValue(nparam, current_level:GetValue())

  if not data then
    return
  end

  result:SetParam(nparam)
  result:SetValues(data)

  luatool:WriteToFile(result)
end

params = { "N-0TO1", "ICING-N" }

for k,param in pairs(params) do
  Interpolate(param)
end
