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
    logger:Error("Data not found")
    return
  end

  if p == "T-K" then
    -- there is no T-K param in database for code table 203, only T-C
    nparam:SetName("T-C")
    nparam:SetGrib1Parameter(4)

    for i = 1, #data do
      local val = data[i]
      if value == value then
        data[i] = val - 273.15 -- scale to celsius
      end
    end
  end

  result:SetParam(nparam)
  result:SetValues(data)

  luatool:WriteToFile(result)
end

params = { "T-K", "RH-PRCNT", "U-MS", "V-MS", "VV-MMS" }

for k,param in pairs(params) do
  Interpolate(param)
end
