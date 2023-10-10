-- Interpolate data from half to full levels
-- Some parameters in ICON (TKE, W) are defined for half levels, where as the rest (U,V,...)
-- are defined for full levels :facepalm:
-- 
-- Because full levels are defined to be right in the middle of
-- half levels, interpolation is actually just an average of the
-- half level values.
--
-- Transformer plugin can also do direct level interpolation but
-- this scenario cannot be configured using the current code
--
-- Script assumes DWD ICON model configuration.
-- https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf

function InterpolateToFullLevel(par)

  -- full levels run from 1 .. 74, half levels 1 .. 75
  -- first full level 1 is between half level 1 and 2 and so on
  
  local z1 = current_level:GetValue()  -- full level value 1
  local hlparam = param("HL-M")        -- half level height param

  local data = luatool:Fetch(current_time, level(HPLevelType.kGeneralizedVerticalLayer, z1, -1), par, current_forecast_type)
  local data_n = luatool:Fetch(current_time, level(HPLevelType.kGeneralizedVerticalLayer, z1+1, -1), par, current_forecast_type)

  if not data or not data_n then
    logger:Error("Some (or all) of the data is not found")
    return
  end

  local intp = {}

  for i = 1, #data do
    intp[i] = (data[i] + data_n[i]) * 0.5
  end

  result:SetParam(par)
  result:SetValues(intp)
  luatool:WriteToFile(result)
end

InterpolateToFullLevel(param(configuration:GetValue("param")))
