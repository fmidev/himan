-- Sum precipitation components to get total precipitation

logger:Info("Calculating total precipitation")

local subhour = current_time:GetStep():Minutes()

if subhour ~= 0 then
  -- RR needs to be calculated only for subhours
  logger:Info("Skipping full hour RR calculation for " .. current_time:GetStep():Minutes())
  return
end

local par1 = param("RACC-KGM2") -- rain
local par2 = param("GR-KGM2") -- graupel
local par3 = param("SNACC-KGM2") -- snowfall
local par4 = param("RR-KGM2") -- total precipitation

local rr = luatool:FetchWithType(current_time, current_level, par1, current_forecast_type)
local gr = luatool:FetchWithType(current_time, current_level, par2, current_forecast_type)
local sn = luatool:FetchWithType(current_time, current_level, par3, current_forecast_type)

if not rr or not gr or not sn then
  return
end

local tp = {}

for i=1, #rr do
  tp[i] = rr[i] + gr[i] + sn[i]
end

result:SetValues(tp)
result:SetParam(par4)
luatool:WriteToFile(result)
