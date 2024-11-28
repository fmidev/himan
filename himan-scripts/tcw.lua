-- Produce total column water for MEPS / MNWC
-- "This parameter is the sum of water vapour, liquid water, cloud ice, 
-- rain and snow in a column extending from the surface of the Earth to 
-- the top of the atmosphere."
-- Source: https://codes.ecmwf.int/grib/param-db/136

local tcwv = luatool:Fetch(current_time, level(HPLevelType.kEntireAtmosphere, 0), param("TOTCWV-KGM2"), current_forecast_type)
local cldice = hitool:VerticalSum(param("CLDICE-KGKG"),8,15000)
local cldwtr = hitool:VerticalSum(param("CLDWAT-KGKG"),8,15000)
local snowmr = hitool:VerticalSum(param("SNOWMR-KGKG"),8,15000)
local rainmr = hitool:VerticalSum(param("RAINMR-KGKG"),8,15000)
local graupmr = hitool:VerticalSum(param("GRAUPMR-KGKG"),8,15000)

if not tcwv or not cldice or not cldwtr or not snowmr or not rainmr or not graupmr then
  logger:Error("Data not found")
  return
end

local tcw = {}
for i=1, #tcwv do
  tcw[i] = tcwv[i] + cldice[i] + cldwtr[i] + snowmr[i] + rainmr[i] + graupmr[i]
end

result:SetParam(param("TCW-KGM2"))
result:SetValues(tcw)
luatool:WriteToFile(result)
