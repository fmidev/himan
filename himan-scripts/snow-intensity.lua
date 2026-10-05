local Missing = missing

preform_par = param("POTPRECF-N") -- precipitation form
precip_par = param("RRR-KGM2") -- total precipitation rate
sn_par = param("SNR-KGM2")

local lvl = level(HPLevelType.kHeight, 0)
local preform = luatool:Fetch(current_time, lvl, preform_par, current_forecast_type)
local precip = luatool:Fetch(current_time, lvl, precip_par, current_forecast_type)


local res = {}

for i=1, #preform do
  local snowfall= 0

  if preform[i] == 3 then
    snowfall = precip[i]
  end

  res[i] = snowfall

end

result:SetParam(param("SNR-MM"))
result:SetValues(res)

logger:Info("Writing results")

luatool:WriteToFile(result)

