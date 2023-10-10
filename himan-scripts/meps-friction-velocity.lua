-- Friction velocity code
-- Tack 6/2015
 
logger:Info("Calculating friction velocity")
 
local par1 = param("UFLMOM-NM2") -- Eastward turbulent surface stress
local par2 = param("VFLMOM-NM2") -- Northward turbulent surface stress
local par3 = param("RHO-KGM3") -- density
local par4 = param("FRVEL-MS") -- friction velocity

local lvl = level(HPLevelType.kHeight, 0)

local ewss = luatool:Fetch(current_time, lvl, par1, current_forecast_type)
local nsss = luatool:Fetch(current_time, lvl, par2, current_forecast_type)
local rho = luatool:Fetch(current_time, lvl, par3, current_forecast_type)

local timestepSize = configuration:GetForecastStep():Hours()*3600

if not (ewss and nsss and rho) then
    print("Data not found")
    return
end

function isnan(v)
    if type(v) == 'number' and tostring(v) == 'nan' then
        return true
    else
        return false
    end
end

function isfinite(v)
    return not isnan(v)
end
 
local i = 0

local res = {}

for i=1, #rho do
    local _ewss=ewss[i]
    local _nsss=nsss[i]
    local _rho=rho[i]

    -- friction velocity is defined as sqrt(sheer stress / desity ) 
    if isfinite(_rho) then
       res[i] = (((_ewss/timestepSize)^2 + (_nsss/timestepSize)^2)/_rho^2)^(1/4)
    else
        res[i] = missing
    end    
end

result:SetValues(res)

result:SetParam(par4)
logger:Info("Writing results")
luatool:WriteToFile(result)
