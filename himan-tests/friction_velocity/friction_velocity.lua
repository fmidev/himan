-- Friction velocity code
-- Tack 6/2015
 
logger:Info("Calculating friction velocity")
 
-- Create three parameters: three for input and one for output
-- Parameter name is "neons-name": http://wikise.fmi.fi/tuottaja or http://scout.fmi.fi/radon/
 
local par1 = param("EWSS-NM2S") -- Eastward turbulent surface stress
local par2 = param("NSSS-NM2S") -- Northward turbulent surface stress
local par3 = param("RHO-KGM3") -- density
local par4 = param("FRVEL-MS") -- friction velocity

par4:SetGrib1Parameter(123) -- Fake a number

logger:Info("Current step: " .. current_time:GetStep())
 
msg = string.format("Current level: %d/%d", current_level:GetType(), current_level:GetValue())
logger:Info(msg)
 
-- EC with level kGround :/
local lvl = level(HPLevelType.kGround, 0)

local ewss = luatool:Fetch(current_time, lvl, par1)
local nsss = luatool:Fetch(current_time, lvl, par2)
local rho = luatool:Fetch(current_time, lvl, par3)

local timestepSize = configuration:GetForecastStep()*3600

if not (ewss and nsss and rho) then
    print("Data not found")
    -- Calling return will stop the execution of this lua script
    return
end
 
local i = 0

local res = {}

for i=1, #ewss do
	local _ewss=ewss[i]
	local _nsss=nsss[i]
	local _rho=rho[i]
	if _ewss ~= kFloatMissing and _nsss ~= kFloatMissing and _rho ~= kFloatMissing then
		-- friction velocity is defined as sqrt(sheer stress / desity ) 
		res[i] = (((_ewss/timestepSize)^2 + (_nsss/timestepSize)^2)/_rho^2)^(1/4)
	end

end

result:SetValues(res)

result:SetParam(par4)
logger:Info("Writing results")
luatool:WriteToFile(result)
