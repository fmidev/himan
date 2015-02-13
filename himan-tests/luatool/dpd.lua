--[[
dewpoint_deficit.lua
 
Example lua script to calculate dew point deficit.
partio/2014-12-12
]]
 
-- "logger" variable is global and is set by himan
-- It should be used to print messages to screen instead of print() function since it
-- follows the global log message configuration of himan.
 
logger:Info("Calculating dewpoint deficit")
 
-- Create three parameters: two for input and one for output
-- Parameter name is "neons-name": http://wikise.fmi.fi/tuottaja or http://scout.fmi.fi/radon/
 
local par1 = param("T-K")
local par2 = param("TD-C") -- name is 'C', but data is 'K'
local par3 = param("DPDEF-C") -- dewpoint deficit

par3:SetGrib1Parameter(123) -- Fake a number

-- "current_time" is a global variable and is set by himan.
-- It is the time this current thread should calculate
-- The value is set in the configuration file.
 
logger:Info("Current step: " .. current_time:GetStep())
 
-- "current_level" is also a global variable and set by himan.
-- It is the level this current thread should calculate
-- The value is set in the configuration file.
 
msg = string.format("Current level: %d/%d", current_level:GetType(), current_level:GetValue())
logger:Info(msg)
 
-- "luatool" variable is a global variable and it is the plugin itself.
-- It is used to fetch and write data.
--
-- Fetch() function retrieves data from any source himan can access.
-- It will return nil if data is not found.

-- EC with level kGround :/
local lvl = level(HPLevelType.kGround, 0)

local t = luatool:Fetch(current_time, lvl, par1)
local td = luatool:Fetch(current_time, lvl, par2)

if not t or not td then
    print("Data not found")
    -- Calling return will stop the execution of this lua script
    return
end
 
local i = 0

local res = {}

for i=1, #t do
	local _t=t[i]
	local _td=td[i]
	if _t ~= kFloatMissing and _td ~= kFloatMissing then
		res[i] = _t - _td
	end

end

result:SetValues(res)

-- SetParam() function will set the parameter information to result variable.
-- When luatool-plugin starts the initial parameter name is "DUMMY". This cannot of
-- course be written to database, so the correct parameter should be set before data
-- is written. Notice that setting the parameter does not affect the data in any way.
 
result:SetParam(par3)

logger:Info("Writing results")
 
-- WriteToFile() function will write data to disk. The file type and name etc is determined
-- from the configuration file.
 
luatool:WriteToFile(result)
