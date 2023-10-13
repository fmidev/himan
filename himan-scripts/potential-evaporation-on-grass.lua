-- Calculate the Penman-Monteith potential evaporation on grass surfaces
--

-- Constants
local ROO   = 1.2923
local CP    = kCp
local L     = kL
local STEF  = 5.67E-8
local EM    = 0.98

-- Input data
-- Configure RH parameter
local prod = configuration:GetSourceProducer(0)

local RHParam = param("RH-0TO1")
local RHScale = 1
if prod:GetId() == 134 then
  RHParam = param("RH-PRCNT")
  RHScale = 0.01
end

-- Fetch
local temp = luatool:Fetch(current_time, current_level, param("T-K"), current_forecast_type)
local windspeed = luatool:Fetch(current_time, level(HPLevelType.kHeight, 10), param("FF-MS"), current_forecast_type)
local netsw = luatool:Fetch(current_time, current_level, param("RNETSW-WM2"), current_forecast_type)
local netlw = luatool:Fetch(current_time, current_level, param("RNETLW-WM2"), current_forecast_type)
local relhum = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), RHParam, current_forecast_type)

-- Output
local evaporation = {}

-- Start calculation
for i=1,#temp do
	local T = temp[i] - 273.15
	local U = math.max(windspeed[i], 0.5)
	local RA = 0.6 * (( math.log(8. / 0.0002) / 0.4 )^2) / U
	local RN = netsw[i] + netlw[i] --skipping some steps and calculate net total radiation from model net radiation directly
	local ES = Es_(temp[i])
	local E = relhum[i] * ES * RHScale
	local KALT = ES * math.log(10.) * (1777.5 / (237. + T)^2)
	local KORKOR = 4. * STEF * EM * ((273.1 + T)^3)

	evaporation[i] = (KALT * RN + ROO * CP * (1 + KORKOR * RA / (ROO * CP)) * (ES - E) / RA) / (KALT + 0.66 * (1 + KORKOR * RA / (ROO * CP))) / L * 3600. * 3.

	evaporation[i] = math.max(0,evaporation[i])
end

local par = param("EVARATE-KGM2S")

result:SetParam(par)
result:SetValues(evaporation)

luatool:WriteToFile(result)
