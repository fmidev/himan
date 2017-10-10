--This macro calculates the relative humidity of ice
--from HIRLAM surface variables temperature, pressure and dew point.
--Toni Amnell 26.10.2010

function rhice(p,t,dp)
  -- p = air pressure in hPa
  -- t = temperature in *C
  -- dp = dew point in *C

  --Saturated vapor pressure of pure ice or water
  local ei=6.112*math.exp(22.46*t/(272.62+t))
  local ew=6.112*math.exp(17.62*dp/(243.12+dp))

  --Saturated vapor pressure of water/ice in moist air using approximated enhancement factors by Sonntag
  local ei_prime=(1+(1.0e-5*ei/(273+t))*((2100-65*t)*(1-ei/p)+(109-0.35*t+t*t/338)*(p/ei-1)))*ei
  local e_prime=(1+(1.0e-4*ew/(273+dp))*((38+173*math.exp(-dp/43))*(1-ew/p)+(6.39+4.28*math.exp(-dp/107))*(p/ew-1)))*ew

  return 100*(p-ei_prime)*e_prime/((p-e_prime)*ei_prime)
end

--Main program
--
local PParam = param("P-PA")

local prod = configuration:GetSourceProducer(0)

if prod:GetId() == 131 then
  PParam = param("PGR-PA")
end

local Missing = missing

local p = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), PParam, current_forecast_type)
local t = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 2), param("T-K"), current_forecast_type)
local dp = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 2), param("TD-K"), current_forecast_type)

local i = 0
local res = {}

for i=1, #t do
  t[i] = t[i]-273.15
  dp[i] = dp[i]-273.15
  p[i] = p[i]/100.0

  if (t[i]<0) then
    res[i] = rhice(p[i],t[i],dp[i])
  else
    res[i] = Missing
  end
end

result:SetValues(res)
result:SetParam(param("RHICE-PRCNT"))
luatool:WriteToFile(result)

