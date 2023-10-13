function fieki(p,t)
  --Saturated vapor pressure of pure ice
  local ei=6.112*math.exp(22.46*t/(272.62+t))
  return (1+(1.0e-5*ei/(273+t))*((2100-65*t)*(1-ei/p)+(109-0.35*t+t*t/338)*(p/ei-1)))*ei
end

function ddt_fieki(p,t,h)
  --Finite difference approximation of derivative of fieki
  return (fieki(p,t+h/2)-fieki(p,t-h/2))/h
end

function fwekw(p,t)
  --Saturated vapor pressure of pure water
  local ew=6.112*math.exp(17.62*t/(243.12+t))
  return (1+(1.0e-4*ew/(273+t))*((38+173*math.exp(-t/43))*(1-ew/p)+(6.39+4.28*math.exp(-t/107))*(p/ew-1)))*ew
end

function fp(p,dp)
  --This function calculates frost point from pressure and dew point.

  --Initial frost point temperature guess is dew point temperature
  local ti=dp

  --Initial value for temperature delta
  local dti=999

  --step width for derivative approximation
  local h=0.01

  --Iterate
  while (math.abs(dti)>0.01) do
    --Use Newton's method to find frost point temp
    local f = fieki(p,ti)-fwekw(p,dp)
    local df_dt = ddt_fieki(p,ti,h)
    ti=ti-f/df_dt

    assert(math.abs(f/df_dt) < dti)
    dti=math.abs(f/df_dt)

  end
  return ti
end

--Main program
--

local PParam = param("P-PA")

local prod = configuration:GetSourceProducer(0)

if prod:GetId() == 131 then
  PParam = param("PGR-PA")
end

local Missing = missing

local P = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), PParam, current_forecast_type)
local DP = luatool:Fetch(current_time, level(HPLevelType.kHeight, 2), param("TD-K"), current_forecast_type)

local i = 0
local res = {}

for i=1, #P do
  --Convert pressure to hPa and temp to *C
  res[i] = fp(P[i]/100,DP[i]-273.15)
  if (res[i] > 0) then
    res[i] = Missing
  end
end

result:SetValues(res)
result:SetParam(param("TF-C"))
luatool:WriteToFile(result)
