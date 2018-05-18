--Round to natural number
function round(n)
  return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

--Main program
--

local MU = level(HPLevelType.kMaximumThetaE,0)
local HL = level(HPLevelType.kHeightLayer,500,0)
local HG = level(HPLevelType.kHeight,0)

EL500 = luatool:FetchWithType(current_time, HL, param("EL-HPA"), current_forecast_type)
LCL500 = luatool:FetchWithType(current_time, HL, param("LCL-HPA"), current_forecast_type)
LFC500 = luatool:FetchWithType(current_time, HL, param("LFC-HPA"), current_forecast_type)
CAPE500 = luatool:FetchWithType(current_time, HL, param("CAPE-JKG"), current_forecast_type)
CIN500 = luatool:FetchWithType(current_time, HL, param("CIN-JKG"), current_forecast_type)

LCLmu = luatool:FetchWithType(current_time, MU, param("LCL-HPA"), current_forecast_type)
LFCmu = luatool:FetchWithType(current_time, MU, param("LFC-HPA"), current_forecast_type)
ELmu = luatool:FetchWithType(current_time, MU, param("EL-HPA"), current_forecast_type)
CINmu = luatool:FetchWithType(current_time, MU, param("CIN-JKG"), current_forecast_type)
CAPEmu = luatool:FetchWithType(current_time, MU, param("CAPE-JKG"), current_forecast_type)

Ttop = luatool:FetchWithType(current_time, HL, param("EL-K"), current_forecast_type)
Tbase = luatool:FetchWithType(current_time, HL, param("LCL-K"), current_forecast_type)
TtopMU = luatool:FetchWithType(current_time, MU, param("EL-K"), current_forecast_type)
--LFC probably better than LCL for elev conv. base
TbaseMU = luatool:FetchWithType(current_time, MU, param("LFC-K"), current_forecast_type)


NL = luatool:FetchWithType(current_time, HG, param("NL-PRCNT"), current_forecast_type)
NM = luatool:FetchWithType(current_time, HG, param("NM-PRCNT"), current_forecast_type)

if not NL then
  NL = luatool:FetchWithType(current_time, HG, param("NL-0TO1"), current_forecast_type)
end

if not NM then
  NM = luatool:FetchWithType(current_time, HG, param("NM-0TO1"), current_forecast_type)
end

RR = luatool:FetchWithType(current_time, HG, param("RRR-KGM2"), current_forecast_type)

CBlimit = 9  --required vertical thickness [degrees C] to consider a CB (tweak this..!)
TCUlimit = 6  --required vertical thickness [degrees C] to consider a TCU (tweak this..!)
CBtopLim = 263.15 --required top T [K] to consider a CB (tweakable!)

--Max height [FL] to check for top
TopLim = 650

--Denominator to calculate overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
overshoot = 35

local i = 0
local res = {}
local Missing = missing

for i=1, #EL500 do

  res[i] = Missing

  --TCU
  if ((Tbase[i]-Ttop[i]>TCUlimit) and (NL[i]>0) and (CIN500[i]>-1)) then
    res[i] = FlightLevel_(EL500[i]*100)
    --Limit top value
    if (res[i] <= TopLim) then
      --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
      res[i] = -(res[i] + CAPE500[i]/overshoot)
    else
      --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
      res[i] = Missing
    end
  end

  --CB
  if ((Ttop[i]<CBtopLim) and (Tbase[i]-Ttop[i]>CBlimit) and (RR[i]>0)) then
    res[i] = FlightLevel_(EL500[i]*100)
    --Limit top value
    if (res[i] <= TopLim) then
      --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
      res[i] = res[i] + CAPE500[i]/overshoot
    else
      res[i] = Missing
    end
  end

  --If no TOP from above, check also with MU values, for elev. conv. only from blw 3,5km
  if ( IsMissing(res[i]) and (LFCmu[i]<LCL500[i]) and (LFCmu[i]>650)) then
    -- TCU
    if ((TbaseMU[i]-TtopMU[i]>TCUlimit) and ((NL[i]>0) or (NM[i]>0)) and (CINmu[i]>-1)) then
      res[i] =  FlightLevel_(ELmu[i]*100)
      --Limit top value
      if (res[i] <= TopLim) then
        --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
        res[i] = -(res[i] + CAPEmu[i]/overshoot)
      else
        res[i] = Missing
      end
    end
    --CB
    if ((TtopMU[i]<CBtopLim) and (TbaseMU[i]-TtopMU[i]>CBlimit) and (RR[i]>0)) then
      res[i] =  FlightLevel_(ELmu[i]*100)
      --Limit top value
      if (res[i] <= TopLim) then
        --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
        res[i] = res[i] + CAPEmu[i]/overshoot
      else
        res[i] = Missing
      end
    end
  end

  res[i] = round(res[i]/10)*10;
end

p = param("CBTCU-FL")

result:SetValues(res)
result:SetParam(p)
luatool:WriteToFile(result)
