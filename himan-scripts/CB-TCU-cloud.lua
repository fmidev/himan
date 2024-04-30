--Round to natural number
function round(n)
  return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

--Main program
--

local MU = level(HPLevelType.kMaximumThetaE,0)
local HL = level(HPLevelType.kHeightLayer,500,0)
local HG = level(HPLevelType.kHeight,0)

EL500 = luatool:Fetch(current_time, HL, param("EL-LAST-M"), current_forecast_type)
pEL500 = luatool:Fetch(current_time, HL, param("EL-LAST-HPA"), current_forecast_type)
LCL500 = luatool:Fetch(current_time, HL, param("LCL-M"), current_forecast_type)
CAPE500 = luatool:Fetch(current_time, HL, param("CAPE-JKG"), current_forecast_type)
CIN500 = luatool:Fetch(current_time, HL, param("CIN-JKG"), current_forecast_type)
LFCmu = luatool:Fetch(current_time, MU, param("LFC-M"), current_forecast_type)
pLFCmu = luatool:Fetch(current_time, MU, param("LFC-HPA"), current_forecast_type)
ELmu = luatool:Fetch(current_time, MU, param("EL-LAST-M"), current_forecast_type)
pELmu = luatool:Fetch(current_time, MU, param("EL-LAST-HPA"), current_forecast_type)
CINmu = luatool:Fetch(current_time, MU, param("CIN-JKG"), current_forecast_type)
CAPEmu = luatool:Fetch(current_time, MU, param("CAPE-JKG"), current_forecast_type)
Ttop = luatool:Fetch(current_time, HL, param("EL-K"), current_forecast_type)
TtopMU = luatool:Fetch(current_time, MU, param("EL-K"), current_forecast_type)

if not EL500 or 
   not pEL500 or 
   not LCL500 or 
   not CAPE500 or
   not CIN500 or
   not LFCmu or
   not pLFCmu or
   not ELmu or
   not pELmu or
   not CINmu or
   not CAPEmu or
   not Ttop or 
   not TtopMU then
  logger:Error("Some data not found")
  return
end

NL = luatool:Fetch(current_time, HG, param("NL-PRCNT"), current_forecast_type)
NM = luatool:Fetch(current_time, HG, param("NM-PRCNT"), current_forecast_type)

if not NL then
  NL = luatool:Fetch(current_time, HG, param("NL-0TO1"), current_forecast_type)
end

if not NM then
  NM = luatool:Fetch(current_time, HG, param("NM-0TO1"), current_forecast_type)
end

RR = luatool:Fetch(current_time, HG, param("RRR-KGM2"), current_forecast_type)

if not NL or not NM or not RR then
  logger:Error("Some data not found")
  return
end

CBlimit = 2000  --required vertical thickness [degrees C] to consider a CB (tweak this..!)
TCUlimit = 1000  --required vertical thickness [degrees C] to consider a TCU (tweak this..!)
CBtopLim = -10  --required top T [K] to consider a CB (tweakable!)
CINlimTCU = -1  --CIN limit for TCu
RRlimit = 0.1 -- precipitation limit [mm/h] to consider a Cb

local i = 0
local res = {}
local Missing = missing

for i=1, #EL500 do

  res[i] = Missing

  --TCU
  if (EL500[i] - LCL500[i] > TCUlimit) then
    --we don't use vertical search for flight level of EL500 but calculate directly from EL500 pressure
    res[i] = FlightLevel_(pEL500[i] * 100)
    --Limit top value
    if (CAPE500[i] > math.exp(1)) then
      --Add for overshooting top based on CAPE 
      res[i] = -(res[i] + CAPE500[i]  / (math.log(CAPE500[i]) * 10))
    else
      res[i] = -res[i]
    end
  end

  --CB
  if ((Ttop[i] < CBtopLim) and (EL500[i] - LCL500[i] > Cblimit) and (RR[i] > RRlimit)) then
    res[i] = FlightLevel_(pEL500[i] * 100)
    --Limit top value
    if (CAPE500[i] > math.exp(1)) then
      --Add for overshooting top based on CAPE
      res[i] = res[i] + CAPE500[i] / (math.log(CAPE500[i]) * 10)
    end
  end

  --If no TOP from above, check also with MU values, for elev. conv. only from blw 3,5km
  if ( IsMissing(res[i]) and (LFCmu[i] > LCL500[i]) and (pLFCmu[i] > 650)) then
    -- TCU elevated
    if ((ELmu[i] - LFCmu[i] > TCUlimit) and ((NL[i] > 0) or (NM[i] > 0)) and (CINmu[i] > CINlimTCU)) then
      res[i] =  FlightLevel_(pELmu[i] * 100)
      --Limit top value
      if (CAPEmu[i] > math.exp(1)) then
        --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
        res[i] = -(res[i] + CAPEmu[i] / (math.log(CAPEmu[i]) * 10))
      else
        res[i] = -res[i]
      end
    end
    --CB elevated
    if ((TtopMU[i] < CBtopLim) and (ELmu[i] - LFCmu[i] > CBlimit) and (RR[i] > RRlimit)) then
      res[i] =  FlightLevel_(pELmu[i] * 100)
      --Limit top value
      if (CAPEmu[i] > math.exp(1)) then
        --Add for overshooting top based on CAPE, +1000ft/350J/kg (tweak this!)
        res[i] = res[i] + CAPEmu[i] / (math.log(CAPEmu[i]) * 10)
      end
    end
  end

  res[i] = round(res[i]/10)*10
end

p = param("CBTCU-FL")

result:SetValues(res)
result:SetParam(p)
luatool:WriteToFile(result)
