--Round to natural number
function round(n)
  return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

local currentProducer = configuration:GetTargetProducer()
local currentProducerName = currentProducer.GetName(currentProducer)

local filter = matrixf(9, 9, 1, missing)
filter:Fill(1)

--Main program
--

local MU = level(HPLevelType.kMaximumThetaE,0)
local HL = level(HPLevelType.kHeightLayer,500,0)
local HG = level(HPLevelType.kHeight,0)

local EL500 = luatool:Fetch(current_time, HL, param("EL-LAST-M"), current_forecast_type)
local pEL500 = luatool:Fetch(current_time, HL, param("EL-LAST-HPA"), current_forecast_type)
local LCL500 = luatool:Fetch(current_time, HL, param("LCL-M"), current_forecast_type)
local CAPE500 = luatool:Fetch(current_time, HL, param("CAPE-JKG"), current_forecast_type)
local CIN500 = luatool:Fetch(current_time, HL, param("CIN-JKG"), current_forecast_type)
local LFCmu = luatool:Fetch(current_time, MU, param("LFC-M"), current_forecast_type)
local pLFCmu = luatool:Fetch(current_time, MU, param("LFC-HPA"), current_forecast_type)
local ELmu = luatool:Fetch(current_time, MU, param("EL-LAST-M"), current_forecast_type)
local pELmu = luatool:Fetch(current_time, MU, param("EL-LAST-HPA"), current_forecast_type)
local CINmu = luatool:Fetch(current_time, MU, param("CIN-JKG"), current_forecast_type)
local CAPEmu = luatool:Fetch(current_time, MU, param("CAPE-JKG"), current_forecast_type)
local Ttop = luatool:Fetch(current_time, HL, param("EL-K"), current_forecast_type)
local TtopMU = luatool:Fetch(current_time, MU, param("EL-K"), current_forecast_type)

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

local NL = luatool:Fetch(current_time, HG, param("NL-PRCNT"), current_forecast_type)
local NM = luatool:Fetch(current_time, HG, param("NM-PRCNT"), current_forecast_type)

if not NL then
  NL = luatool:Fetch(current_time, HG, param("NL-0TO1"), current_forecast_type)
end

if not NM then
  NM = luatool:Fetch(current_time, HG, param("NM-0TO1"), current_forecast_type)
end

local RR = luatool:Fetch(current_time, HG, param("RRR-KGM2"), current_forecast_type)

if not NL or not NM or not RR then
  logger:Error("Some data not found")
  return
end

if currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  local Nmat = matrixf(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
  Nmat:SetValues(EL500)
  EL500 = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(pEL500)
  pEL500 = Min2D(Nmat,filter,configuration:GetUseCuda()):GetValues()
  
  Nmat:SetValues(ELmu)
  ELmu = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()
  
  Nmat:SetValues(pELmu)
  pELmu = Min2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(RR)
  RR = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(NL)
  NL = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(NM)
  NM = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  filter:Fill(1/81)

  Nmat:SetValues(LCL500)
  LCL500 = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(CAPE500)
  CAPE500 = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(CIN500)
  CIN500 = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(LFCmu)
  LFCmu = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(pLFCmu)
  pLFCmu = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(CAPEmu)
  CAPEmu = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

  Nmat:SetValues(CINmu)
  CINmu = Filter2D(Nmat,filter,configuration:GetUseCuda()):GetValues()
end

local CBlimit = 2000  --required vertical thickness [m] to consider a CB (tweak this..!)
local TCUlimit = 1500  --required vertical thickness [m] to consider a TCU (tweak this..!)
local CBtopLim = 263.15  --required top T [K] (-10 degC) to consider a CB (tweakable!)
local CINlimTCU = -1  --CIN limit for TCu
local RRlimit = 0.1 -- precipitation limit [mm/h] to consider a Cb
local CAPElimit = 2.71828 --euler constant

local i = 0
local res = {}
local Missing = missing

for i=1, #EL500 do

  res[i] = Missing

  --TCU
  if ((EL500[i] - LCL500[i] > TCUlimit) and (NL[i] > 0) and (CIN500[i] > CINlimTCU) ) then
    --we don't use vertical search for flight level of EL500 but calculate directly from EL500 pressure
    res[i] = FlightLevel_(pEL500[i] * 100)
    --Limit top value
    if (CAPE500[i] > CAPElimit) then
      --Add for overshooting top based on CAPE 
      res[i] = -(res[i] + CAPE500[i]  / (math.log(CAPE500[i]) * 10))
    else
      res[i] = -res[i]
    end
  end

  --CB
  if ((Ttop[i] < CBtopLim) and (EL500[i] - LCL500[i] > CBlimit) and (RR[i] > RRlimit)) then
    res[i] = FlightLevel_(pEL500[i] * 100)
    --Limit top value
    if (CAPE500[i] > CAPElimit) then
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
      if (CAPEmu[i] > CAPElimit) then
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
      if (CAPEmu[i] > CAPElimit) then
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
