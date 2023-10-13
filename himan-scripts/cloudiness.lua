-- Pilvisyys jatkopäiville (Jatkojalostettu aiemmista N+RR -pilvimakroista)
-- Jesse 20.3.2018
-- 20.3.2018
-- For Himan/lua: partio / 2019-02-04
--
-- Alapilvet; EC: LCC - Low Cloud Cover 1-0.8 of the surface pressure (Noin 1000 - 850 hPa)
-- Keskipilvet; EC: MCC - Medium Cloud Cover 0.8-0.45 of the surface pressure (Noin 850 - 500 hPa)
-- Yläpilvet; EC: HCC - High Cloud Cover 0.45 to the model top of the surface pressure (Noin 500 - 300 hPa)

local N = param("N-0TO1")

hitool:SetHeightUnit(HPParameterUnit.kHPa)

local N300 = hitool:VerticalValue(N, 300)
local N500 = hitool:VerticalValue(N, 500)
local N700 = hitool:VerticalValue(N, 700)
local N850 = hitool:VerticalValue(N, 850)
local N925 = hitool:VerticalValue(N, 925)
local RH300 = luatool:Fetch(current_time, level(HPLevelType.kPressure, 300), param("RH-PRCNT"), current_forecast_type)
local RH500 = luatool:Fetch(current_time, level(HPLevelType.kPressure, 500), param("RH-PRCNT"), current_forecast_type)
local RH700 = luatool:Fetch(current_time, level(HPLevelType.kPressure, 700), param("RH-PRCNT"), current_forecast_type)
local RH850 = luatool:Fetch(current_time, level(HPLevelType.kPressure, 850), param("RH-PRCNT"), current_forecast_type)
local RH925 = luatool:Fetch(current_time, level(HPLevelType.kPressure, 925), param("RH-PRCNT"), current_forecast_type)
local NL = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("NL-0TO1"), current_forecast_type)
local NM = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("NM-0TO1"), current_forecast_type)
local NH = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("NH-0TO1"), current_forecast_type)
local PRET = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("PRECFORM2-N"), current_forecast_type)
local RRR = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("RRR-KGM2"), current_forecast_type)

if not N300 or not N500 or not N700 or not N850 or not N925 or not RH300 or not RH500 or not RH700 or not RH850 or not RH925 or not NL or not NM or not NH then
  return
end

local TOTN = {}
local CL = {}
local CM = {}
local CH = {}

for i=1,#N300 do
  local N_EC_925 = N925[i] * 100
  local N_EC_850 = N850[i] * 100
  local N_EC_700 = N700[i] * 100
  local N_EC_500 = N500[i] * 100
  local N_EC_300 = N300[i] * 100

  local RH_EC_925 = RH925[i]
  local RH_EC_850 = RH850[i]
  local RH_EC_700 = RH700[i]
  local RH_EC_500 = RH500[i]
  local RH_EC_300 = RH300[i]

  local CL_EC = NL[i] * 100
  local CM_EC = NM[i] * 100
  local CH_EC = NH[i] * 100

  local RR = RRR[i]
  local PT = PRET[i]

  local cl = CL_EC
  local cm = CM_EC
  local ch = CH_EC
  local N = missing

-- ALAPILVET (Käytetty painepintoja 850 ja 925)

-- Kun alapilvisyyden arvo on pieni ; CL < 20

  if (N_EC_850 < 20 and N_EC_925 < 20 and CL_EC < 20) then
    cl = (CL_EC)*0.9 + ((RH_EC_850+RH_EC_925)/2)*0.1        -- Vaihdettu kertoimet 10.7.2018 0.8 ja 0.2 -> 0.9 ja 0.1
  end
 
  -- Kun alapilvisyyden tai kokonaispilvisyyden arvo on "puolipilvinen" ; 20 < CL < 70

  if ((N_EC_850 >= 20 and N_EC_850 < 70) or (N_EC_925 >= 20 and N_EC_925 < 70) or (CL_EC >= 20 and CL_EC < 70)) then
    cl = (CL_EC)*0.8 +((RH_EC_850+RH_EC_925)/2)*0.2
  end

  -- Kun alapilvisyys tai kokonaispilvisyys on suurta ; CL > 70

  if (N_EC_850 >= 70 or N_EC_925 >= 70 or CL_EC >= 70) then
    cl = (CL_EC)*0.8 + ((RH_EC_850+RH_EC_925)/2)*0.2    -- N_EC vaihdettu pintojen 925 ja 850 keskiarvoksi 2.5.2018
  end

  -- KESKIPILVET (Käytetty painepintaa 700)

  -- Kun keskipilvisyyden arvo on pieni ; CM < 20

  if (N_EC_700 < 20 and CM_EC < 20) then
    cm = (CM_EC)*0.9 + (RH_EC_700)*0.1        -- Vaihdettu kertoimet 10.7.2018 ; 0.8 ja 0.2 -> 0.9 ja 0.1
  end

  -- Kun keskipilvisyyden arvo on "Puolipilvinen" ; 20 < CM < 70

  if (N_EC_700 >=20 and N_EC_700 < 70 and CM_EC >= 20 and CM_EC < 70) then
    cm = (CM_EC)*0.7 + (RH_EC_700)*0.3
  end

  -- Kun keskipilvisyys on suurta ; CM > 70

  if (N_EC_700 >= 70 or CM_EC >= 70) then
    cm = (CM_EC)*0.8 + (RH_EC_700)*0.2        -- Kertoimia säädetty 29.11.2018 0.4 ja 0.6 -> 0.8 ja 0.2
  end

  -- YLÄPILVET (Eli painepinnat 500 ja 300) ; Yläpilvien osalta saattaa tarvita hiomista(?)

  -- Kun yläpilvisyyden arvo on pieni ; CH < 20

  if (N_EC_500 < 20 and N_EC_300 < 20 and CH_EC < 20) then
    ch = (CH_EC)*0.9 + ((RH_EC_500+RH_EC_300)/2)*0.1        -- Lisätty heikko RH riippuvuus
  end

  -- Kun yläpilvisyyden arvo on "puolipilvinen" ; 20 < CH < 70

  if ((N_EC_500 >= 20 and N_EC_500 < 70) or (N_EC_300 >= 20 and N_EC_300 < 70) or (CH_EC >= 20 and CH_EC < 70)) then
    ch = (CH_EC)*0.4+((RH_EC_500+RH_EC_300)/2)*0.6
  end

  -- Kun yläpilvisyyden arvo on suurta ; CH > 70

  if (N_EC_500 >= 70 or N_EC_300 >= 70 or CH_EC >= 70) then
    ch = (CH_EC)*0.5+((RH_EC_500+RH_EC_300)/2)*0.4
  end

  ch = math.min(ch, 100)
  cm = math.min(cm, 100)
  cl = math.min(cl, 100)
  N = math.max(ch, cm, cl)

  -- YLÄPILVIEN OSALTA VÄHENNETÄÄN PILVISYYTTÄ, KUN YLÄPILVISYYS DOMINOI (10.7.2018 Lisätty) ; Karkea leikkaus

  if (ch >= cl+cm and cl+cm < 60) then
    N = ch/2
  end

  -- PILVISYYDEN LISÄYS JOS SATAA

  if (RR > 0 and N < 50) then
    N = 50
  end

  if (RR > 0.2 and PT == 1 and N < 100) then
    N = 100
  end

  CL[i] = cl
  CM[i] = cm
  CH[i] = ch
  TOTN[i] = N
end 

-- AIKA- JA HILATASOITUS
local filter = matrix(3, 3, 1, missing)
filter:Fill(1/9)
local Nmat = matrix(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
Nmat:SetValues(TOTN)

local Nfilt = Filter2D(Nmat, filter, configuration:GetUseCuda())

result:SetParam(param("N-PRCNT"))
result:SetValues(Nfilt:GetValues())
luatool:WriteToFile(result)

result:SetParam(param("NL-PRCNT"))
result:SetValues(CL)
luatool:WriteToFile(result)

result:SetParam(param("NM-PRCNT"))
result:SetValues(CM)
luatool:WriteToFile(result)

result:SetParam(param("NH-PRCNT"))
result:SetValues(CH)
luatool:WriteToFile(result)
