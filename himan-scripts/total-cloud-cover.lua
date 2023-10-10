-- Cloudiness for MEPS / MNWC forecasts.
-- Will reduce total cloud cover if it is dominated by high clouds.
-- This is closer to "percieved cloudiness".
-- Original macro: Jesse H 12.5.2020
-- For Himan/lua: partio 15.5.2020
--

local RH300 = luatool:FetchWithType(current_time, level(HPLevelType.kPressure, 300), param("RH-0TO1"), current_forecast_type)
local RH500 = luatool:FetchWithType(current_time, level(HPLevelType.kPressure, 500), param("RH-0TO1"), current_forecast_type)
local N300 = luatool:FetchWithType(current_time, level(HPLevelType.kPressure, 300), param("N-0TO1"), current_forecast_type)
local N500 = luatool:FetchWithType(current_time, level(HPLevelType.kPressure, 500), param("N-0TO1"), current_forecast_type)
local NL = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("NL-0TO1"), current_forecast_type)
local NM = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("NM-0TO1"), current_forecast_type)
local NH = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("NH-0TO1"), current_forecast_type)

if not RH300 or not RH500 or not N300 or not N500 or not NL or not NM or not NH then
  return
end

local N = {}
local CH = {}

for i=1,#N300 do
  local n500 = N500[i] * 100
  local n300 = N300[i] * 100

  local rh500 = RH500[i] * 100
  local rh300 = RH300[i] * 100

  local nl = NL[i] * 100
  local nm = NM[i] * 100
  local nh = NH[i] * 100

  -- "out" parameters

  local n = missing
  local ch = missing

  -- YLÄPILVET (Eli painepinnat 500 ja 300) ; Yläpilvien osalta saattaa tarvita hiomista(?)
  -- Kun yläpilvisyyden arvo on pieni ; CH < 20

  if (math.min(n500, n300, nh) < 20) then
    ch = nh * 0.9 + ((rh500 + rh300) * 0.5) * 0.1
  end

  -- Kun yläpilvisyyden arvo on "puolipilvinen" ; 20 < CH < 70

  if ((n500 >= 20 and n500 < 70) or (n300 >= 20 and n300 < 70) or (nh >= 20 and nh < 70)) then
    ch = nh * 0.4 + ((rh500 + rh300) * 0.5) * 0.6
  end

  -- Kun yläpilvisyyden arvo on suurta ; CH > 70

  if (math.max(n500, n300, nh) >= 70) then
    ch = nh * 0.5 + ((rh500 + rh300) * 0.5) * 0.4
  end

  -- total cloudiness is the largest value of the layers
  
  n = math.max(ch, nm, nl)

  -- YLÄPILVIEN OSALTA VÄHENNETÄÄN PILVISYYTTÄ, KUN YLÄPILVISYYS DOMINOI (10.7.2018 Lisätty) ; Karkea leikkaus

  if (ch >= nl + nm and nl + nm < 60) then
    n = ch * 0.5
  end

  CH[i] = ch
  N[i] = n
end 

result:SetParam(param("N-PRCNT"))
result:SetValues(N)
luatool:WriteToFile(result)

result:SetParam(param("NH-PRCNT"))
result:SetValues(CH)
luatool:WriteToFile(result)
