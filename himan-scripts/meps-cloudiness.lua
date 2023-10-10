-- Cloudiness for MEPS / MNWC forecasts.
-- Will reduce total cloud cover if it is dominated by high clouds.
-- This is closer to "percieved cloudiness".
-- Original macro: Jesse H 12.5.2020
-- For Himan/lua: partio 15.5.2020
--

local o = {forecast_time = current_time,
           level = level(HPLevelType.kPressure, 300),
           params = params.create({param("RH-0TO1"), param("RH-PRCNT")}),
           forecast_type = current_forecast_type
}

local RH300 = luatool:FetchWithArgs(o)

o["level"] = level(HPLevelType.kPressure, 500)
local RH500 = luatool:FetchWithArgs(o)

o["level"] = level(HPLevelType.kPressure, 300)
o["params"] = params.create({param("N-0TO1"), param("N-PRCNT")})
local N300 = luatool:FetchWithArgs(o)

o["level"] = level(HPLevelType.kPressure, 500)
local N500 = luatool:FetchWithArgs(o)

o["level"] = level(HPLevelType.kHeight, 0)
o["params"] = params.create({param("NL-0TO1"), param("NL-PRCNT")})

local NL = luatool:FetchWithArgs(o)

o["params"] = params.create({param("NM-0TO1"), param("NM-PRCNT")})
local NM = luatool:FetchWithArgs(o)

o["params"] = params.create({param("NH-0TO1"), param("NH-PRCNT")})
local NH = luatool:FetchWithArgs(o)

if not RH300 or not RH500 or not N300 or not N500 or not NL or not NM or not NH then
  return
end

Nscale = 100
RHscale = 100

if math.max(RH500) > 2 then
  RHscale = 1
end

if math.max(N500) > 2 then
  RHscale = 1
end


local N = {}
local CH = {}

for i=1,#N300 do
  local n500 = N500[i] * Nscale
  local n300 = N300[i] * Nsacle

  local rh500 = RH500[i] * RHscale
  local rh300 = RH300[i] * RHscale

  local nl = NL[i] * Nscale
  local nm = NM[i] * Nscale
  local nh = NH[i] * Nscale

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
