-- Calculating three weather-related parameters
--
-- WeatherNumber:
-- An 8-digit number that describes the current weather. All digits are independent
-- of each other and describe a single parameter.
--
-- Example:
--
-- 12345678
-- xabcdefg
--
-- x (first digit): version number, currently 1
-- a: thunder probability (number of possible values: 3)
-- b: potential precipitation form (9)
-- c: potential precipitation type (3)
-- d: one hour precipitation sum (8)
-- e: fog (4)
-- f: total cloud cover (9)
-- g: cloud type (8)
--
-- The theoretical number of possible permutations of the parameter is:
-- 3 * 7 * 3 * 8 * 4 * 9 * 8 = 186624
--
-- SmartSymbol:
-- For a client application a weather parameter with this many possible values is just
-- unusable. Therefore we 'dumb down' the WeatherNumber to SmartSymbol. This procedure
-- takes all permutations of one weather state (for example 'Clear skies') and assigns just
-- one number for it. This reduces the number of different values to 47.
--
-- We still leave WeatherNumber as it is, because it can be used when for example 
-- doing interpolation for the data.
--
-- Hessaa:
-- For backwards compatibility, we map WeatherNumber to Hessaa (also called WeatherSymbol3) 

local MISS = missing
local SmartSymbolTable = {}
local HessaaTable = {}

function Fetch(...)
  local first = true
  local arg = {...}
  for i,param in ipairs(arg) do
    if #arg > 1 and i == #arg then
      return nil
    end

    ret = luatool:FetchWithType(current_time, current_level, param, current_forecast_type)

    if ret then
      if not first then
        for j=1,#ret do
          ret[j] = ret[j] * arg[#arg]
        end
      end
      return ret
    end

    first = false
  end
end

function DiscretizePOT(value)
  if value < 30 then
    return 0
  elseif value < 60 then
    return 1
  elseif value >= 60 then
    return 2
  end

  return MISS
end

function DiscretizeRR(value)
  if value < 0.025 then
    return 0
  elseif value < 0.04 then
    return 1
  elseif value < 0.4 then
    return 2
  elseif value < 1.5 then
    return 3
  elseif value < 2.0 then
    return 4
  elseif value < 4.0 then
    return 5
  elseif value < 7.0 then
    return 6
  elseif value >= 7.0 then 
    return 7
  end

  return MISS
end

function DiscretizeN(value)
  if value < 0.07 then
    return 0
  elseif value < 0.2 then
    return 1
  elseif value < 0.33 then
    return 2
  elseif value < 0.46 then
    return 3
  elseif value < 0.59 then
    return 4
  elseif value < 0.72 then
    return 5
  elseif value < 0.85 then
    return 6
  elseif value < 0.93 then 
    return 7
  elseif value >= 0.93 then
    return 8
  end

  return MISS
end

function MakeNumber(a, b, c, d, e, f, g)
  return 10000000 + -- version number
         1000000 * a +
         100000 * b +
         10000 * c +
         1000 * d +
         100 * e +
         10 * f +
         g
end

function Generate(an, bn, cn, dn, en, fn, gn, s1, h, s2)
  -- Generate all possible permutations of the weather number that 
  -- match to the given smartsymbol and hessaa values
  for a=an[1],an[2] do
    for b=bn[1],bn[2] do
      for c=cn[1],cn[2] do
        for d=dn[1],dn[2] do
          for e=en[1],en[2] do
            for f=fn[1],fn[2] do
              for g=gn[1],gn[2] do
                local number = MakeNumber(a, b, c, d, e, f, g)
                SmartSymbolTable[number] = s2
                HessaaTable[number] = h
              end
            end
          end
        end
      end
    end
  end
end

function CreateLookupTable()
  -- WeatherNumber must be 'dumbed down' to SmartSymbol: all possible permutations
  -- of a given weather scenario are matched to just one number.
  -- This makes deciphering the number client-side *much* easier.

  -- Variables used are: 
  -- a: pot  0-2
  -- b: pref 0-8
  -- c: pret 0-2
  -- d: rr   0-7
  -- e: fog  0-3
  -- f: n    0-8
  -- g: cldt 0-7

  -- last three arguments are:
  -- 3: original smartsymbol number
  -- 2: hessaa number
  -- 1: new smartsymbol number (ASI)

  -- Clear
  Generate({0,0},{0,8},{0,2},{0,0},{0,0},{0,1},{0,7},10000000,1,1)

  -- Mostly Clear
  Generate({0,0},{0,8},{0,2},{0,0},{0,0},{2,2},{0,7},10000020,1,2)

  -- Partly cloudy
  Generate({0,0},{0,8},{0,2},{0,0},{0,0},{3,5},{0,7},10000030,2,4)

  -- Mostly cloudy
  Generate({0,0},{0,8},{0,2},{0,0},{0,0},{6,7},{0,7},10000060,3,6)

  -- Overcast
  Generate({0,0},{0,8},{0,2},{0,0},{0,0},{8,8},{0,7},10000080,3,7)

  -- Fog
  Generate({0,0},{0,8},{0,2},{0,0},{1,2},{0,8},{0,7},10000100,3,9)

  -- Isolated thundershowers
  Generate({1,2},{0,8},{0,2},{0,7},{0,3},{0,5},{0,7},11000000,61,71)

  -- Scattered thundershowers
  Generate({1,2},{0,8},{0,2},{0,7},{0,3},{6,7},{0,7},11000060,63,74)

  -- Thundershowers
  Generate({1,2},{0,8},{0,2},{0,7},{0,3},{8,8},{0,7},11000080,63,77)

  -- Isolated showers
  Generate({0,0},{1,1},{2,2},{1,7},{0,3},{0,5},{0,7},10121000,21,21)

  -- Scattered showers
  Generate({0,0},{1,1},{2,2},{1,7},{0,3},{6,7},{0,7},10121060,31,24)

  -- Showers
  Generate({0,0},{1,1},{2,2},{1,7},{0,3},{8,8},{0,7},10121080,31,27)

  -- Freezing drizzle
  Generate({0,0},{4,4},{0,2},{1,7},{0,3},{0,8},{0,7},10401000,31,14)

  -- Freezing rain
  Generate({0,0},{5,5},{0,2},{1,7},{0,3},{0,8},{0,7},10501000,31,17)

  -- Drizzle
  Generate({0,0},{0,0},{0,2},{1,7},{0,3},{0,8},{0,7},10001000,31,11)

  -- Periods of light rain
  Generate({0,0},{1,1},{1,1},{1,2},{0,3},{0,5},{0,7},10111000,21,31)

  -- Periods of light rain
  Generate({0,0},{1,1},{1,1},{1,2},{0,3},{6,7},{0,7},10111060,31,34)

  -- Light rain
  Generate({0,0},{1,1},{1,1},{1,2},{0,3},{8,8},{0,7},10111080,31,37)

  -- Periods of moderate rain
  Generate({0,0},{1,1},{1,1},{3,5},{0,3},{0,5},{0,7},10113000,22,32)

  -- Periods of moderate rain
  Generate({0,0},{1,1},{1,1},{3,5},{0,3},{6,7},{0,7},10113060,32,35)

  -- Moderate rain
  Generate({0,0},{1,1},{1,1},{3,5},{0,3},{8,8},{0,7},10113080,32,38)

  -- Periods of heavy rain
  Generate({0,0},{1,1},{1,1},{6,7},{0,3},{0,5},{0,7},10116000,23,33)

  -- Periods of heavy rain
  Generate({0,0},{1,1},{1,1},{6,7},{0,3},{6,7},{0,7},10116060,33,36)

  -- Heavy rain
  Generate({0,0},{1,1},{1,1},{6,7},{0,3},{8,8},{0,7},10116080,33,39)

  -- Isolated light sleet showers
  Generate({0,0},{2,2},{0,2},{1,2},{0,3},{0,5},{0,7},10201000,71,41)

  -- Scattered light sleet showers
  Generate({0,0},{2,2},{0,2},{1,2},{0,3},{6,7},{0,7},10201060,81,44)

  -- Light sleet
  Generate({0,0},{2,2},{0,2},{1,2},{0,3},{8,8},{0,7},10201080,81,47)

  -- Isolated moderate sleet showers
  Generate({0,0},{2,2},{0,2},{3,3},{0,3},{0,5},{0,7},10203000,72,42)

  -- Scattered moderate sleet showers
  Generate({0,0},{2,2},{0,2},{3,3},{0,3},{6,7},{0,7},10203060,82,45)

  -- Moderate sleet
  Generate({0,0},{2,2},{0,2},{3,3},{0,3},{8,8},{0,7},10203080,82,48)

  -- Isolated heavy sleet showers
  Generate({0,0},{2,2},{0,2},{4,7},{0,3},{0,5},{0,7},10204000,72,43)

  -- Scattered heavy sleet showers
  Generate({0,0},{2,2},{0,2},{4,7},{0,3},{6,7},{0,7},10204060,83,46)

  -- Heavy sleet
  Generate({0,0},{2,2},{0,2},{4,7},{0,3},{8,8},{0,7},10204080,83,49)

  -- Isolated light snow showers
  Generate({0,0},{3,3},{0,2},{1,2},{0,3},{0,5},{0,7},10301000,41,51)

  -- Scattered light snow showers
  Generate({0,0},{3,3},{0,2},{1,2},{0,3},{6,7},{0,7},10301060,51,54)

  -- Light snowfall
  Generate({0,0},{3,3},{0,2},{1,2},{0,3},{8,8},{0,7},10301080,51,57)

  -- Isolated moderate snow showers
  Generate({0,0},{3,3},{0,2},{3,3},{0,3},{0,5},{0,7},10303000,42,52)

  -- Scattered moderate snow showers
  Generate({0,0},{3,3},{0,2},{3,3},{0,3},{6,7},{0,7},10303060,52,55)

  -- Moderate snowfall
  Generate({0,0},{3,3},{0,2},{3,3},{0,3},{8,8},{0,7},10303080,52,58)

  -- Isolated heavy snow showers
  Generate({0,0},{3,3},{0,2},{4,7},{0,3},{0,5},{0,7},10304000,43,53)

  -- Scattered heavy snow showers
  Generate({0,0},{3,3},{0,2},{4,7},{0,3},{6,7},{0,7},10304060,53,56)

  -- Heavy snowfall
  Generate({0,0},{3,3},{0,2},{4,7},{0,3},{8,8},{0,7},10304080,53,59)

  -- Isolated hail showers
  Generate({0,0},{6,6},{0,2},{1,7},{0,3},{0,5},{0,7},10601000,41,61)

  -- Scattered hail showers
  Generate({0,0},{6,6},{0,2},{1,7},{0,3},{6,7},{0,7},10601060,31,64)

  -- Hail showers
  Generate({0,0},{6,6},{0,2},{1,7},{0,3},{8,8},{0,7},10601080,31,67)

  -- Snow or ice grains
  Generate({0,0},{7,8},{0,2},{1,7},{0,3},{0,8},{0,7},10301080,51,57)

  -- for k,v in pairs(HessaaTable) do print(k,v) end
end

function WeatherNumber()
  logger:Debug("Calculating WeatherNumber")

  local POT  = Fetch(param("POT-PRCNT"))
  local PREF = Fetch(param("POTPRECF-N"))
  local PRET = Fetch(param("POTPRECT-N"))
  local RR   = Fetch(param("RRR-KGM2"))
  local FOG  = Fetch(param("FOGINT-N"))
  local N    = Fetch(param("N-0TO1"), param("N-PRCNT"), 0.01)
  -- local CLDT = Fetch(param("CLDTYPE-N"))

  if not POT or not PREF or not PRET or not RR or not FOG or not N then
    logger:Error("Aborting WeatherNumber calculation")
    return
  end

  local WeatherNumber = {}

  for i=1,#N do
    local pot  = DiscretizePOT(POT[i])
    local pref = math.floor(PREF[i])
    local pret = math.floor(PRET[i])
    local rr   = DiscretizeRR(RR[i])
    local fog  = math.floor(FOG[i])
    local n    = DiscretizeN(N[i])
    local cldt = 0 -- disabled for now

    WeatherNumber[i] = MakeNumber(pot, pref, pret, rr, fog, n, cldt)
  end

  result:SetParam(param("WEATHERNUMBER-N"))
  result:SetValues(WeatherNumber)
  luatool:WriteToFile(result)

  return WeatherNumber
end

function SmartSymbol(WeatherNumber)
  logger:Debug("Calculating SmartSymbol")

  CreateLookupTable()

  SmartSymbol = {}

  for i=1,#WeatherNumber do
    local number = WeatherNumber[i]
    local symbol = MISS

    if IsValid(number) then
      symbol = SmartSymbolTable[number]

      if symbol == nil then
        -- Actually this should not happen ever!
        logger:Warning(string.format("number %d has no match in table!", number))
      end
    end

    SmartSymbol[i] = symbol
  end

  result:SetParam(param("SMARTSYMBOL-N"))
  result:SetValues(SmartSymbol)
  luatool:WriteToFile(result)
end

function Hessaa(WeatherNumber)
  logger:Debug("Calculating Hessaa")

  Hessaa = {}

  for i=1,#WeatherNumber do
    local number = WeatherNumber[i]
    local hessaa = MISS

    if IsValid(number) then
      hessaa = HessaaTable[number]
    end

    Hessaa[i] = hessaa
  end

  result:SetParam(param("HESSAA-N"))
  result:SetValues(Hessaa)
  luatool:WriteToFile(result)
end

number = WeatherNumber()

if number then
  SmartSymbol(number)

  if not configuration:Exists("hessaa") or configuration:GetValue("hessaa") == "true" then
    Hessaa(number)
  else
    logger:Info("Hessaa calculation disabled")
  end
end
