-- Calculating hsade1 aka WeatherSymbol1, ie. precipitation symbol from
-- WeatherNumber.
--
-- Possible values:

-- 0 = No rain
-- 50 = Drizzle, not freezing, intermittent, slight at time of ob
-- 53 = Drizzle, not freezing, continuous, moderate at time of ob
-- 56 = Drizzle, freezing, slight
-- 57 = Drizzle, freezing, moderate or heavy (dense)
-- 60 = Rain, not freezing, intermittent, slight at time of ob
-- 63 = Rain, not freezing, continuous, moderate at time of ob
-- 65 = Rain, not freezing, continuous, heavy at time of ob
-- 66 = Rain, freezing, slight
-- 67 = Rain, freezing, moderate or heavy
-- 68 = Rain or drizzle and snow, slight
-- 70 = Intermittent fall of snowflakes, slight at time of ob
-- 73 = Continuous fall of snowflakes, moderate at time of ob
-- 75 = Continuous fall of snowflakes, heavy at time of ob
-- 77 = Snow grains (with or without fog)
-- 79 = Ice pellets
-- 80 = Rain shower(s), slight
-- 81 = Rain shower(s), moderate
-- 82 = Rain shower(s), violent
-- 83 = Sleet shower(s), slight
-- 84 = Sleet shower(s), moderate or heavy
-- 85 = Snow shower(s), slight
-- 86 = Snow shower(s), moderate or heavy
-- 89 = Shower(s) of hail, with or without rain or rain and snow mixed, not associated with thunder, slight
-- 90 = Shower(s) of hail, with or without rain or rain and snow mixed, not associated with thunder, moderate or heavy
-- 95 = Thunderstorm, slight or moderate, without hail, but with rain and/or snow at time of observation
-- 97 = Thunderstorm, heavy, without hail, but with rain and/or snow at time of observation 

function GetHSade(wn)

  -- decoding weather number
  --
  -- 1. number: version number
  -- 2. number: probability of thunder
  -- 3. number: precipitation form
  -- 4. number: precipitation type
  -- 5. number: precipitation intensity
  -- 6. number: fog
  -- 7. number: cloud cover
  -- 8. number: cloud type

  local pot  = math.floor((wn / 1000000) % 10) 
  local pref = math.floor((wn / 100000) % 10) 
  local pret = math.floor((wn / 10000) % 10) 
  local rr   = math.floor((wn / 1000) % 10) 

  local hsade = 0 -- no rain

  -- ukkonen
  if (pot == 1)  then
    hsade = 95
  elseif (pot == 2) then
    hsade = 97

  -- tihkusade
  elseif (pref == 0 and pot == 0 and rr == 1) then
    hsade = 50 -- Heikko tihkusade
  elseif (pref == 0 and pot == 0 and rr >= 2 and rr <= 7) then
    hsade = 53 -- Kohtalainen tihkusade

  -- vesisade
  elseif (pref == 1 and pot == 0 and pret == 1 and rr >= 1 and rr <= 2) then
    hsade = 60 -- Heikko vesisade
  elseif (pref == 1 and pot == 0 and pret == 1 and rr >= 3 and rr <= 5) then
    hsade = 63 -- Kohtalainen vesisade
  elseif (pref == 1 and pot == 0 and pret == 1 and rr >= 6 and rr <= 7) then
    hsade = 65 -- Voimakas vesisade

  -- vesikuuro
  elseif (pref == 1 and pot == 0 and pret == 2 and rr >= 1 and rr <= 2) then
    hsade = 80 -- Heikko vesikuuro
  elseif (pref == 1 and pot == 0 and pret == 2 and rr >= 3 and rr <= 5) then
    hsade = 81 -- Kohtalainen vesikuuro
  elseif (pref == 1 and pot == 0 and pret == 2 and rr >= 6 and rr <= 7) then
    hsade = 82 -- Voimakas vesikuuro

  -- räntäsade
  elseif (pref == 2 and pot == 0 and pret == 1) then
    hsade = 68

  -- räntäkuuro
  elseif (pref == 2 and pot == 0 and pret == 2 and rr >= 1 and rr <= 2) then
    hsade = 83
  elseif (pref == 2 and pot == 0 and pret == 2 and rr >= 3 and rr <= 7) then
    hsade = 84

  -- lumisade
  elseif (pref == 3 and pot == 0 and pret == 1 and rr >= 1 and rr <= 2) then
    hsade = 70 -- heikko lumisade
  elseif (pref == 3 and pot == 0 and pret == 1 and rr == 3) then
    hsade = 73 -- kohtalainen lumisade
  elseif (pref == 3 and pot == 0 and pret == 1 and rr >= 4 and rr <= 7) then
    hsade = 75 -- runsas lumisade

  -- lumikuuro
  elseif (pref == 3 and pot == 0 and pret == 2 and rr >= 1 and rr <= 2) then
    hsade = 85 -- heikko lumikuuro
  elseif (pref == 3 and pot == 0 and pret == 2 and rr >= 3 and rr <= 7) then
    hsade = 86 -- kohtalainen tai sakea lumikuuro

  -- jäätävä tihku
  elseif (pref == 4 and pot == 0 and rr >= 1 and rr <= 5) then
    hsade = 56 -- jäätävä tihku
  elseif (pref == 4 and pot == 0 and rr >= 6 and rr <= 7) then
    hsade = 57 -- voimakas jäätävä tihku

  -- jäätävä sade
  elseif (pref == 5 and pot == 0 and rr >= 1 and rr <= 5) then
    hsade = 66 -- jäätävä sade
  elseif (pref == 5 and pot == 0 and rr >= 6 and rr <= 7) then
    hsade = 67 -- voimakas jäätävä sade

  -- rakeet
  elseif (pref == 6 and pot == 0 and rr >= 1 and rr <= 2) then
    hsade = 89 -- heikko raekuuro
  elseif (pref == 6 and pot == 0 and rr >= 3 and rr <= 7) then
    hsade = 90 -- kohtalainen tai kova raekuuro

  -- lumijyvänen
  elseif (pref == 7 and pot == 0 and rr >= 1 and rr <= 7) then
    hsade = 77

  -- jääjyvänen
  elseif (pref == 8 and pot == 0 and rr >= 1 and rr <= 7) then
    hsade = 79
  end

  return hsade

end

local MISS = missing

logger:Debug("Calculating Hsade1")

local WN = luatool:Fetch(current_time, current_level, param("WEATHERNUMBER-N"), current_forecast_type)

if not WN then
  return
end

local HSADE1 = {}

for i=1,#WN do
  local wn = WN[i]
  local hsade = MISS

  if (IsValid(wn)) then
    local version = math.floor((wn / 10000000) % 10)

    if (version ~= 1) then
      logger:Error(string.format("WeatherNumber version %d found, not supported", version))
      return
    end

    hsade = GetHSade(wn)
  end

  HSADE1[i] = hsade
end

result:SetParam(param("HSADE1-N"))
result:SetValues(HSADE1)
luatool:WriteToFile(result)
