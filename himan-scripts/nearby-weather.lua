function max (...)
  maxindex = 0
  maxvalue = 0
  for i,v in ipairs(arg) do
    if (v > maxvalue) then
      maxindex = i
      maxvalue = v
    end
  end
  
  return maxindex, maxvalue
end

--Main program
--

local Missing = missing

--Get the weather number
local w = luatool:FetchInfo(current_time, current_level, param("WEATHERNUMBER-N"))

local Weather = w:GetValues()

--Get gust
local g = luatool:FetchInfo(current_time, level(HPLevelType.kGround,0), param("FFG-MS"))
local gust = g:GetValues()

if not Weather or not gust then
  return
end

local thunder = {}
local prec_form = {}
local prec_type = {}
local prec_intensity = {}
local fog = {}
local cloud_cover = {}
local cloud_type = {}

for i=1, #Weather do
  --chop weather number into pieces
  thunder[i] = math.floor((Weather[i]/10^6)%10)
  prec_form[i] = math.floor((Weather[i]/10^5)%10)
  prec_type[i] = math.floor((Weather[i]/10^4)%10)
  prec_intensity[i] = math.floor((Weather[i]/10^3)%10)
  fog[i] = math.floor((Weather[i]/10^2)%10)
  cloud_cover[i] = math.floor((Weather[i]/10)%10)
  cloud_type[i] = math.floor(Weather[i]%10)

  if (prec_intensity[i] == 0 ) then
    prec_form[i] = Missing
    prec_type[i] = Missing
  end
end

-- todo set a mask matrix to match 50/100km radius in smartmet edited data
local rain_radius = matrix(13, 13, 1, Missing)
rain_radius:Fill(1)
local wind_radius = matrix(27, 27, 1, Missing)
wind_radius:Fill(1)

-------------
-- THUNDER --
-------------
local tmp = w:GetData()
tmp:SetValues(thunder)

local thunder_awareness = {}
local thunder_percentage = ProbLimitGe2D(tmp, rain_radius, 1):GetValues()


for i=1, #thunder_percentage do
  thunder_awareness[i] = 0

  if (thunder_percentage[i] >= 0.05) then
    thunder_awareness[i] = 1
  end

  if (thunder_percentage[i] >= 0.30) then
    thunder_awareness[i] = 2
  end
end

----------
-- RAIN --
----------

local rain = {}
local dry_limit = {}  
local rainy_weather = {}
local rain_frequency = {}

tmp:SetValues(prec_intensity)

local dry = ProbLimitEq2D(tmp, rain_radius, 0):GetValues()

dry_limit = 0.02
for i=1, #dry do
  rain[i] = 1 - dry[i]

  if (rain[i] >= dry_limit ) then
    rainy_weather[i] =1

    if(rain[i] < 0.50) then 
      rain_frequency[i] = 1
    elseif(rain[i] >= 0.50 and rain[i] < 0.90) then
      rain_frequency[i] = 2
    else
      rain_frequency[i] = 3
    end
  end
end

--------------------
-- RAIN INTENSITY --
--------------------

local weak = ProbLimitEq2D(tmp, rain_radius, 1):GetValues() 
local weak2 = ProbLimitEq2D(tmp, rain_radius, 2):GetValues()

for i=1, #weak do
  weak[i] = weak[i] + weak2[i]
end

local strong_water = ProbLimitGt2D(tmp, rain_radius, 5):GetValues()
local strong_snow = ProbLimitGe2D(tmp, rain_radius, 3):GetValues()

------------------------
-- PRECIPITATION TYPE --
------------------------

-- Laskee vallitsevan sateen tyypin alueelle
tmp:SetValues(prec_type)
rain_radius:Fill(1/169) --set weights adding up to 1 for mean filtering

local prevalent_type = Filter2D(tmp, rain_radius, false):GetValues()

-- round to whole number
for i=1, #prevalent_type do
    prevalent_type[i] = math.floor(prevalent_type[i] + 0.5 )
end

------------------------
-- PRECIPITATION FORM --
------------------------

local prevalent_form = {}
local prevalent_cover = {}
 
tmp:SetValues(prec_form)
rain_radius:Fill(1) --reset weights to original state

local drizzle = ProbLimitEq2D(tmp, rain_radius, 0):GetValues()
local water = ProbLimitEq2D(tmp, rain_radius, 1):GetValues() 
local sleet = ProbLimitEq2D(tmp, rain_radius, 2):GetValues()
local snow = ProbLimitEq2D(tmp, rain_radius, 3):GetValues()
local freezingdrizzle = ProbLimitEq2D(tmp, rain_radius, 4):GetValues()
local freezingrain = ProbLimitEq2D(tmp, rain_radius, 5):GetValues()
local hail = ProbLimitEq2D(tmp, rain_radius, 6):GetValues()
local snowgrain = ProbLimitEq2D(tmp, rain_radius, 7):GetValues()
local icegrain = ProbLimitEq2D(tmp, rain_radius, 8):GetValues()

-- this can be done by simply picking the one from above with highest probability 
for i=1, #drizzle do
  hail[i] = hail[i] + snowgrain[i] + freezingrain[i]
  freezingrain[i] = freezingrain[i] + freezingdrizzle[i]

  if(prec_intensity[i] > 0) then
    prevalent_form[i],prevalent_cover[i] = max(drizzle[i],water[i],sleet[i],snow[i],freezingdrizzle[i],freezingrain[i],hail[i],snowgrain[i],icegrain[i])
  else
    prevalent_form[i] = Missing
    prevalent_cover[i] = Missing
  end

  -- does this make sense? icerain is an area-probability, prevalent_form is a numeral.
  if (freezingrain[i] > drizzle[i] or freezingrain[i] > water[i] or freezingrain[i] > sleet[i] or freezingrain[i] > snow[i]) then
    prevalent_form[i] = 5
  elseif (hail[i] > drizzle[i] or hail[i] > water[i] or hail[i] > sleet[i] or hail[i] > snow[i]) then
    prevalent_form[i] = 6
  end
end

---------
-- FOG --
---------

local fog_index = {}

tmp:SetValues(fog)
local fog_share = ProbLimitGe2D(tmp, rain_radius, 1):GetValues()

for i=1, #fog_share do
  fog_index[i] = 0
  if (fog_share[i] > 0.20) then
    if (fog_share[i] < 0.40) then
      fog_index[i] = 1
    else
      if (fog_share[i] < 0.75) then
        fog_index[i] = 2
      else
        fog_index[i] = 3
      end
    end
  end
end

-----------------------
-- CLOUDINESS AMOUNT --
-----------------------

-- use mean
tmp:SetValues(cloud_cover)
rain_radius:Fill(1/169) --set weight for mean filter
local MedianCloudiness = Filter2D(tmp, rain_radius, false):GetValues()
rain_radius:Fill(1) --reset original value
local MinCloudiness = Min2D(tmp, rain_radius, false):GetValues()
local MaxCloudiness = Max2D(tmp, rain_radius, false):GetValues()

-- Pilvinen (pilvisyys >= 7)
local Cloudy =  ProbLimitGe2D(tmp, rain_radius, 7):GetValues() 
-- Melkein pilvinen (pilvisyys = 6)
local AlmostCloudy = ProbLimitEq2D(tmp, rain_radius, 6):GetValues()

-- Puolipilvinen ( 3<= pilvisyys <= 5 )
local HalfCloudy = ProbLimitGe2D(tmp, rain_radius, 3):GetValues() 
local HalfCloudy2 = ProbLimitGt2D(tmp, rain_radius, 5):GetValues()

for i=1, #HalfCloudy do
  HalfCloudy[i] = HalfCloudy[i] - HalfCloudy2[i]
end

-- Aurinkoinen (pilvisyys < 2)
local Sunny = ProbLimitLt2D(tmp, rain_radius, 2):GetValues()

-----------------
-- STRONG GUST --
-----------------

local gust_number = {}

-- Tuulenpuuskat >= 15 m/s

tmp:SetValues(gust)
local windgust1 = ProbLimitGe2D(tmp, wind_radius, 15):GetValues()

for i=1, #windgust1 do
  if(windgust1[i] > 0.20) then
    gust_number[i] = 1
  end
end

-- Tuulenpuuskat >= 20 m/s
local windgust2 = ProbLimitGe2D(tmp, wind_radius, 20):GetValues()

for i=1, #windgust2 do
  if(windgust2[i] > 0.20) then
    gust_number[i] = 2
  end
end

-----------------------------
-- NEARBY WEATHER CALCULATION

local NEIGHBOUR = Weather

for i=1, #NEIGHBOUR do

  if (thunder_percentage[i] > 0) then
    if (thunder_awareness[i] >= 2) then
      NEIGHBOUR[i] = 92000
    else
      NEIGHBOUR[i] = 91000
    end
  elseif (rainy_weather[i] == 1)  then
    NEIGHBOUR[i] = 80000 + rain_frequency[i]*1000

    if (prevalent_type[i] == 1 or prevalent_form[i] == 0 or prevalent_form[i] == 5) then
         NEIGHBOUR[i] = NEIGHBOUR[i] + prevalent_form[i]*100+10
    else
         NEIGHBOUR[i] = NEIGHBOUR[i] + prevalent_form[i]*100+20
    end
-- Yhditelmäsateita OSA 1. Laitetaan kaikki aluksi jatkuvan sateen teksteille

    if ( (prevalent_form[i] == 1 and snow[i] >= 0.20 and snow[i] >= sleet[i]) or (prevalent_form[i] == 3 and water[i] >= 0.20 and water[i] >= sleet[i]) ) then
      NEIGHBOUR[i] = 80130 + rain_frequency[i]*1000 -- vesi- tai lumisadetta
    elseif ( (prevalent_form[i] == 1 and sleet[i] >= 0.20 and sleet[i] > snow[i]) or (prevalent_form[i] == 2 and water[i] >= 0.20 and water[i] > snow[i]) ) then
      NEIGHBOUR[i] = 80140 + rain_frequency[i]*1000 -- vesi- tai räntäsadetta
    elseif ( (prevalent_form[i] == 2 and snow[i] >= 0.20 and snow[i] > water[i]) or (prevalent_form[i] == 3 and sleet[i] >= 0.20 and sleet[i] > water[i]) ) then
      NEIGHBOUR[i] = 80250 + rain_frequency[i]*1000 -- lumi- tai räntäsadetta
    end

-- Yhditelmäsateita OSA 2. ...korvataan edellisiä tarpeen mukaan kuuroteksteillä.

    if (prevalent_type[i] == 2) then
      if ( (prevalent_form[i] == 1 and snow[i] >= 0.20 and snow[i] >= sleet[i]) or (prevalent_form[i] == 3 and water[i] >= 0.20 and water[i] >= sleet[i]) ) then
        NEIGHBOUR[i] = 80160 + rain_frequency[i]*1000 -- vesi- tai lumikuuroja
      elseif ( (prevalent_form[i] == 1 and sleet[i] >= 0.20 and sleet[i] > snow[i]) or (prevalent_form[i] == 2 and water[i] >= 0.20 and water[i] > snow[i]) ) then
        NEIGHBOUR[i] = 80170 + rain_frequency[i]*1000 -- vesi- tai räntäkuuroja
      elseif ( (prevalent_form[i] == 2 and snow[i] >= 0.20 and snow[i] > water[i]) or (prevalent_form[i] == 3 and sleet[i] >= 0.20 and sleet[i] > water[i]) ) then
        NEIGHBOUR[i] = 80280 + rain_frequency[i]*1000 -- lumi- tai räntäkuuroja
      end
    end

-- Laitetaan voimakkaiden (sakeiden) sateiden tekstejä (vesisateella rr>=2.0 mm/h, räntä/lumisateella >=0.4 mm/h)

    if(strong_water[i] >= 0.30 and prevalent_form[i] == 1 and (snow[i] < 0.20 and sleet[i] < 0.20)) then
      NEIGHBOUR[i] = NEIGHBOUR[i] - 70000 -- Tehdään ykkösellä alkavia numeroita
      if(strong_snow[i] >= 0.65 and (prevalent_form[i] == 2 or prevalent_form[i] ==3) and water[i] < 0.20) then
        NEIGHBOUR[i] = NEIGHBOUR[i] - 70000 -- Tehdään ykkösellä alkavia numeroita
      end
    end
  else
-- SUMUTEKSTEJÄ
    if (fog_index[i] > 0 ) then
      if (fog_index[i] == 1) then
        NEIGHBOUR[i] = 61000 -- "paikoin sumua"
      end
      if (fog_index[i] == 2) then
        NEIGHBOUR[i] = 62000 -- "monin paikoin sumua"
      end
      if (fog_index[i] == 3) then
        NEIGHBOUR[i] = 63000 -- "sumua" 
      end
    else  -- PILVITEKSTEJÄ
      if (MedianCloudiness[i] == 3 or MedianCloudiness[i] >= 6) then -- Laitetaan pohjalle pilvisyys mediaanin avulla ja sovitetaan se tässä käytettävään numerointiin
        NEIGHBOUR[i] = 70000 + (MedianCloudiness[i]+1)*1000
      else                                       
        NEIGHBOUR[i] = 70000 + MedianCloudiness[i]*1000
      end
      if (Cloudy[i] + AlmostCloudy[i] >= 0.65) then -- "Pilvistä tai melkein pilvistä"
        NEIGHBOUR[i] = 78000
      end
      if (Cloudy[i] >= 0.90) then -- "Pilvistä"
        NEIGHBOUR[i] = 79000
      end
      if ((HalfCloudy[i] > 0.50 and HalfCloudy[i] <0.80) and Sunny[i] < 0.05) then -- "Puolipilvistä tai pilvistä"
        NEIGHBOUR[i] = 77000
      end 
      if ((Sunny[i] > 0.30 and Sunny[i] < 0.60) and Cloudy[i] < 0.20) then -- "Selkeää tai puolipilvistä"
        NEIGHBOUR[i] = 73000
      end
      if ((Sunny[i] >= 0.60 and Sunny[i] <= 0.80) and Cloudy[i] < 0.20) then -- "Melkein selkeää" tai "Enimmäkseen selkeää" tai "Pilvisyys on vähäistä"
        NEIGHBOUR[i] = 72000
      end 
      if (Sunny[i] > 0.80) then
        NEIGHBOUR[i] = 71000 -- "Selkeää"
      end
      if ((MaxCloudiness[i] - MinCloudiness[i] >= 0.05) and cloud_cover[i] >= 6 and (Cloudy[i] < 0.75)) then -- "Vaihtelevaa pilvisyyttä"
        NEIGHBOUR[i] = 70500
      end
      if (cloud_cover[i] <= 2 and (NEIGHBOUR[i] >= 75000 and NEIGHBOUR[i] <= 79000) ) then -- "Vaihtelevaa pilvisyyttä" , kun pisteessä pilvisyys vähäistä, mutta lähellä puolipilvisestä pilviseen
        NEIGHBOUR[i] = 70500
      end
      if (NEIGHBOUR[i] == 70000 ) then
        NEIGHBOUR[i] = 71000
      end
      if (NEIGHBOUR[i] == 74000 ) then
        NEIGHBOUR[i] = 75000
      end
    end
  end

-- PUUSKATEKSTIEN LISÄYS TARVITTAESSA
-- Puuskat >= 15 m/s ja  >= 20 m/s

  if (gust_number[i] == 1) then
    NEIGHBOUR[i] = NEIGHBOUR[i] + 1 -- >= 15 m/s
  end 

  if (gust_number[i] == 2) then
    NEIGHBOUR[i] = NEIGHBOUR[i] + 2 -- >= 20 m/s
  end
end

result:SetValues(NEIGHBOUR)
result:SetParam(param("NEARW-N"))
luatool:WriteToFile(result)

