-- ICAO TAF/METAR weather code (WX) mapping to numbers from model data (to be used in ADF/AMFIS tool)

local MISS = missing

-- precipitation intensity mm/h
local PreInt = param("RRR-KGM2")

-- snowfall intensity mm/h
local Snow = param("SNR-KGM2")

-- snow accumulation
local Snacc = param("SN-12-MM")

-- precipitation form FMI
local PreForm = param("PRECFORM2-N")

-- visibility FMI (m)
local visib = param("VV2-M")

-- POT FMI 
local POT = param("POT-PRCNT")

-- CbTCu top FMI (FL)
local cb = param("CBTCU-FL")

-- model CAPE (J/kg)
local CAPEm = param("CAPE-JKG")

-- bulk shear 6km (m/s)
local BS = param("WSH-MS")

-- temperture
local t = param("T-K")

-- skin temperature
local t0m = param("SKT-K")

-- wind speed
local ws = param("FF-MS")

-- wind gust
local wg = param("FFG-MS")

-- fetch input params
local PreIntdata = luatool:Fetch(current_time, current_level, PreInt, current_forecast_type)
local Snowdata = luatool:Fetch(current_time, current_level, Snow, current_forecast_type)
local PreFormdata = luatool:Fetch(current_time, current_level, PreForm, current_forecast_type)
local visibdata = luatool:Fetch(current_time, current_level, visib, current_forecast_type)
local POTdata = luatool:Fetch(current_time, current_level, POT, current_forecast_type)
local cbdata = luatool:Fetch(current_time, current_level, cb, current_forecast_type)
local CAPEmdata = luatool:Fetch(current_time, current_level, CAPEm, current_forecast_type)
local BSdata = luatool:Fetch(current_time, level(HPLevelType.kHeightLayer,6000,0), BS, current_forecast_type)
local Tdata = luatool:Fetch(current_time, level(HPLevelType.kHeight,2), t, current_forecast_type)

local TGdata = luatool:Fetch(current_time, level(HPLevelType.kHeight,0), t, current_forecast_type)
-- for EC fetch param skin temperature
if (configuration:GetTargetProducer():GetId() == 240) then
    TGdata = luatool:Fetch(current_time, level(HPLevelType.kGround,0), t0m, current_forecast_type)
end

local wsdata = luatool:Fetch(current_time, level(HPLevelType.kHeight,10), ws, current_forecast_type)
local wgdata = luatool:Fetch(current_time, level(HPLevelType.kHeight,10), wg, current_forecast_type)

-- fetch snow accumulation
-- Use older analysis time if not enough time steps are available for 12 accumulation period
local Snaccdata
if (current_time:GetStep():Hours() < 12) then
  local new_time = forecast_time(current_time)

  -- set origin time adjustment depending on producer, default -12h
  -- for meps we adjust in steps of 3h
  local adjustment = -12
  if (configuration:GetTargetProducer():GetId() == 260) then
    adjustment = math.floor(current_time:GetStep():Hours()/3) * 3 - 12
  end

  new_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjustment)
  Snaccdata = luatool:Fetch(new_time,current_level,Snacc,current_forecast_type)
-- After step 12h use current forecast
else
  Snaccdata = luatool:Fetch(current_time,current_level,Snacc,current_forecast_type)
end

-- calculate area_max fields with ~30km box
local filter
if (configuration:GetTargetProducer():GetId() == 260) then
  filter = matrixf(12, 12, 1, 1)
elseif (configuration:GetTargetProducer():GetId() == 240) then
  filter = matrixf(3, 3, 1, 1)
end

local Nmat = matrixf(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
Nmat:SetValues(POTdata)
local areaMaxPOT = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()
Nmat:SetValues(cbdata)
local areaMaxCB = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

-- set constants
local rrLim = 0
local shCAPE = 10
local TSlim = 60
local CbTSlim = 150
local HailCAPE = 500
local HailBS = 9

-- Limit for moderate/heavy drizzle [mm/h]
local ModDzLim = 0.1
local HvyDzLim = 0.2

-- Limits for light/moderate/heavy rain [mm/h]
local ModRaLim = 1
local HvyRaLim = 4

-- Limit for moderate/heavy sleet [mm/h]
local ModSleetLim = 0.7
local HvySleetLim = 1.5

-- Limits for light/moderate/heavy snow [mm/h]
local ModSnLim = 0.7
local HvySnLim = 1.5

-- Limits for moderate/heavy snow grains [mm/h]
local ModSGlim = 0.2
local HvySGlim = 0.4

-- Limit for moderate/heavy fzdz [mm/h]
local ModFzdzLim = 0.1
local HvyFzdzLim = 0.2

-- Limit for moderate/heavy fzra [mm/h]
local ModFzraLim = 0.7
local HvyFzraLim = 1.5

-- Limit for snowy sleet (ratio Snow/PreInt) [0...1]
local SnSleet = 0.5

-- Min required snowfall accumulation for Drifting Snow
local DRSNlim = 0.5

-- Min required mean wind and gust (m/s) for Blowing Snow
local BLSNwind = 10
local BLSNgust = 15

--- start the algorithm
local wx = {}
for i=1, #PreIntdata do
  wx[i] = missing
  local PreType = missing
  if (CAPEmdata[i] > shCAPE) then
    PreType = 2
  else
    PreType = 1
  end

  -- Mist BR
  if ((visibdata[i] >= 1000) and (visibdata[i] < 5000)) then
    wx[i] = 10
  end

  -- Fog FG
  if (visibdata[i] < 1000) then
    wx[i] = 11
  end

  -- Freezing fog FZFG
  if ((visibdata[i] < 1000) and (Tdata[i] < 273.15)) then
    wx[i] = 12
  end

  -- Drizzle
  if (PreFormdata[i] == 0) then
    -- -DZ
    wx[i] = 50
    -- DZ
    if (PreIntdata[i] > ModDzLim) then
      wx[i] = 51
    end
    -- +DZ
    if (PreIntdata[i] > HvyDzLim) then
      wx[i] = 52
    end
  end

  -- Rain
  if ((PreFormdata[i] == 1) and (PreIntdata[i] > rrLim)) then
    -- continuous rain
    if (PreType == 1) then
      -- -RA
      wx[i] = 60
      -- RA
      if (PreIntdata[i] > ModRaLim) then
        wx[i] = 61
      end
      -- +RA
      if (PreIntdata[i] > HvyRaLim) then
        wx[i] = 62
      end

      -- Thunderstorm check also for contiuous rain
      if ((POTdata[i] > TSlim) and (cbdata[i] > CbTSlim)) then
        -- -TSRA
        wx[i] = 20
        -- -TSGR
        if ((CAPEmdata[i] > HailCAPE) and (BSdata[i] > HailBS)) then
          wx[i] = 23
        end
        -- TSRA/TSGR
        if (PreIntdata[i] > ModRaLim) then
          -- TSRA
          wx[i] = 21
          -- TSRG
          if ((CAPEmdata[i] > HailCAPE) and (BSdata[i] > HailBS)) then
            wx[i] = 24
          end
        end
        -- +TSRA/+TSRG
        if (PreIntdata[i] > HvyRaLim) then
          -- +TSRA
          wx[i] = 22
          -- +TSRG
          if ((CAPEmdata[i] > HailCAPE) and (BSdata[i] > HailBS)) then
            wx[i] = 25
          end
        end
      end
    end

    if (PreType ==2) then
      -- -SHRA
      wx[i] = 81
      -- SHRA
      if (PreIntdata[i] > ModRaLim) then
        wx[i] = 82
      end
      -- +SHRA
      if (PreIntdata[i] > HvyRaLim) then
        wx[i] = 83
      end
      -- Thunderstorm check also for contiuous rain
      if ((POTdata[i] > TSlim) and (cbdata[i] > CbTSlim)) then
        -- -TSRA
        wx[i] = 20
        -- -TSGR
        if ((CAPEmdata[i] > HailCAPE) and (BSdata[i] > HailBS)) then
          wx[i] = 23
        end
        -- TSRA/TSGR
        if (PreIntdata[i] > ModRaLim) then
          -- TSRA
          wx[i] = 21
          -- TSRG
          if ((CAPEmdata[i] > HailCAPE) and (BSdata[i] > HailBS)) then
            wx[i] = 24
          end
        end
        -- +TSRA/+TSRG
        if (PreIntdata[i] > HvyRaLim) then
          -- +TSRA
          wx[i] = 22
          -- +TSRG
          if ((CAPEmdata[i] > HailCAPE) and (BSdata[i] > HailBS)) then
            wx[i] = 25
          end
        end
      end
    end
  end

  -- TS (thunderstorm nearby within 8km of the airport, but no precipitation) = 32
  if (PreIntdata[i] == 0 and areaMaxPOT[i] > TSlim and areaMaxCB[i] > CbTSlim) then
    wx[i] = 32
  end

  -- Sleet (possibly with thunderstorms)
  if ((PreFormdata[i] == 2) and (PreIntdata[i] > rrLim)) then
    -- continuous
    if (PreType == 1) then
      -- wet sleet
      if (Snowdata[i]/PreIntdata[i] <= SnSleet) then
        -- -RASN
        wx[i] = 66
        -- RASN
        if (PreIntdata[i] > ModSleetLim) then
          wx[i] = 67
        end
        -- +RASN
        if (PreIntdata[i] > HvySleetLim) then
          wx[i] = 68
        end

      -- snowy sleet
      else
        -- -SNRA
        wx[i] = 69
        -- SNRA
        if (PreIntdata[i] > ModSleetLim) then
          wx[i] = 70
        end
        -- +SNRA
        if (PreIntdata[i] > HvySleetLim) then
          wx[i] = 71
        end
      end
    end
    if (PreType == 2) then
      -- wet sleet shower
      if (Snowdata[i] / PreIntdata[i] <= SnSleet) then
        -- -SHRASN
        wx[i] = 84
        -- SHRASN
        if (PreIntdata[i] > ModSleetLim) then
          wx[i] = 85
        end
        -- +SHRASN
        if (PreIntdata[i] > HvySleetLim) then
          wx[i] = 86
        end
        -- Thunderstorm and wet sleet
        if (POTdata[i] > TSlim and cbdata > CbTSlim) then
          -- -TSRASN
          wx[i] = 33
          -- TSRASN
          if (PreIntdata[i] > ModSleetLim) then
            wx[i] = 34
          end
          -- +TSRASN
          if (PreIntdata[i] > HvySleetLim) then
            wx[i] = 35
          end
        end
      -- Snowy sleet shower
      else
        -- -SHSNRA
        wx[i] = 87
        -- SHSNRA
        if (PreIntdata[i] > ModSleetLim) then
          wx[i] = 88
        end
        -- +SHSNRA
        if (PreIntdata[i] > HvySleetLim) then
          wx[i] = 89
        end
        -- Thunderstorm and snowy sleet
        if (POTdata[i] > TSlim and cbdata[i] > CbSlim) then
          -- -TSSNRA
          wx[i] = 36
          -- TSSNRA
          if(PreIntdata[i] > ModSleetLim) then
            wx[i] = 37
          end
          -- +TSSNRA
          if(PreIntdata[i] > HvySleetLim) then
            wx[i] = 38
          end
        end
      end
    end
  end

  -- Simplified guesses for DRSN/BLSN
  -- Tsfc to discard open water areas (no ice)
  -- DRSN
  if (Snaccdata ~= nil) then
    if (Snaccdata[i] > DRSNlim and wsdata[i] >= 6 and Tdata[i] < 273.15 and TGdata[i] < 273.15) then
      wx[i] = 15
    end

    -- BLSN
    if (Snaccdata[i] > DRSNlim and wsdata[i] >= BLSNwind and wgdata[i] >=BLSNgust and Tdata[i] < 273.15 and TGdata[i] < 273.15) then
      wx[i] = 16
    end
  end

  --Snow (possibly with thunderstorms)
  if ((PreFormdata[i] == 3) and (PreIntdata[i] > rrLim)) then
    --continuous
    if (PreType == 1) then
      -- -SN
      wx[i] = 72
      -- SN
      if (PreIntdata[i] > ModSnLim) then
        wx[i] = 73
      end
      -- +SN
      if (PreIntdata[i] > ModSnLim) then
        wx[i] = 74
      end
    end
    -- Shower
    if (PreType == 2) then
      -- -SHSN
      wx[i] = 90
      -- SHSN
      if (PreIntdata[i] > ModSnLim) then
        wx[i] = 91
      end
      -- +SHSN
      if (PreIntdata[i] > HvySnLim) then
        wx[i] = 92
      end
    end
    -- Thunderstorm and snowfall
    if (POTdata[i] > TSlim and cbdata[i] > CbTSlim) then
      -- -TSSN
      wx[i] = 26
      -- TSSN
      if (PreIntdata[i] > ModSnLim) then
        wx[i] = 27
      end
      -- +TSSN
      if (PreIntdata[i] > HvySnLim) then
        wx[i] = 28
      end
    end
  end

  -- Freezing Drizzle
  if (PreFormdata[i] == 4) then
    -- -FZDZ
    wx[i] = 53
    -- FZDZ
    if (PreIntdata[i] > ModFzdzLim) then
      wx[i] = 54
    end
    -- +FZDZ
    if (PreIntdata[i] > HvyFzdzLim) then
      wx[i] = 55
    end
  end

  -- Freezing Rain
  if (PreFormdata[i] == 5) then
    -- -FZRA
    wx[i] = 63
    -- FZRA
    if (PreIntdata[i] > ModFzraLim) then
      wx[i] = 64
    end
    -- +FZRA
    if (PreIntdata[i] > HvyFzraLim) then
      wx[i] = 65
    end
  end

  -- Snow grains
  if (PreFormdata[i] == 7) then
    -- -SG
    wx[i] = 75
    -- SG
    if (PreIntdata[i] > ModSGLim) then
      wx[i] = 76
    end
    -- +SG
    if (PreIntdata[i] > HvySGLim) then
      wx[i] = 77
    end
  end

  -- Ice pellets
  if (PreFormdata[i] == 8) then
    -- -PL
    wx[i] = 78
    -- PL
    if (PreIntdata[i] > ModFzraLim) then
      wx[i] = 79
    end
    -- +PL
    if (PreIntdata[i] > HvyFzraLim) then
      wx[i] = 80
    end
  end
end

-- write output
p = param("ICAOWX-N")
result:SetValues(wx)
result:SetParam(p)
luatool:WriteToFile(result)
