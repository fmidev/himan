-- ICAO TAF/METAR weather code (WX) mapping to numbers from model data (to be used in ADF/AMFIS tool)
--
-- v2.0: may return a code for two WX phenomena (precipitation + fog/mist, or snow + drifting/blowing snow).
-- Combined code = base code + 100 (FG) / 200 (FZFG) / 300 (BR) / 400 (DRSN) / 500 (BLSN).
-- Ported from icao-smartool v2.0.

local MISS = missing

-- producer ids
local ECGMTA = 240
local MEPSMTA = 260

local producerId = configuration:GetTargetProducer():GetId()

-- commonly used levels
local level2m = level(HPLevelType.kHeight, 2)
local level10m = level(HPLevelType.kHeight, 10)
local levelGround = level(HPLevelType.kHeight, 0)

-- freezing point [K]
local T0 = 273.15

-- precipitation intensity mm/h
local PreInt = param("RRR-KGM2")

-- snowfall intensity mm/h
-- use solid precipitation rate for MEPS
local Snow
if (producerId == MEPSMTA) then
  Snow = param("RRRS-KGM2")
else
  Snow = param("SNR-KGM2")
end

-- snow accumulation
-- use solid precipitation rate for MEPS
local Snacc
if (producerId == MEPSMTA) then
  Snacc = param("RRS-12-MM")
else
  Snacc = param("SN-12-MM")
end

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

-- relative humidity (%)
local RH = param("RH-0TO1")

-- wind speed
local ws = param("FF-MS")

-- wind gust
local wg = param("FFG-MS", aggregation(HPAggregationType.kMaximum, time_duration(HPTimeResolution.kHourResolution, 1)), processing_type())

-- length of the averaging window of the 2m temperature (dry-snow check for DRSN/BLSN) [h]
local TavgHours = 5

-- analysis time interval of the producer [h], needed when the averaging window
-- reaches over the analysis time and an older forecast has to be used
local runInterval = 12
if (producerId == MEPSMTA) then
  runInterval = 3
end

-- fetch input params
local PreIntdata = luatool:Fetch(current_time, current_level, PreInt, current_forecast_type)
local Snowdata = luatool:Fetch(current_time, current_level, Snow, current_forecast_type)
local PreFormdata = luatool:Fetch(current_time, current_level, PreForm, current_forecast_type)
local visibdata = luatool:Fetch(current_time, current_level, visib, current_forecast_type)
local POTdata = luatool:Fetch(current_time, current_level, POT, current_forecast_type)
local cbdata = luatool:Fetch(current_time, current_level, cb, current_forecast_type)
-- use most-unstable CAPE for EC, surface level for MEPS
local CAPEmdata
if (producerId == ECGMTA) then
  CAPEmdata = luatool:Fetch(current_time, level(HPLevelType.kMaximumThetaE, 0), CAPEm, current_forecast_type)
else
  CAPEmdata = luatool:Fetch(current_time, current_level, CAPEm, current_forecast_type)
end
local BSdata = luatool:Fetch(current_time, level(HPLevelType.kHeightLayer,6000,0), BS, current_forecast_type)
local Tdata = luatool:Fetch(current_time, level2m, t, current_forecast_type)
local RHdata = luatool:Fetch(current_time, level2m, RH, current_forecast_type)

-- Mean 2m temperature of the past hours (dry-snow check for DRSN/BLSN).
-- Radon has no time aggregated T-K -- and asking for one silently returns the
-- instantaneous field -- so the mean is calculated here from the instantaneous
-- 2m temperatures of the current and the preceding hours. Hours that are before
-- the analysis time are read from an older forecast, hours that are not found
-- at all are left out of the mean.
local function MeanTemperature(hours)
  local sum = nil
  local count = 0

  for h=0, hours-1 do
    local ftime = forecast_time(current_time)
    ftime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -h)

    -- valid time is before the analysis time: use an older analysis time
    local step = ftime:GetStep():Hours()

    if (step < 0) then
      local adjustment = math.ceil(-step / runInterval) * runInterval
      ftime:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, -adjustment)
    end

    local data = luatool:Fetch(ftime, level2m, t, current_forecast_type)

    if data then
      if (sum == nil) then
        sum = {}
        for i=1, #data do
          sum[i] = 0
        end
      end

      for i=1, #data do
        sum[i] = sum[i] + data[i]
      end

      count = count + 1
    else
      logger:Warning(string.format("2m temperature not found for -%dh, leaving it out of the %dh mean", h, hours))
    end
  end

  if (count == 0) then
    logger:Error(string.format("No 2m temperature found for the %dh mean", hours))
    error("luatool:Fetch failed")
  end

  -- with a single hour the "mean" is the instantaneous temperature, which makes
  -- the dry-snow check weaker but still produces data for the time step
  if (count == 1) then
    logger:Warning(string.format("Only 1 of %d hours found, using the instantaneous 2m temperature", hours))
  end

  local mean = {}

  for i=1, #sum do
    mean[i] = sum[i] / count
  end

  return mean
end

local Tavgdata = MeanTemperature(TavgHours)

local TGdata = luatool:Fetch(current_time, levelGround, t, current_forecast_type)
-- for EC fetch param skin temperature
if (producerId == ECGMTA) then
    TGdata = luatool:Fetch(current_time, level(HPLevelType.kGround,0), t0m, current_forecast_type)
end

local wsdata = luatool:Fetch(current_time, level10m, ws, current_forecast_type)
local wgdata = luatool:Fetch(current_time, level10m, wg, current_forecast_type)

-- fetch snow accumulation
-- Use older analysis time if not enough time steps are available for 12 accumulation period
local Snaccdata
if (current_time:GetStep():Hours() < 12) then
  local new_time = forecast_time(current_time)

  -- take the origin time back by the accumulation period, rounded to whole
  -- analysis times of the producer (12h for EC, 3h for MEPS)
  local adjustment = math.floor(current_time:GetStep():Hours()/runInterval) * runInterval - 12

  new_time:GetOriginDateTime():Adjust(HPTimeResolution.kHourResolution, adjustment)
  Snaccdata = luatool:Fetch(new_time,current_level,Snacc,current_forecast_type)
-- After step 12h use current forecast
else
  Snaccdata = luatool:Fetch(current_time,current_level,Snacc,current_forecast_type)
end

-- calculate area_max fields with ~30km box
local filter
if (producerId == MEPSMTA) then
  filter = matrixf(12, 12, 1, 1)
elseif (producerId == ECGMTA) then
  filter = matrixf(3, 3, 1, 1)
end

local Nmat = matrixf(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
Nmat:SetValues(POTdata)
local areaMaxPOT = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()
Nmat:SetValues(cbdata)
local areaMaxCB = Max2D(Nmat,filter,configuration:GetUseCuda()):GetValues()

-- set constants
-- rr limit for MEPS (which has large areas of near zero hourly precipitation)
local rrLim = 0.04

-- Relative humidity threshold [%] for (freezing) misty/foggy conditions in precipitation
local rhMoist = 0.95

-- Threshold for showery precipitation (may need tweaking) [J/kg]
local shCAPE = 10

-- Minimum required thunderstorm index (POT) value for TS
local TSlim = 60

-- Minimum required CbTCuTop for thunderstorms [FL]
local CbTSlim = 150

-- CAPE limit for hail [J/kg]
-- Tweak this and/or add more hail criteria
local HailCAPE = 500

-- Bulk Shear 0-6km limit for hail [m/s]
-- Tweak this and/or add more hail criteria
local HailBS = 9

-- Limit for moderate/heavy drizzle [mm/h]
local ModDzLim = 0.1
local HvyDzLim = 0.2

-- Limits for light/moderate/heavy rain [mm/h]
local ModRaLim = 1
local HvyRaLim = 4

-- Limit for moderate/heavy "wet" sleet (RASN) [mm/h]
local ModRaSleetLim = 1
local HvyRaSleetLim = 2.5

-- Limit for moderate/heavy "snowy" sleet (SNRA) [mm/h]
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
  if ((visibdata[i] < 1000) and (Tdata[i] < T0)) then
    wx[i] = 12
  end

  -- Drizzle, possibly with mist/fog
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

    -- -DZ BR (DZ/+DZ BR not included due to already misty visibility caused by drizzle)
    if (wx[i] == 50 and visibdata[i] >= 1000 and visibdata[i] < 5000 and RHdata[i] > rhMoist) then
      wx[i] = wx[i] + 300
    end

    -- -DZ/DZ/+DZ FG
    if (wx[i] < 100 and visibdata[i] < 1000 and RHdata[i] > rhMoist) then
      wx[i] = wx[i] + 100
    end
  end

  -- Rain, possibly with thunderstorm and/or fog/mist
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

    -- -RA/-SHRA/-TSRA/-TSGR BR (moderate/heavy rain not considered due to highly variable visibility in them)
    if ((wx[i] == 20 or wx[i] == 23 or wx[i] == 60 or wx[i] == 81) and visibdata[i] >= 1000 and visibdata[i] < 5000 and RHdata[i] > rhMoist) then
      wx[i] = wx[i] + 300
    end

    -- -RA/RA/+RA/-SHRA/SHRA/+SHRA/-TSRA/TSRA/+TSRA/-TSGR/TSGR/+TSGR FG
    if (wx[i] < 100 and visibdata[i] < 1000 and RHdata[i] > rhMoist) then
      wx[i] = wx[i] + 100
    end
  end

  -- TS (thunderstorm nearby within 8km of the airport, but no precipitation), possibly with mist/fog = 32
  if (PreIntdata[i] == 0 and areaMaxPOT[i] > TSlim and areaMaxCB[i] > CbTSlim) then
    wx[i] = 32

    -- TS BR
    if (visibdata[i] >= 1000 and visibdata[i] < 5000) then
      wx[i] = wx[i] + 300
    end

    -- TS FG/FZFG
    if (wx[i] < 100 and visibdata[i] < 1000) then
      if (Tdata[i] >= T0) then
        wx[i] = wx[i] + 100
      else
        wx[i] = wx[i] + 200
      end
    end
  end

  -- Sleet (possibly with thunderstorms and/or fog)
  if ((PreFormdata[i] == 2) and (PreIntdata[i] > rrLim)) then
    -- continuous
    if (PreType == 1) then
      -- wet sleet
      if (Snowdata[i]/PreIntdata[i] <= SnSleet) then
        -- -RASN
        wx[i] = 66
        -- RASN
        if (PreIntdata[i] > ModRaSleetLim) then
          wx[i] = 67
        end
        -- +RASN
        if (PreIntdata[i] > HvyRaSleetLim) then
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

      -- -RASN/-SNRA FG (RASN/+RASN/SNRA/+SNRA FG not allowed)
      if ((wx[i] == 66 or wx[i] == 69) and visibdata[i] < 1000 and RHdata[i] > rhMoist) then
        wx[i] = wx[i] + 100
      end
    end
    if (PreType == 2) then
      -- wet sleet shower
      if (Snowdata[i] / PreIntdata[i] <= SnSleet) then
        -- -SHRASN
        wx[i] = 84
        -- SHRASN
        if (PreIntdata[i] > ModRaSleetLim) then
          wx[i] = 85
        end
        -- +SHRASN
        if (PreIntdata[i] > HvyRaSleetLim) then
          wx[i] = 86
        end
        -- Thunderstorm and wet sleet
        if (POTdata[i] > TSlim and cbdata[i] > CbTSlim) then
          -- -TSRASN
          wx[i] = 33
          -- TSRASN
          if (PreIntdata[i] > ModRaSleetLim) then
            wx[i] = 34
          end
          -- +TSRASN
          if (PreIntdata[i] > HvyRaSleetLim) then
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
        if (POTdata[i] > TSlim and cbdata[i] > CbTSlim) then
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

      -- -SHRASN/-TSRASN/-SHSNRA/-TSSNRA FG (+ variants not allowed)
      if ((wx[i] == 84 or wx[i] == 33 or wx[i] == 87 or wx[i] == 36) and visibdata[i] < 1000 and RHdata[i] > rhMoist) then
        wx[i] = wx[i] + 100
      end
    end
  end

  
  -- set when DRSN/BLSN applies, so Snow section can combine it with the precipitation code
  local DRBL = missing

  -- Simplified guesses for DRSN/BLSN
  -- Tsfc to discard open water areas (no ice), Tavg to discard wet-snow cases
  -- DRSN
  if (Snaccdata ~= nil) then
    if (Snaccdata[i] > DRSNlim and wsdata[i] >= 6 and Tdata[i] < T0 and TGdata[i] < T0 and Tavgdata[i] < T0) then
      DRBL = 15
      wx[i] = DRBL
    end

    -- BLSN
    if (Snaccdata[i] > DRSNlim and wsdata[i] >= BLSNwind and wgdata[i] >=BLSNgust and Tdata[i] < T0 and TGdata[i] < T0 and Tavgdata[i] < T0) then
      DRBL = 16
      wx[i] = DRBL
    end
  end

  --Snow (possibly with thunderstorm and/or drifting/blowing snow or fog)
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
      if (PreIntdata[i] > HvySnLim) then
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

    -- add DRSN/BLSN when applicable
    if (DRBL == 15) then
      wx[i] = wx[i] + 400
    end
    if (DRBL == 16) then
      wx[i] = wx[i] + 500
    end

    -- -SN/-SHSN/-TSSN FG/FZFG, only if DRSN/BLSN wasn't already added (+ variants not allowed)
    if ((wx[i] == 72 or wx[i] == 90 or wx[i] == 26) and visibdata[i] < 1000 and RHdata[i] > rhMoist) then
      if (Tdata[i] >= T0) then
        wx[i] = wx[i] + 100
      else
        wx[i] = wx[i] + 200
      end
    end
  end

  -- Freezing Drizzle, possibly with mist or freezing fog
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

    -- -FZDZ BR (FZDZ/+FZDZ BR not included due to already misty visibility caused by drizzle)
    if (wx[i] == 53 and visibdata[i] >= 1000 and visibdata[i] < 5000) then
      wx[i] = wx[i] + 300
    end

    -- -FZDZ/FZDZ/+FZDZ FZFG
    if (wx[i] < 100 and visibdata[i] < 1000) then
      wx[i] = wx[i] + 200
    end
  end

  -- Freezing Rain, possibly with mist or freezing fog
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

    -- -FZRA BR (FZRA/+FZRA not included due to highly variable visibility in them)
    if (wx[i] == 63 and visibdata[i] >= 1000 and visibdata[i] < 5000) then
      wx[i] = wx[i] + 300
    end

    -- -FZRA/FZRA/+FZRA FZFG
    if (wx[i] < 100 and visibdata[i] < 1000) then
      wx[i] = wx[i] + 200
    end
  end

  -- Snow grains
  if (PreFormdata[i] == 7) then
    -- -SG
    wx[i] = 75
    -- SG
    if (PreIntdata[i] > ModSGlim) then
      wx[i] = 76
    end
    -- +SG
    if (PreIntdata[i] > HvySGlim) then
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
