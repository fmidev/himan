--[[

cloud symbol ('Pilvisymboli')

SmartTool version: https://wiki.fmi.fi/display/PROJEKTIT/CloudSymbol_ec

Comments pertaining to the cloud symbol derivation are from the original smarttool macro.

Lua conversion and extension for Hirlam and MEPS: vanhatam, 2016

Cloud types:
+ 156 Cu: few/sct low clouds, and less than bkn middle clouds
+ 157 Towering cumulus: sct or more low clouds with sufficiently large vertical extent (sfc or 500m based)
+ 158 Cb with tops lacking clear-cut outlines: light convective precipitation
+ 160 Sc: bkn/ovc low clouds
+ 161 St: bkn/ovc clouds near the surface and no precipitation
+ 162 St fractus: bkn/ovc cloud near the surface and precipitation
+ 164 Anvil cb: mod or greater convective precipitation
+ 166 Thin As: few/sct middle clouds and less than bkn high clouds
+ 167 Ns/Thick As: bkn/ovc middle clouds and precipitation
+ 168 Thin Ac: bkn middle clouds
+ 170 Thick Ac: ovc middle clouds
+ 176 Thin Ci (filaments): few/sct high clouds
+ 177 Dense Ci: bkn/ovc high clouds
+ 253 Thick Ns (hatched symbol): bkn/ovc low and middle clouds and (more than very light) precipitation

]]

local Missing = missing

local function CheckedFetch(parm_name, lvl, time)
  local p = luatool:FetchWithType(time, lvl, param(parm_name), current_forecast_type)
  if not p then
    logger:Error(string.format("Failed to find parameter '%s'", parm_name))
    error("luatool:Fetch failed")
  else
    return p
  end
end

hitool:SetHeightUnit(HPParameterUnit.kHPa)

--
local currentProducer = configuration:GetSourceProducer(1)
local currentProducerName = currentProducer.GetName(currentProducer)

logger:Info(string.format("Calculating cloud symbol for producer: %s", currentProducerName))

local ground_level = level(HPLevelType.kGround, 0)
local height_level = level(HPLevelType.kHeight, 0)
local mixed500_level = level(HPLevelType.kHeightLayer, 500, 0)
local mu_level = level(HPLevelType.kMaximumThetaE, 0)

-- required vertical thickness [C] to consider a CB (tweakable)
local CBlimit = 9
-- required vertical thickness [C] to consider a TCU (tweakable)
local TCUlimit = 6
-- required cloud top T [C] to consider a CB (tweakable)
local CBtopLim = -10

-- Should probably assert the following assumptions:
-- CB = top T < -10C
-- EL = CB / TCU top
-- LCL500 (or LFCmu) = CB / TCU base

local cl = nil
local cm = nil
local ch = nil
local t = nil

if currentProducerName == "ECG" or currentProducerName == "ECGMTA" then
  cl  = CheckedFetch("NL-0TO1", ground_level, current_time)
  cm  = CheckedFetch("NM-0TO1", ground_level, current_time)
  ch  = CheckedFetch("NH-0TO1", ground_level, current_time)
  t   = CheckedFetch("T-K", ground_level, current_time)
elseif currentProducerName == "HL2" or currentProducerName == "HL2MTA" then
  cl  = CheckedFetch("NL-0TO1", height_level, current_time)
  cm  = CheckedFetch("NM-0TO1", height_level, current_time)
  ch  = CheckedFetch("NH-0TO1", height_level, current_time)
  t   = CheckedFetch("T-K", height_level, current_time)
elseif currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  cl = CheckedFetch("NL-0TO1", height_level, current_time)
  cm = CheckedFetch("NM-0TO1", height_level, current_time)
  ch = CheckedFetch("NH-0TO1", height_level, current_time)
  t  = CheckedFetch("T-K", height_level, current_time)
else
  error("Unkown producer for cloud symbol!")
end

local rh_level = function()
  if currentProducerName == "ECG" or currentProducerName == "ECGMTA" then
    return level(HPLevelType.kHybrid, 137)
  elseif currentProducerName == "HL2" or currentProducerName == "HL2MTA" then
    return level(HPLevelType.kHybrid, 65)
  elseif currentProducerName == "AROMTA" or currentProducerName == "MEPSMTA" then
    return level(HPLevelType.kHybrid, 1)
  else
    error("Unknown producer for cloud symbol!")
  end end

local rh  = CheckedFetch("RH-PRCNT", rh_level(), current_time)

local rr = nil
if currentProducerName == "ECG" or currentProducerName == "ECGMTA" or
  currentProducerName == "HL2" or currentProducerName == "HL2MTA" or
  currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  rr = CheckedFetch("RRR-KGM2", height_level, current_time)
else
  error("Unknown producer for cloud symbol!")
end

--
-- LCL, CIN, EL
--

-- local cin_sfc = nil
-- local el_sfc = nil
-- local lfc_sfc = nil
-- local lcl_sfc = nil
local cin_500mix = nil
local el_500mix = nil
local lfc_500mix = nil
local lcl_500mix = nil
local cin_mu = nil
local el_mu = nil
local lfc_mu = nil
local lcl_mu = nil

if currentProducerName == "ECG" or currentProducerName == "ECGMTA"
  or currentProducerName == "HL2" or currentProducerName == "HL2MTA"
  or currentProducerName == "MEPS" or currentProducerName == "MEPSMTA"
then
  -- el_sfc     = CheckedFetch("EL-HPA", height_level, current_time)
  -- cin_sfc    = CheckedFetch("CIN-JKG", height_level, current_time)
  -- lfc_sfc    = CheckedFetch("LFC-HPA", height_level, current_time)
  -- lcl_sfc    = CheckedFetch("LCL-HPA", height_level, current_time)
  el_500mix  = CheckedFetch("EL-HPA", mixed500_level, current_time)
  cin_500mix = CheckedFetch("CIN-JKG", mixed500_level, current_time)
  lfc_500mix = CheckedFetch("LFC-HPA", mixed500_level, current_time)
  lcl_500mix = CheckedFetch("LCL-HPA", mixed500_level, current_time)
  el_mu      = CheckedFetch("EL-HPA", mu_level, current_time)
  cin_mu     = CheckedFetch("CIN-JKG", mu_level, current_time)
  lfc_mu     = CheckedFetch("LFC-HPA", mu_level, current_time)
  lcl_mu     = CheckedFetch("LCL-HPA", mu_level, current_time)
else
  error("Unknown producer for cloud symbol!")
end

--if cin_sfc == nil or el_sfc == nil or lfc_sfc == nil then
--    logger:Error(string.format("Surface based sounding indices not found"))
--end

if not cin_500mix or not el_500mix or not lfc_500mix then
  logger:Error(string.format("500m mixed sounding indices not found"))
end

if not cin_mu or not el_mu or not lfc_mu or not lcl_mu then
  logger:Error(string.format("MU sounding indices not found"))
end

-- total cloud cover
local cloudcov_par = nil
if currentProducerName == "ECG" or currentProducerName == "ECGMTA" or
  currentProducerName == "HL2" or currentProducerName == "HL2MTA" or
  currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  cloudcov_par = param("N-0TO1")
elseif currentProducername == "AROMTA" then
  error("cloud cover not defined for AROMTA")
else
  error("unknown producer")
end

hitool:SetHeightUnit(HPParameterUnit.kM)
local stratus = hitool:VerticalMaximum(cloudcov_par, 0, 305)
--
local vert_level = hitool:VerticalValue(cloudcov_par, 61)

if not stratus then
  logger:Warning(string.format("hitool:VerticalMaximum() returned nil"))
end

if not vert_level then
  logger:Warning(string.format("hitool:VerticalValue() returned nil"))
end

--
-- Statistics for debugging / understanding
--
local cloud_info = (function()
  cinfo = {}

  cinfo["156"] = { 0, 156, "Cu: few/sct low clouds, and less than bkn middle clouds" }
  cinfo["157"] = { 0, 157, "Towering cumulus: sct or more low clouds with sufficiently large vertical extent (sfc or 500m based)" }
  cinfo["158"] = { 0, 158, "Cb with tops lacking clear-cut outlines: light convective precipitation" }
  cinfo["160"] = { 0, 160, "Sc: bkn/ovc low clouds" }
  cinfo["161"] = { 0, 161, "St: bkn/ovc clouds near the surface and no precipitation" }
  cinfo["162"] = { 0, 162, "St fractus: bkn/ovc cloud near the surface and precipitation" }
  cinfo["164"] = { 0, 164, "Anvil cb: mod or greater convective precipitation" }
  cinfo["166"] = { 0, 166, "Thin As: few/sct middle clouds and less than bkn high clouds" }
  cinfo["167"] = { 0, 167, "Ns/Thick As: bkn/ovc middle clouds and precipitation" }
  cinfo["168"] = { 0, 168, "Thin Ac: bkn middle clouds" }
  cinfo["170"] = { 0, 170, "Thick Ac: ovc middle clouds" }
  cinfo["176"] = { 0, 176, "Thin Ci (filaments): few/sct high clouds" }
  cinfo["177"] = { 0, 177, "Dense Ci: bkn/ovc high clouds" }
  cinfo["253"] = { 0, 253, "Thick Ns (hatched symbol): bkn/ovc low and middle clouds and (more than very light) precipitation" }

  return cinfo
end)()

local function PrintCloudStats(cinfo)
  for k in pairs(cinfo) do
    logger:Info(string.format("--- (%d): %d | %s", cinfo[k][2], cinfo[k][1], cinfo[k][3]))
  end
end

local function CloudInfoIncr(cloud_n)
  cloud_info[cloud_n][1] = cloud_info[cloud_n][1] + 1
end

-- Output 'symbol' is stored here
local Cloud   = {}

hitool:SetHeightUnit(HPParameterUnit.kHPa)

-- 500m mix based convective cloud top and base temperature
local Ttop = hitool:VerticalValueGrid(param("T-K"), el_500mix)
local TBase = hitool:VerticalValueGrid(param("T-K"), lcl_500mix)

-- MU based convective cloud top and base temperature
local TtopMU = hitool:VerticalValueGrid(param("T-K"), el_mu)
local TbaseMU = hitool:VerticalValueGrid(param("T-K"), lfc_mu)

for i=1, #rh do
  local _rh = rh[i]
  local _tk  = t[i] -- in kelvin
  local _cl = cl[i]
  local _cm = cm[i]
  local _ch = ch[i]
  local _rr = rr[i]
  local _t = _tk - kKelvin

  Cloud[i] = Missing

  local _st = nil
  local _vert_level = nil

  if not vert_level then
    if _rh > 0.9 and _t >= -8.0 then
      _st = _cl
    end
    if _rh > 0.8 and _t < -8.0 then
      _st = _cl
    end
  else
    _vert_level = vert_level[i]
    _st = stratus[i]
  end


  -- Ci
  --

  -- Thin Ci
  if _ch > 0.0 and _ch <= 0.5 then
    Cloud[i] = 176
    CloudInfoIncr("176")
  end

  -- Dense Ci: bkn/ovc high clouds
  if _ch > 0.5 then
    Cloud[i] = 177
    CloudInfoIncr("177")
  end

  -- Thin As: few/sct middle clouds and less than bkn high clouds
  if _cm > 0.0 and _cm <= 0.5 and _ch < 0.5 then
    Cloud[i] = 166
    CloudInfoIncr("166")
  end

  -- Thin Ac: bkn middle clouds
  if _cm > 0.5 and _cm <= 0.8 then
    Cloud[i] = 168
    CloudInfoIncr("168")
  end

  -- Thick Ac: ovc middle clouds
  if _cm > 0.8 then
    Cloud[i] = 170
    CloudInfoIncr("170")
  end

  -- Ns/Thick As: bkn/ovc middle clouds and precipitation
  if _cm > 0.5 and _rr > 0.0 then
    Cloud[i] = 167
    CloudInfoIncr("167")
  end

  -- Cu/Sc
  --

  -- Cu: few/sct low clouds, and less than bkn middle clouds
  if _cl > 0.0 and _cl <= 0.5 and _cm <= 0.5 then
    Cloud[i] = 156
    CloudInfoIncr("156")
  end

  -- Sc: bkn/ovc low clouds
  if _cl > 0.5 then
    Cloud[i] = 160
    CloudInfoIncr("160")
  end

  -- TCu
  --

  local _el500  = el_500mix[i]
  local _cin500 = cin_500mix[i]
  local _lfc500 = lfc_500mix[i]
  local _lcl500 = lcl_500mix[i]

  local _elmu = el_mu[i]
  local _cinmu = cin_mu[i]
  local _lfcmu = lfc_mu[i]
  local _lclmu = lcl_mu[i]

  local _tbase = TBase[i] - kKelvin
  local _ttop = Ttop[i] - kKelvin
  local _tbasemu = TbaseMU[i] - kKelvin
  local _ttopmu = TtopMU[i] - kKelvin

  -- Towering cumulus: low clouds (or middle clouds for elevated convection) with sufficiently large vertical extent
  if (_cl > 0 and (_tbase - _ttop) > TCUlimit and _cin500 > -1.0) or
    ((_cl > 0.0 or _cm > 0.0) and (_tbasemu - _ttopmu > TCUlimit) and _cinmu > -1.0) and
    _lfcmu < _lcl500 and _lfcmu > 650.0 then
    Cloud[i] = 157
    CloudInfoIncr("157")
  end

  -- St
  --

  -- St fractus: sct clouds near the surface, and less than bkn middle clouds
  if stratus[i] > 0.2 and stratus[i] <= 0.5 and _cm <= 0.5 then
    Cloud[i] = 162
    CloudInfoIncr("162")
  end

  -- St or St fractus
  --

  -- St: bkn/ovc clouds near the surface and no precipitation
  if stratus[i] > 0.5 then
    if _rr == 0.0 then
      Cloud[i] = 161
      CloudInfoIncr("161")
      -- St fractus: bkn/ovc cloud near the surface and precipitation
    else
      Cloud[i] = 162
      CloudInfoIncr("162")
    end
  end

  -- Thick Ns (hatched symbol): bkn/ovc low and middle clouds and (more than very light) precipitation
  if _cl > 0.5 and _cm > 0.5 and _rr > 0.2 then
    Cloud[i] = 253
    CloudInfoIncr("253")
  end

  -- Cb: (convective) precipitation and clouds with sufficiently large vertical extent
  if _rr > 0.0 and ((_ttop < CBtopLim and (_tbase - _ttop) > CBlimit) or
    (_ttopmu < CBtopLim and (_tbasemu - _ttopmu) > CBlimit and _lfcmu < _lcl500 and _lfcmu > 650.0)) then
    -- Anvil cb: ~mod or greater (convective) precipitation
    if _rr > 1.0 then
      Cloud[i] = 164
      CloudInfoIncr("164")
      -- CB with tops lacking clear-cut outlines: ~light (convective) precipitation
    else
      Cloud[i] = 158
      CloudInfoIncr("158")
    end
  end
end

PrintCloudStats(cloud_info)

result:SetParam(param("CLDTYPE-N"))
result:SetValues(Cloud)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)
