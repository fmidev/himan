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

local currentProducer = configuration:GetTargetProducer()
local currentProducerName = currentProducer.GetName(currentProducer)

logger:Info(string.format("Calculating cloud symbol for producer: %s", currentProducerName))

local ground_level = level(HPLevelType.kGround, 0)
local height_level = level(HPLevelType.kHeight, 0)

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

local tcu = CheckedFetch("CBTCU-FL", height_level, current_time)

local rr = nil
if currentProducerName == "ECG" or currentProducerName == "ECGMTA" or
  currentProducerName == "HL2" or currentProducerName == "HL2MTA" or
  currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  rr = CheckedFetch("RRR-KGM2", height_level, current_time)
else
  error("Unknown producer for cloud symbol!")
end

-- total cloud cover
local cloudcov_par = nil
if currentProducerName == "ECG" or currentProducerName == "ECGMTA" or
  currentProducerName == "HL2" or currentProducerName == "HL2MTA" or
  currentProducerName == "MEPS" or currentProducerName == "MEPSMTA" then
  cloudcov_par = param("N-0TO1")
else
  error("unknown producer")
end

hitool:SetHeightUnit(HPParameterUnit.kM)
local stratus = hitool:VerticalMaximum(cloudcov_par, 0, 305)

if not stratus then
  logger:Error(string.format("hitool:VerticalMaximum() returned nil"))
  return
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

for i=1, #t do
  local _tk  = t[i] -- in kelvin
  local _cl = cl[i]
  local _cm = cm[i]
  local _ch = ch[i]
  local _rr = rr[i]
  local _t = _tk - kKelvin
  local _tcu = tcu[i]
  local _st = stratus[i]

  Cloud[i] = Missing

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

  -- Towering cumulus: low clouds (or middle clouds for elevated convection) with sufficiently large vertical extent
  if _tcu < 0 then
    Cloud[i] = 157
    CloudInfoIncr("157")
  end

  -- St
  --

  -- St fractus: sct clouds near the surface, and less than bkn middle clouds
  if _st > 0.2 and _st <= 0.5 and _cm <= 0.5 then
    Cloud[i] = 162
    CloudInfoIncr("162")
  end

  -- St or St fractus
  --

  -- St: bkn/ovc clouds near the surface and no precipitation
  if _st > 0.5 then
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
  if _tcu > 0 then
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
