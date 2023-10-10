--[[

Frosty
Frozen soil depth
Supported producers: ECSEASONAL, EC

]]

local Missing = missing
local currentTime = current_time
local currentProducer = configuration:GetSourceProducer(0)
local currentProducerName = currentProducer.GetName(currentProducer)

msg = string.format("Calculating frozen soil bottom and top depth for producer: %s", currentProducerName)
logger:Info(msg)

local function CheckedFetch(paramName, lvl, time)
  local p = luatool:Fetch(time, lvl, param(paramName), current_forecast_type)
  if not p then
    msg = string.format("Failed to find parameter '%s'", paramName)
    error(msg)
  else
    return p
  end
end

local STL1
local STL2
local STL3
local STL4

if currentProducerName == "ECSEASONAL" then
  STL1 = CheckedFetch("TSOIL-K", level(HPLevelType.kDepthLayer, 0, 7), currentTime)
  STL2 = CheckedFetch("TSOIL-K", level(HPLevelType.kDepthLayer, 7, 28), currentTime)
  STL3 = CheckedFetch("TSOIL-K", level(HPLevelType.kDepthLayer, 28, 100), currentTime)
  STL4 = CheckedFetch("TSOIL-K", level(HPLevelType.kDepthLayer, 100, -1), currentTime)
elseif currentProducerName == "ECG" then
  STL1 = CheckedFetch("TG-K", level(HPLevelType.kGroundDepth, 0, 7), currentTime)
  STL2 = CheckedFetch("TSOIL-K", level(HPLevelType.kGroundDepth, 7, 28), currentTime)
  STL3 = CheckedFetch("TSOIL-K", level(HPLevelType.kGroundDepth, 28, 100), currentTime)
  STL4 = CheckedFetch("TSOIL-K", level(HPLevelType.kGroundDepth, 100, -1), currentTime)
else
  error("Unsupported producer")
end

local T2   = CheckedFetch("T-K",level(HPLevelType.kGround, 0), currentTime)

local Frsb = {} -- frozen soil bottom depth in cm
local Frst = {} -- frozen soil top depth in cm

for i=1, #STL1 do
  local _s1 = STL1[i]
  local _s2 = STL2[i]
  local _s3 = STL3[i]
  local _s4 = STL4[i]
  local _t2 = T2[i]

  Frsb[i] = missing
  Frst[i] = missing

  -- bottom
  if _s4 == _s4 and _s3 == _s3 and
    _s2 == _s2 and _s1 == _s1 and
    _t2 == _t2
  then
    if _s4 < 273.15 then
      Frsb[i] = 177.5 + 113.5 * (273.15 - _s4) / (_s4 - _s3)
      if Frsb[i] > 255.0 then
        Frsb[i] = 255
      end
    elseif _s3 < 273.15 then
      Frsb[i] = 64.0  + 113.5 * (273.15 - _s4) / (_s4 - _s3)
    elseif _s2 < 273.15 then
      Frsb[i] = 17.5  + 46.5  * (273.15 - _s3) / (_s3 - _s2)
    elseif _s1 < 273.15 then
      Frsb[i] = 3.5  + 14.0   * (273.15 - _s2) / (_s2 - _s1)
    elseif _t2 < 273.15 then
      Frsb[i] = 0.0  + 3.5    * (273.15 - _s1) / (_s1 - _t2)
    else
      Frsb[i] = 0.0
    end
  end

  -- top
  if _s4 == _s4 and _s3 == _s3 and
    _s2 == _s2 and _s1 == _s1 and
    _t2 == _t2
  then
    if Frsb[i] > 0.0 and _t2 > 273.15 and _s1 < 273.15 then
      Frst[i] = 0.0  + 3.5   * (273.15 - _t2) / (_s1 - _t2)
    elseif Frsb[i] > 3.5 and _s1 > 273.15 and _s2 < 273.15 then
      Frst[i] = 3.5  + 14.0  * (273.15 - _s1) / (_s2 - _s1)
    elseif Frsb[i] > 17.5 and _s2 > 273.15 and _s3 < 273.15 then
      Frst[i] = 17.5 + 46.5  * (273.15 - _s2) / (_s3 - _s2)
    elseif Frsb[i] > 64.0 and _s3 > 273.15 and _s4 < 273.15 then
      Frst[i] = 64.0 + 113.5 * (273.15 - _s3) / (_s4 - _s3)
    elseif Frsb[i] > 177.5 and _s4 > 273.15 then
      Frst[i] = 177.5 + 113.5 * (273.15 - _s4) / (_s4 - _s3)
      if Frst[i] > 255.0 then
        Frst[i] = 255
      end
    else
      Frst[i] = 0.0
    end
  end

  -- if all frost is thawed, set soil frost values both to 0
  if Frst[i] > Frsb[i] then
    Frst[i] = 0.0
    Frsb[i] = 0.0
  end

  if Frst[i] < 0.0 then
    Frst[i] = 0.0
  end

  if Frsb[i] < 0.0 then
    Frsb[i] = 0.0
  end
end

local p_frsb = param("FRSB-CM")
local p_frst = param("FRST-CM")

if configuration:Exists("FRST") then
  result:SetParam(p_frst)
  result:SetValues(Frst)
elseif configuration:Exists("FRSB") then
  result:SetParam(p_frsb)
  result:SetValues(Frsb)
else
  error("select FRST or FRSB")
end

logger:Info("Writing result data to file")
luatool:WriteToFile(result)
