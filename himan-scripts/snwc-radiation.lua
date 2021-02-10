--
-- SmartMet NWC parameters
--
-- Create radiation parameters to SNWC using data from "edited data"
-- and fixing the raditation if total cloudiness has changed.
-- 

local MISS = missing
local editor_prod = producer(181, "SMARTMET")

editor_prod:SetCentre(86)
editor_prod:SetProcess(181)

local editor_origintime = raw_time(radon:GetLatestTime(editor_prod, "", 0))
local editor_time = forecast_time(editor_origintime, current_time:GetValidDateTime())

local SW = luatool:FetchWithProducer(editor_time, level(HPLevelType.kHeight, 0), param("RADGLO-WM2"), current_forecast_type, editor_prod, "")
local LW = luatool:FetchWithProducer(editor_time, level(HPLevelType.kHeight, 0), param("RADLW-WM2"), current_forecast_type, editor_prod, "")
local CC = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("N-0TO1"), current_forecast_type)
local CC_ORIG = luatool:FetchWithProducer(editor_time, level(HPLevelType.kHeight, 0), param("N-PRCNT"), current_forecast_type, editor_prod, "")

if not SW or not LW or not CC or not CC_ORIG then
  return
end

local _SW = {}
local _LW = {}

for i=1, #SW do
  local cc = CC[i]
  local cc_orig = CC_ORIG[i] * 0.01
  _SW[i] = MISS
  _LW[i] = MISS

  if cc == cc and SW[i] == SW[i] and LW[i] == LW[i] and CC_ORIG[i] == CC_ORIG[i] then
    _SW[i] = (1 - 0.67 * math.pow(cc, 3.32)) / (1 - 0.67 * math.pow(cc_orig, 3.32)) * SW[i]
    _LW[i] = (1 + 0.22 * math.pow(cc, 2.75)) / (1 + 0.22 * math.pow(cc_orig, 2.75)) * LW[i]

    if (_SW[i] > 0 and ElevationAngle_(result:GetLatLon(i), current_time:GetValidDateTime()) <= 0) then
      _SW[i] = 0
    end

    _SW[i] = math.max(0, _SW[i])
  end
end

result:SetParam(param("RADGLO-WM2"))
result:SetValues(_SW)
luatool:WriteToFile(result)
result:SetParam(param("RADLW-WM2"))
result:SetValues(_LW)
luatool:WriteToFile(result)
