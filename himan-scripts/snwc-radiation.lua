--
-- SmartMet NWC parameters
--
-- Create radiation parameters to SNWC using data from ECMWF
-- and fixing the radiation if total cloudiness has changed.
-- 

local MISS = missing
local cc_prod = producer(131, "ECG")
local rad_prod = producer(240, "ECGMTA")
local ftype = forecast_type(HPForecastType.kDeterministic)

local latest_origintime = raw_time(radon:GetLatestTime(rad_prod, "", 0))

logger:Info("Latest ECMWF analysis: " .. latest_origintime:String("%Y-%m-%d %H:%M:%S"))

local latest_time = forecast_time(latest_origintime, current_time:GetValidDateTime())

local lvl0 = level(HPLevelType.kHeight, 0)
local n = param("N-0TO1")
local glob = param("RADGLO-WM2")
local lw = param("RADLW-WM2")

local SW = luatool:FetchWithProducer(latest_time, lvl0, glob, ftype, rad_prod, "")
local LW = luatool:FetchWithProducer(latest_time, lvl0, lw, ftype, rad_prod, "")
local CC = luatool:FetchWithType(current_time, lvl0, n, current_forecast_type)
local CC_ORIG = luatool:FetchWithProducer(latest_time, lvl0, n, ftype, cc_prod, "")

if not CC then
  return
end

if not SW or not LW or not CC_ORIG then
  -- HIMAN-338
  -- try earlier forecast
  -- also accept ECMWF 06/18 runs
  latest_origintime:Adjust(HPTimeResolution.kHourResolution, -6)
  latest_time = forecast_time(latest_origintime, current_time:GetValidDateTime())

  logger:Info("Latest forecast data missing, trying older forecast with analysis time: " .. latest_origintime:String("%Y-%m-%d %H:%M:%S"))

  SW = luatool:FetchWithProducer(latest_time, lvl0, glob, ftype, rad_prod, "")
  LW = luatool:FetchWithProducer(latest_time, lvl0, lw, ftype, rad_prod, "")
  CC_ORIG = luatool:FetchWithProducer(latest_time, lvl0, n, ftype, cc_prod, "")

  if not SW or not LW or not CC_ORIG then
    return
  end
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
