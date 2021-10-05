--
-- SmartMet NWC parameters
--
-- Create radiation parameters to SNWC using data from ECMWF
-- and fixing the radiation if total cloudiness has changed.
-- 

local MISS = missing
local rad_prod = producer(240, "ECGMTA")
local ftype = forecast_type(HPForecastType.kDeterministic)

local latest_origintime = ""

if configuration:Exists("ecmwf_origintime") then
  logger:Debug("Using hard-coded ecmwf origintime") -- for testing purposes
  latest_origintime = raw_time(configuration:GetValue("ecmwf_origintime"))
else
  latest_origintime = raw_time(radon:GetLatestTime(rad_prod, "", 0))
end

logger:Info("Latest ECMWF analysis: " .. latest_origintime:String("%Y-%m-%d %H:%M:%S"))

local latest_time = forecast_time(latest_origintime, current_time:GetValidDateTime())

local prodid = configuration:GetTargetProducer():GetId()
local lvl0 = level(HPLevelType.kHeight, 0)
local glob = param("RADGLOC-WM2") -- "clear sky"
local lw = param("RADLWC-WM2") -- "clear sky"
local n = param("N-0TO1")
local cc_scale = 1.0

if prodid == 280 then
  n = param("N-PRCNT")
  cc_scale = 0.01
end

local SW = luatool:FetchWithProducer(latest_time, lvl0, glob, ftype, rad_prod, "")
local LW = luatool:FetchWithProducer(latest_time, lvl0, lw, ftype, rad_prod, "")
local CC = luatool:FetchWithType(current_time, lvl0, n, current_forecast_type)

if not CC then
  return
end

if not SW or not LW then
  -- HIMAN-338
  -- try earlier forecast
  -- also accept ECMWF 06/18 runs
  for i=1, 2 do
    latest_origintime:Adjust(HPTimeResolution.kHourResolution, -6)
    latest_time = forecast_time(latest_origintime, current_time:GetValidDateTime())

    logger:Info("Latest forecast data missing, trying older forecast with analysis time: " .. latest_origintime:String("%Y-%m-%d %H:%M:%S"))

    SW = luatool:FetchWithProducer(latest_time, lvl0, glob, ftype, rad_prod, "")
    LW = luatool:FetchWithProducer(latest_time, lvl0, lw, ftype, rad_prod, "")

    if SW and LW then
      break
    end
  end
end

if not SW or not LW then
  return
end

local out_SW = {}
local out_LW = {}

for i=1, #SW do
  local cc = CC[i] * cc_scale
  local sw = SW[i]
  local lw = LW[i]

  out_SW[i] = MISS
  out_LW[i] = MISS

  if cc == cc and sw == sw and lw == lw then
    out_SW[i] = (1 - 0.67 * math.pow(cc, 3.32)) * sw
    out_LW[i] = (1 + 0.22 * math.pow(cc, 2.75)) * lw

    if (out_SW[i] > 0 and ElevationAngle_(result:GetLatLon(i), current_time:GetValidDateTime()) <= 0) then
      out_SW[i] = 0
    end

    out_SW[i] = math.max(0, out_SW[i])
  end
end

result:SetParam(param("RADGLO-WM2"))
result:SetValues(out_SW)
luatool:WriteToFile(result)
result:SetParam(param("RADLW-WM2"))
result:SetValues(out_LW)
luatool:WriteToFile(result)
