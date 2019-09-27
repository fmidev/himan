--
-- Calculate precipitation rate for GFS from APCP and ACPCP, and
-- FMI type precipitation form from CRAIN, CFRZR, CICEP and CSNOW.
--
-- NB! This script works ONLY FOR GFS!
--
-- partio 2016-06-15
--

local MISS = missing
local groundlevel = level(HPLevelType.kGround, 0)
local step = current_time:GetStep():Hours()

function GetSourceInfo(rrparam, adjustment)
  local validtime = current_time:GetValidDateTime()
  local newvalidtime = raw_time(validtime:String("%Y-%m-%d %H:%M:%S"))
  newvalidtime:Adjust(HPTimeResolution.kHourResolution, adjustment)

  newtime = forecast_time(current_time:GetOriginDateTime(), newvalidtime)
  return luatool:FetchInfo(newtime, groundlevel, rrparam)

end

function Rate(rrparam, targetparam)
  -- GFS has 1-6 or 12 hour cumulation depending on the forecast valid time

  -- Try to get current time data

  local rrinfo = GetSourceInfo(rrparam, 0)

  if not rrinfo then
    -- try next time
    local adjustment = 1
    while adjustment < 13 do
      rrinfo = GetSourceInfo(rrparam, adjustment)
      if rrinfo then
        break
      end
      adjustment = adjustment+1
    end

    if not rrinfo then
      return
    end
  end

  local curstep = rrinfo:GetTime():GetStep():Hours()

  local rrdata = rrinfo:GetValues()
  local prevrrdata = {}
  local prevstep = 0

  if current_time:GetStep():Hours() <= 6 then
    for i = 1,#rrdata do
      prevrrdata[i] = 0
    end
  else
    local prevrrinfo = GetSourceInfo(rrparam, -1)

    if not prevrrinfo then
      adjustment = -2

      while adjustment > -14 do
        prevrrinfo = GetSourceInfo(rrparam, adjustment)
        if prevrrinfo then
          break
        end
        adjustment = adjustment - 1
      end

      if not prevrrinfo then
        return
      end
    end
    prevrrdata = prevrrinfo:GetValues()

    -- accumulation is 'zeroed' every 6 hours: 6, 12, 18, 24, ...
    if prevrrinfo:GetTime():GetStep():Hours() % 6 == 0 then
      for i = 1,#rrdata do
        prevrrdata[i] = 0
      end
    end

    prevstep = prevrrinfo:GetTime():GetStep():Hours()
  end


  local rate = {}

  for i = 1, #rrdata do
    local cur = rrdata[i]
    local prev = prevrrdata[i]

    if cur ~= MISS and prev ~= MISS then
      rate[i] = (cur - prev) / (curstep-prevstep)

      if rate[i] < 0 then
        rate[i] = 0
      end
    end
  end

  -- Set correct time range indicator (ie. aggregation type)

  agg = aggregation(HPAggregationType.kAccumulation, time_duration("01:00"))
  agg:SetType(HPAggregationType.kAccumulation)

  targetparam:SetAggregation(agg)

  result:SetParam(targetparam)
  result:SetLevel(level(HPLevelType.kHeight, 0))
  result:SetValues(rate)

  luatool:WriteToFile(result)

end

function Form()

  local snowdata = luatool:Fetch(current_time, groundlevel, param("CSNOW-0OR1"))
  local raindata = luatool:Fetch(current_time, groundlevel, param("CRAIN-0OR1"))
  local frzrdata = luatool:Fetch(current_time, groundlevel, param("CFRZR-0OR1"))
  local icepdata = luatool:Fetch(current_time, groundlevel, param("CICEP-0OR1"))

  local preformdata = {}

  if not snowdata or not raindata or not frzrdata or not icepdata then
    return
  end

  for i = 1, #snowdata do
    local preform = MISS

    -- GFS precipitation form data are all "categorical", meaning that data is
    -- either 0 or 1 (nothing else).

    local snow = snowdata[i]
    local rain = raindata[i]
    local frzr = frzrdata[i]
    local icep = icepdata[i]

    -- FMI precipitation form for grib2 is:
    -- 0 = drizzle, 1 = water, 2 = sleet, 3 = snow, 4 = freezing drizzle, 5 = freezing rain
    -- 6 = snow or ice grain, 7 = snow pellet, 8 = ice pellet
    if snow == 1 then
      preform = 3
    elseif rain == 1 then
      preform = 1
    elseif frzr == 1 then
      preform = 5
    elseif icep == 1 then
      preform = 8
    end

    preformdata[i] = preform
  end

  p = param("PRECFORM3-N")

  result:SetParam(p)
  result:SetValues(preformdata)

  luatool:WriteToFile(result)

end

-- Rate(param("RR-KGM2"), param("RRR-KGM2"))
-- Rate(param("RRC-KGM2"), param("RRRC-KGM2"))
Form()
