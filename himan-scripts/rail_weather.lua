function SumT(TMinus,TPlus,curTime,par)
  local psum = {}

  local stopStep = math.max(curTime:GetStep():Hours() + TMinus , 0)
  local startStep = math.min(curTime:GetStep():Hours() + TPlus , 240)

  -- set initial time step
  local mytime = forecast_time(curTime:GetOriginDateTime(),time_duration(HPTimeResolution.kHourResolution,startStep))

  while true do

    if mytime:GetStep():Hours() < stopStep or mytime:GetStep():Hours() < 0 then
      break
    end

    local p = luatool:FetchWithType(current_time, current_level, par, current_forecast_type)

    if p then
      if #psum == 0 then
        for i=1, #p do
         psum[i] = 0
        end
      end

      for i=1, #p do
        psum[i] = psum[i] + p[i]
      end
    end

    -- adjust time
    mytime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)

  end

  return psum
end

function MinT(TMinus,TPlus,curTime,par)
  local pmin = {}

  local stopStep = math.max(curTime:GetStep():Hours() + TMinus , 0)
  local startStep = math.min(curTime:GetStep():Hours() + TPlus , 240)

  -- set initial time step
  local mytime = forecast_time(curTime:GetOriginDateTime(),time_duration(HPTimeResolution.kHourResolution,startStep))

  while true do

    if mytime:GetStep():Hours() < stopStep or mytime:GetStep():Hours() < 0 then
      break
    end

    local p = luatool:FetchWithType(mytime, current_level, par, current_forecast_type)

    if p then
      if #pmin == 0 then
        for i=1, #p do
         pmin[i] = 1e38
        end
      end

      for i=1, #p do
        pmin[i] = math.min(pmin[i],p[i])
      end
    end

    -- adjust time
    mytime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)

  end

  return pmin
end

function MaxT(TMinus,TPlus,curTime,par,curLevel)
  local pmax = {}

  local stopStep = math.max(curTime:GetStep():Hours() + TMinus , 0)
  local startStep = math.min(curTime:GetStep():Hours() + TPlus , 240)

  -- set initial time step
  local mytime = forecast_time(curTime:GetOriginDateTime(),time_duration(HPTimeResolution.kHourResolution,startStep))

  while true do

    if mytime:GetStep():Hours() < stopStep or mytime:GetStep():Hours() < 0 then
      break
    end

    local p = luatool:FetchWithType(mytime, curLevel, par, current_forecast_type)

    if p then
      if #pmax == 0 then
        for i=1, #p do
         pmax[i] = -1e38
        end
      end 

      for i=1, #p do
        pmax[i] = math.max(pmax[i],p[i])
      end
    end

    -- adjust time
    mytime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)

  end

  return pmax
end

-- Implementation of the Regulation 5 conditions for railroad operation
-- Conditions in particular: 
-- 1. Temperature -5C or less
-- 2. Snowfallrate 5cm/12h; Calculate +/-6h
-- 3. Wind Speed 5 m/s or above

local min_t = MinT(-2,2,current_time,param("T-K"))
local snow_sum = SumT(-6,6,current_time,param("SNR-KGM2"))
local max_ws = MaxT(-2,2,current_time,param("FF-MS"),level(HPLevelType.kHeight,10))

if #min_t == 0 or #snow_sum == 0 or #max_ws == 0 then  
    return
end

local res = {}

for i=1, #min_t do
  res[i] = 0

  -- Calculate the index
  -- Determine forecast value Missing or 1; 
  if ( min_t[i] <= (-4.9 + kKelvin) and snow_sum[i] >= 4.9 and max_ws[i] >= 4.8) then
    res[i] = 1
  end
end

result:SetParam(param("RAIL-N"))
result:SetValues(res)
luatool:WriteToFile(result)
