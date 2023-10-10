-- Create PrecipitationType and PrecipitationForm from Potential parameter version
-- Only for smartmet editor data
-- partio 2018-02-16

local ppf = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("POTPRECF-N"), current_forecast_type)
local ppt = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("POTPRECT-N"), current_forecast_type)
local rrr = luatool:FetchWithType(current_time, level(HPLevelType.kHeight, 0), param("RRR-KGM2"), current_forecast_type)

if not ppf or not ppt or not rrr then
  return
end

local i = 0
local pf = {}
local pt = {}

local MISS = missing

for i=1, #ppf do

  local _pf = MISS
  local _pt = MISS

  if rrr[i] > 0 then
    _pf = ppf[i]
    _pt = ppt[i]
  end 

  pf[i] = _pf
  pt[i] = _pt
end

-- using param PRECFORM2-N because that has mapping ready in radon
result:SetValues(pf)
result:SetParam(param("PRECFORM2-N"))
luatool:WriteToFile(result)

result:SetValues(pt)
result:SetParam(param("PRECTYPE-N"))
luatool:WriteToFile(result)
