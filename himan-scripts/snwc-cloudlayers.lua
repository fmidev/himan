--
-- SmartMet NWC parameters
--
-- Create cloud layer data for SNWC from 'own' total cloudiness and
-- cloud layers from Smartmet data.
-- 

local MISS = missing
local editor_prod = producer(181, "SMARTMET")

editor_prod:SetCentre(86)
editor_prod:SetProcess(181)

local editor_origintime = raw_time(radon:GetLatestTime(editor_prod, "", 0))
local editor_time = forecast_time(editor_origintime, current_time:GetValidDateTime())

local H0 = level(HPLevelType.kHeight, 0)
local CC = luatool:FetchWithType(current_time, H0, param("N-0TO1"), current_forecast_type)
local CH = luatool:FetchWithType(current_time, H0, param("NH-0TO1"), current_forecast_type)
local CM = luatool:FetchWithType(current_time, H0, param("NM-0TO1"), current_forecast_type)
local CL = luatool:FetchWithType(current_time, H0, param("NL-0TO1"), current_forecast_type)

if not CC or not CH or not CM or not CL then
  return
end

local _CH = {}
local _CM = {}
local _CL = {}

for i=1, #CC do
  local cc = CC[i]
  local ch = CH[i]
  local cm = CM[i]
  local cl = CL[i]

  _CH[i] = ch
  _CM[i] = cm
  _CL[i] = cl

  -- From 'DBChecker'

  -- Lisätään ala-, keski- tai yläpilviä niin että jokin kerroksista on sama kuin kokonaispilvisyys.
  -- Valinta tehdään sen mukaan mitä on jo alunperin enemmän.
  -- Tehdään ensin alapilville

  if (cl < cc and cl >= cm and cl >= ch) then
    _CL[i] = cc
  end

  -- ja keskipilville

  if (cm < cc and cl < cm and ch < cm) then
    _CM[i] = cc
  end

  -- lopuksi lisätään yläpilviä

  if (cm < cc and cl < ch and cm < ch) then
    _CH[i] = cc
  end

  -- Vähennetään ala- ja keskipilviä, jos ne ovat suurempia kuin kokonaispilvisyys.

  _CL[i] = math.min(cl, cc)
  _CM[i] = math.min(cm, cc)

end

result:SetParam(param("NH-0TO1"))
result:SetValues(_CH)
luatool:WriteToFile(result)
result:SetParam(param("NM-0TO1"))
result:SetValues(_CM)
luatool:WriteToFile(result)
result:SetParam(param("NL-0TO1"))
result:SetValues(_CL)
luatool:WriteToFile(result)
