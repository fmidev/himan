-- Calculating low and middle clouds parameter 
--
-- This script will replace qdversionfilter; the parameter is very
-- rudimentary and should not be used in any new products

logger:Info("Calculating low and middle clouds")

local MISS = missing

local N = luatool:Fetch(current_time, current_level, param("N-0TO1")) -- total cloudiness
local NL = luatool:Fetch(current_time, current_level, param("NL-0TO1")) -- low clouds
local NM = luatool:Fetch(current_time, current_level, param("NM-0TO1")) -- middle clouds
local NH = luatool:Fetch(current_time, current_level, param("NH-0TO1")) -- high clouds

if not N or not NL or not NM or not NH then
  return
end

local NLM = {} -- low and middle clouds

for i=1,#N do
  local n = N[i]
  local nl = NL[i]
  local nm = NM[i]
  local nh = NH[i]

  local nlm = MISS

  if nh == 0 or IsMissing(nh) then
    if IsValid(n) then
      nlm = n
    end
  else
    if IsValid(nl) then
      if IsValid(nm) then
        nlm = math.max(nl, nm)
      else
        nlm = nl
      end
    else
      if IsValid(nm) then
        nlm = nm
      end  
    end
  end  

  NLM[i] = nlm * 100 -- to percents
end

result:SetParam(param("NLM-PRCNT"))
result:SetValues(NLM)
luatool:WriteToFile(result)
