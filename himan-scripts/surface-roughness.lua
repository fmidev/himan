-- Total surface roughness for Hirlam
-- partio 2017-02-20

logger:Info("Calculating total surface roughness")

local sr = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("SR-M"))
local srmom = luatool:Fetch(current_time, level(HPLevelType.kHeight, 0), param("SRMOM-M"))

if not sr or not srmom then
  logger:Error("Data not found")
  return
end

local i = 0
local res = {}

for i=1, #sr do
  local _sr = sr[i]
  local _srmom = srmom[i]

  res[i] = _sr + _srmom
end

result:SetValues(res)
result:SetParam(param("SR-M"))

luatool:WriteToFile(result)
