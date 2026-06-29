-- Compute first, second and last freezing levels
-- Used in ILME 

local utils = require("utils")

local t_param = param("T-K")
local fzT = 273.15

local first  = hitool:VerticalHeight(t_param, 0, 15000, fzT, 1)
local second = hitool:VerticalHeight(t_param, 0, 15000, fzT, 2)
local last   = hitool:VerticalHeight(t_param, 0, 15000, fzT, 0)

for i = 1, #first do
  first[i]  = utils.round(first[i]  / 0.3048)
  second[i] = utils.round(second[i] / 0.3048)
  last[i]   = utils.round(last[i]   / 0.3048)
end

result:SetParam(param("H0C-1ST-FT"))
result:SetValues(first)
luatool:WriteToFile(result)

result:SetParam(param("H0C-2ND-FT"))
result:SetValues(second)
luatool:WriteToFile(result)

result:SetParam(param("H0C-HIGHEST-FT"))
result:SetValues(last)
luatool:WriteToFile(result)