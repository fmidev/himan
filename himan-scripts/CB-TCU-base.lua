-- Cb or TCu cloud base (ft) and cover (%)
-- Translated from https://wiki.fmi.fi/spaces/PROJEKTIT/pages/48558154/CbTCu_base_ft v1.2

local utils = require("utils")

local MU = level(HPLevelType.kMaximumThetaE, 0)
local HL = level(HPLevelType.kHeightLayer, 500, 0)
local HG = level(HPLevelType.kHeight, 0)

local CBTCU_FL = luatool:Fetch(current_time, HG, param("CBTCU-FL"), current_forecast_type)
local LCL500 = luatool:Fetch(current_time, HL, param("LCL-M"), current_forecast_type)
local LCLmu = luatool:Fetch(current_time, MU, param("LCL-M"), current_forecast_type)
local ProbCb = luatool:Fetch(current_time, HG, param("PROB-CBTCU-1"), current_forecast_type)

-- skip optional ProbCb
if not CBTCU_FL or not LCL500 or not LCLmu then
  logger:Error("Required data not found")
  return
end

local Nmat = matrixf(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)
local avg_mask = utils.create_mask(result:GetGrid():GetDi()/1000, 10, "circle", true)

Nmat:SetValues(LCL500)
LCL500 = Filter2D(Nmat, avg_mask, configuration:GetUseCuda()):GetValues()

Nmat:SetValues(LCLmu)
LCLmu = Filter2D(Nmat, avg_mask, configuration:GetUseCuda()):GetValues()

-- Build per-grid-point base heights for vertical N lookup.
local safe_base_heights = {}
for i = 1, #CBTCU_FL do
  if IsMissing(CBTCU_FL[i]) then
    safe_base_heights[i] = missing
  elseif LCLmu[i] < LCL500[i] then
    -- surface based convection (500m mixed layer) Cb/TCu base (m)
    safe_base_heights[i] = LCL500[i]
  else
    -- elevated convection Cb/TCu base
    safe_base_heights[i] = LCLmu[i]
  end
end

-- Get cloud fraction N (0–1) at each grid point's base height
local N_at_base = hitool:VerticalValueGrid(param("N-0TO1"), safe_base_heights)

local covdef = 20  -- default Cb/TCu cover [%] when N at LCL < 1%

local base_res = {}
local cov_res  = {}

for i = 1, #CBTCU_FL do
  local cover = N_at_base[i] * 100  -- convert 0–1 to %
  if cover < 1 then -- nan < 1 == false
    cover = covdef
  end

  -- Tweak cover upward by Cb probability if available, silently skip otherwise
  if ProbCb and cover < ProbCb[i] then
    cover = ProbCb[i]
  end

  base_res[i] = utils.round(safe_base_heights[i] / 0.3048 / 100) * 100  -- metres → feet, 100 ft resolution
  cov_res[i]  = utils.round(cover)
end

result:SetParam(param("CBTCU-FT"))
result:SetValues(base_res)
luatool:WriteToFile(result)

result:SetParam(param("CBTCU-PRCNT"))
result:SetValues(cov_res)
luatool:WriteToFile(result)
