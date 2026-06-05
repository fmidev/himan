-- Cb or TCu cloud base (ft) and cover (%)
-- Translated from https://wiki.fmi.fi/spaces/PROJEKTIT/pages/48558154/CbTCu_base_ft v1.2

function round(n)
  return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

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



-- Spatial averaging of LCL within ~10 km radius using circular 9x9 kernel
-- (MEPS ~2.5 km resolution → radius ~4 grid cells ≈ 10 km)
local kernel = {0,0,0,0,1,0,0,0,0,
                0,0,1,1,1,1,1,0,0,
                0,1,1,1,1,1,1,1,0,
                0,1,1,1,1,1,1,1,0,
                1,1,1,1,1,1,1,1,1,
                0,1,1,1,1,1,1,1,0,
                0,1,1,1,1,1,1,1,0,
                0,0,1,1,1,1,1,0,0,
                0,0,0,0,1,0,0,0,0}

local avgkernel = {}
for i = 1, #kernel do
  avgkernel[i] = kernel[i] / 49
end

local avg_filter = matrixf(9, 9, 1, missing)
avg_filter:SetValues(avgkernel)

local Nmat = matrixf(result:GetGrid():GetNi(), result:GetGrid():GetNj(), 1, 0)

Nmat:SetValues(LCL500)
LCL500 = Filter2D(Nmat, avg_filter, configuration:GetUseCuda()):GetValues()

Nmat:SetValues(LCLmu)
LCLmu = Filter2D(Nmat, avg_filter, configuration:GetUseCuda()):GetValues()


-- Build per-grid-point base heights for vertical N lookup.
local safe_base_heights = {}
for i = 1, #CBTCU_FL do
  if IsMissing(CBTCU_FL[i]) then
    -- Replace missing with dummy 1.0 m; results for those points are discarded below.
    safe_base_heights[i] = 1.0
  elseif IsMissing(LCLmu[i]) or LCLmu[i] < LCL500[i] then
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
local Missing  = missing

for i = 1, #CBTCU_FL do
  base_res[i] = Missing
  cov_res[i]  = Missing

  if not IsMissing(CBTCU_FL[i]) then
    local base
    if IsMissing(LCLmu[i]) or LCLmu[i] < LCL500[i] then
      base = LCL500[i]   -- surface-based convection
    else
      base = LCLmu[i]    -- elevated convection
    end

    local cover = Missing
    if not IsMissing(N_at_base[i]) then
      cover = N_at_base[i] * 100  -- convert 0–1 to %
    end
    if IsMissing(cover) or cover < 1 then
      cover = covdef
    end

    -- Tweak cover upward by Cb probability if available, silently skip otherwise
    if ProbCb and not IsMissing(ProbCb[i]) and cover < ProbCb[i] then
      cover = ProbCb[i]
    end

    base_res[i] = round(base / 0.3048 / 100) * 100  -- metres → feet, 100 ft resolution
    cov_res[i]  = round(cover)
  end
end

result:SetParam(param("CBTCU-FT"))
result:SetValues(base_res)
luatool:WriteToFile(result)

result:SetParam(param("CBTCU-PRCNT"))
result:SetValues(cov_res)
luatool:WriteToFile(result)
