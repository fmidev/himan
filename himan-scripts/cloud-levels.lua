local MISS = missing

-- Parameters
local ceil = param("CEIL-2-M")
local N = param("N-0TO1")

-- Constants
local maxH = 15000
local few = 0.05
local sct = 0.35
local bkn = 0.55
local dz = 300

local ceiling = luatool:Fetch(current_time, current_level, ceil, current_forecast_type)

maxHdata= {}
zerodata= {}
fewdata= {}
sctdata= {}
bkndata= {}
dzdata= {}

for i = 1, #ceiling do
  maxHdata[i] = maxH
  zerodata[i] = 0

  fewdata[i] = few
  sctdata[i] = sct
  bkndata[i] = bkn
  dzdata[i] = dz
end

--// 1st cloud height: few-sct
local base1 = hitool:VerticalHeightGreaterThan(N, 0, maxH, few,1)
local top1 = hitool:VerticalHeightLessThanGrid(N, base1, maxHdata, fewdata,1)
local top1_sct = hitool:VerticalHeightLessThanGrid(N, base1, maxHdata, sctdata,1)
local Nbase1 = hitool:VerticalMaximumGrid(N, zerodata, base1)

local tmp1 = hitool:VerticalHeightGreaterThanGrid(N, base1, top1, sctdata,1)
local tmp2 = hitool:VerticalHeightGreaterThanGrid(N, base1, top1, bkndata,1)

local Ntmp = Nbase1
local Ntmp1 = hitool:VerticalMaximumGrid(N, base1, tmp1)

for i = 1, #base1 do
  if Ntmp[i] < bkn then 
    if base1[i] < tmp1[i] and tmp1[i] < tmp2[i] then
      top1[i] = base1[i]
    end
    if base1[i] < tmp1[i] and IsMissing(tmp2[i]) then
      top1[i] = base1[i]
    end
    if Ntmp1[i] < sct and IsMissing(tmp2[i]) then
      top1[i] = top1_sct[i]
    end
  else
    if base1[i] < ceiling[i] then
      base1[i] = ceiling[1]
    end
    top1[i] = top1_sct[i]
  end    
end

--// cloud layer cover/amount (%)
local cov1 = hitool:VerticalMaximumGrid(N, base1, top1)

--// *** 2nd layer, at least SCT
local base2 = hitool:VerticalHeightGreaterThanGrid(N, top1, maxHdata, sctdata,1)
local top2 = hitool:VerticalHeightLessThanGrid(N, base2, maxHdata, sctdata,1)

tmp2 = hitool:VerticalHeightGreaterThanGrid(N, base2, top2, bkndata,1)

Ntmp = hitool:VerticalMaximumGrid(N, top1, base2)

for i = 1, #base2 do
  if Ntmp[i] >= bkn and (IsMissing(ceiling[i]) or (base1[i] < tmp2[i] and tmp2 < ceiling)) then
    cov1[i] = 49
  end
  if Ntmp[i] >= bkn and base2 < ceiling then
    base2[i] = ceiling[i]
  end
  if base2[i] < tmp2[i] and base2[i] < ceiling[i] then
    top2[i] = base2[i]
  end
end

local cov2 = hitool:VerticalMaximumGrid(N, base2, top2)

--// *** 3rd layer, at least BKN
local base3 = hitool:VerticalHeightGreaterThanGrid(N, top2, maxHdata, bkndata,1)

--// bkn+ base cannot be lower than ceiling
for i = 1, #base3 do
  if base3[i] < ceiling[i] then
    base3[i] = ceiling[i]
  end
end

local top3 = hitool:VerticalHeightLessThanGrid(N, base3, maxHdata, sctdata,1)
local cov3 = hitool:VerticalMaximumGrid(N, base3, top3)

--// *** (at least SCT) 4th layer, in case any lower layer is discarded
local base4 = hitool:VerticalHeightGreaterThanGrid(N, top3, maxHdata, sctdata,1)
local top4 = hitool:VerticalHeightLessThanGrid(N, base4, maxHdata, sctdata,1)

tmp2 = hitool:VerticalHeightGreaterThanGrid(N, base4, top4, bkndata,1)
Ntmp = hitool:VerticalMaximumGrid(N, top3, base4)

for i = 1, #base4 do
  if Ntmp[i] >= bkn and base4[i] < ceiling[i] then
    base4[i] = ceiling[i]
  end
  if base4[i] < tmp2[i] and base4[i] < ceiling[i] then
    top4[i] = base4[i]
  end
end

local cov4 = hitool:VerticalMaximumGrid(N, base4, top4)

--// *** (at least BKN) 5th layer, in case two lower layers are discarded
local base5 = hitool:VerticalHeightGreaterThanGrid(N, top4, maxHdata, bkndata,1)

for i = 1, #base5 do
  if base5[i] < ceiling[i] then
    base5[i] = ceiling[i]
  end
end

local top5 = hitool:VerticalHeightLessThanGrid(N, base4, maxHdata, sctdata,1)
local cov5 = hitool:VerticalMaximumGrid(N, base5, top5)

for i = 1, #base1 do
  base1[i] = math.floor(base1[i]/0.3048/100)*100
  base2[i] = math.floor(base2[i]/0.3048/100)*100
  base3[i] = math.floor(base3[i]/0.3048/100)*100
  base4[i] = math.floor(base4[i]/0.3048/100)*100
  base5[i] = math.floor(base5[i]/0.3048/100)*100
end


local changed = {}
--// case 1: 2 layers found, too close, choose based on cover

for i = 1, #base1 do
  changed[i] = 0
  if IsMissing(base3[i]) and (base2[i] - base1[i]) < dz then
    --// 1st FEW-SCT, 2nd BKN, keep only the higher
    if cov1[i] < bkn  then
      if cov2[i] >= bkn then
        base1[i] = base2[i]
        top1[i] = top2[i]
        cov1[i] = cov2[i] 
        base2[i] = missing
        top2[i] = missing
        cov2[i] = missing
      end
    --// 1st BKN+, keep only the lower
    else
      base2[i] = missing
      top2[i] = missing
      cov2[i] = missing
    end
  end

  --// cases 2a-2c, 3 or more layers found
  if base2[i] - base1[i] < dz and base3[i] - base2[i] >= dz then
    --// 1st and 2nd FEW-SCT (too close), 3rd BKN+
    if cov1[i] < bkn and cov2[i] < bkn then
      base2[i] = base3[i]
      top2[i] = top3[i]
      cov2[i] = cov3[i]
      --// assume that always base4-base3 >=dz (i.e not checked)
      base3[i] = base4[i]
      top3[i] = top4[i]
      cov3[i] = cov4[i]
      changed[i] = 1
    end
    --// 1st FEW-SCT (too close to 2nd), 2nd and 3rd BKN+
    if cov1[i] < bkn and cov2[i] >= bkn and changed[i] == 0 then
      base1[i] = base2[i]
      top1[i] = top2[i]
      cov1[i] = cov2[i]
      base2[i] = base3[i]
      top2[i] = top3[i]
      cov2[i] = cov3[i]
      --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
      base3[i] = base4[i]
      top3[i] = top4[i]
      cov3[i] = cov4[i]
      base4[i] = base5[i]
      top4[i] = top5[i]
      cov4[i] = cov5[i]
      changed[i] = 2
    end
    --// 1st BKN+ (too close to 2nd), 2nd SCT-BKN+, 3rd BKN+
    if cov1[i] >= bkn and changed[i] == 0 then
      --// base1 and base3 not too close (because base3-base2>=dz), discard 2nd (becomes bkn)
      base2[i] = base3[i]
      top2[i] = top3[i]
      cov2[i] = cov3[i]
      if cov4[i] < bkn then
        base3[i] = base5[i]
	top3[i] = top5[i]
	cov3[i] = cov5[i]
      end
      if cov4[i] >= bkn then
        base3[i] = base4[i]
	top3[i] = top4[i]
	cov3[i] = cov4[i]
      end
      changed[i] = 3
    end
  end

  --// case 2b: 3 or more layers found, 1st not too close to 2nd, 2nd too close to 3rd
  if base2[i] - base1[i] >= dz and base3[i] - base2[i] < dz and changed[i] == 0 then
    --// 1st FEW-BKN+, 2nd SCT, 3rd BKN+ (too close to 2nd)
    if cov2[i] < bkn then
      base2[i] = base3[i]
      top2[i] = top3[i]
      cov2[i] = cov3[i]
    end
    --// choose new 3rd based on cover
    --// (cov4 = sct+, i.e. don't accept possibly sct as 3rd layer)
    --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
    if cov4[i] < bkn then
      base3[i] = base5[i]
      top3[i] = top5[i]
      cov3[i] = cov5[i]
    end
    if cov4[i] >= bkn then
      base3[i] = base4[i]
      top3[i] = top4[i]
      cov3[i] = cov4[i]
    end
    changed[i] = 4
  end

  --// 1st FEW-BKN+, 2nd BKN+ too close to 3rd BKN+
  if cov2[i] >= bkn and changed[i] == 0 then
    --// discard 3rd
    --// choose new 3rd based on cover
    --// (cov4 = sct+, i.e. don't accept possibly sct as 3rd layer)
    --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
    if cov4[i] < bkn then
      base3[i] = base5[i]
      top3[i] = top5[i]
      cov3[i] = cov5[i]
    end
    if cov4[i] >= bkn then
      base3[i] = base4[i]
      top3[i] = top4[i]
      cov3[i] = cov4[i]
    end
    changed[i] = 5
  end

  --// case 2c: 3 or more layers found, 1st too close to 2nd and 2nd too close to 3rd
  if base2[i] - base1[i] < dz and base3[i] - base2[i] < dz and changed[i] == 0 then
    --// 1st and 2nd FEW-SCT, 3rd BKN+ (1st-2nd and 2nd-3rd too close)
    if cov1[i] < bkn and cov2[i] < bkn then
      --// 1st and 3rd not too close, discard 2nd
      if base3[i] - base1[i] >= dz then
        base2[i] = base3[i]
	top2[i] = top3[i]
	cov2[i] = cov3[i]

	--// choose new 3rd based on cover
        --// (cov4 = sct+, i.e. don't accept possibly sct as 3rd layer)
        --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
	if cov4[i] < bkn then
          base3[i] = base5[i]
	  top3[i] = top5[i]
	  cov3[i] = cov5[i]
	end
	if cov4[i] >= bkn then
	  base3[i] = base4[i]
	  top3[i] = top4[i]
	  cov3[i] = cov4[i]
	end
      end
      changed[i] = 6
    else
      base1[i] = base3[i]
      top1[i] = top3[i]
      cov1[i] = cov3[i]
      --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
      --// 2nd becomes sct+, 3rd becomes bkn+
      base2[i] = base4[i]
      top2[i] = top4[i]
      cov2[i] = cov4[i]
      base3[i] = base5[i]
      top3[i] = top5[i]
      cov3[i] = cov5[i]
      changed[i] = 7
    end
  end

  --// 1st FEW-SCT, 2nd and 3rd BKN+ (1st-2nd and 2nd-3rd too close)
  if cov1[i] < bkn and cov2[i] >= bkn and changed[i] == 0 then
    --// keep only 2nd (bkn)
    base1[i] = base2[i]
    top1[i] = top2[i]
    cov1[i] = cov2[i]
    --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
    --// 2nd becomes sct+, 3rd becomes bkn+
    base2[i] = base4[i]
    top2[i] = top4[i]
    cov2[i] = cov4[i]
    base3[i] = base5[i]
    top3[i] = top5[i]
    cov3[i] = cov5[i]
    changed[i] = 8
  end

  --// 1st BKN+, 2nd SCT-BKN+ and 3rd BKN+ (1st-2nd and 2nd-3rd too close)
  if cov1[i] >= bkn and changed[i] == 0 then
    --//  1st also too close to 3rd
    if base3[i] - base1[i] < dz then
      base2[i] = base4[i]
      top2[i] = top4[i]
      cov2[i] = cov4[i]
      base3[i] = base5[i]
      top3[i] = top5[i]
      cov4[i] = cov5[i]
      changed[i] = 9
  
    --// 1st not too close to 3rd
    else
      --// discard 2nd (becomes bkn+)
      base2[i] = base3[i]
      top2[i] = top3[i]
      cov2[i] = cov3[i]
      --// choose new 3rd based on cover
      --// (cov4 = sct+, i.e. don't accept possibly sct as 3rd layer)
      --// assume that always base4-base3 and base5-base4 >=dz (i.e not checked)
      if cov4[i] < bkn then
        base3[i] = base5[i]
        top3[i] = top5[i]
        cov3[i] = cov5[i]
      end
      if cov4[i] >= bkn then
        base3[i] = base4[i]
        top3[i] = top4[i]
        cov3[i] = cov4[i]
      end
      changed[i] = 10
    end
  end
end

-- write output
p = param("CLOUDBASE1-FT")
result:SetValues(base1)
result:SetParam(p)
luatool:WriteToFile(result)

p = param("CLOUDBASE2-FT")
result:SetValues(base2)
result:SetParam(p)
luatool:WriteToFile(result)

p = param("CLOUDBASE3-FT")
result:SetValues(base3)
result:SetParam(p)
luatool:WriteToFile(result)

p = param("CLOUDCOV1-0TO1")
result:SetValues(cov1)
result:SetParam(p)
luatool:WriteToFile(result)

p = param("CLOUDCOV2-0TO1")
result:SetValues(cov2)
result:SetParam(p)
luatool:WriteToFile(result)

p = param("CLOUDCOV3-0TO1")
result:SetValues(cov3)
result:SetParam(p)
luatool:WriteToFile(result)

