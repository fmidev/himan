-- Cloud ceiling code, smarttool script including comments by Simo Neiglick
-- https://wiki.fmi.fi/display/PROJEKTIT/Ceiling_ft
-- partio 2/2015  / junila 5/2015
 
logger:Info("Calculating Cloud ceiling in feet")
  
local kFloatMissing = kFloatMissing
-- another value for missing data is needed for output because cloud ceiling in feet might get mixed up with kFloatMissing
local Missing = -1 

-- Max height to check for cloud base [m]
local maxH = 14000
 
-- Threshold for N (cloud amount) to calculate base [%]
local Nthreshold = 0.55
 
local N = param("N-0TO1")
 
-- findh: 1 = first value (of threshold) from sfc upwards
local Nheight = hitool:VerticalHeight(N, 0, maxH, Nthreshold, 1)
 
if not Nheight then
    N = param("N-PRCNT")
     
-- N-PRCNT values also 0...1 despite the name
 
    Nheight = hitool:VerticalHeight(N, 0, maxH, Nthreshold, 1)
end

 
if not Nheight then
    logger:Error("No data found")
    return
end
 
-- N max value near the surface (15m ~ 50ft)
local lowNmax = hitool:VerticalMaximum(N, 0, 15)
 
if not lowNmax then
    logger:Error("No data found")
    return
end
 
local ceiling = {}
 
for i = 1, #lowNmax do
    local nh = Nheight[i]
    local nmax = lowNmax[i]
 
    local ceil = Missing
 
    if nh ~= kFloatMissing and nmax ~= kFloatMissing then
 
        -- Nthreshold is not always found for low clouds starting from ~sfc
 
        if nmax > Nthreshold then
            nh = 14
        end
 
        -- Result converted to feet:
        -- below 100ft at 50ft resolution  (15m ~ 50ft)
        -- below 10000ft at 100ft resolution  (10000ft = 3048m)
        -- above 10000ft at 500ft resolution
 
        if nh < 15 then
            ceil = math.floor(0.5 + nh/30.48*2) * 50
        elseif nh < 3048 then
            ceil = math.floor(0.5 + nh/30.48) * 100
        else
            ceil = math.floor(0.5 + nh/304.8*2) * 500
        end
    end
 
    ceiling[i] = ceil
 
end
 
result:SetParam(param("CL-FT"))
result:SetMissingValue(Missing)
result:SetValues(ceiling)
 
luatool:WriteToFile(result)
