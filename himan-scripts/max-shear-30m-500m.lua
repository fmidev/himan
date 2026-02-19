logger:Info("Calculating Maximum wind shear in 100ft-1600ft level")

local MISS = missing
local windShear = param("WSHR-KTHFT")  -- Wind shear

-- units to meters (this is also a default)
hitool:SetHeightUnit(HPParameterUnit.kM)

-- hight in meters
local lowlimit = 30     -- ~100ft
local highlimit = 500   -- ~1600ft

-- Maximum wind shear
local maxShearData = hitool:VerticalMaximum(windShear, lowlimit, highlimit)

-- write 
result:SetParam(param("WSHR-MAX-KTHFT"))
result:SetValues(maxShearData)
luatool:WriteToFile(result)
