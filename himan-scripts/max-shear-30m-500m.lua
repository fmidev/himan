-- STU-29741
-- Calculating maximum wind shear in 100ft-1600ft level
logger:Info("Calculating maximum wind shear in 100ft-1600ft level")

local MISS = missing                   -- Missing value placeholder
local windShear = param("WSHR-KTHFT")  -- Wind shear parameter

-- Set height unit to meters (default)
hitool:SetHeightUnit(HPParameterUnit.kM)

-- Height limits in meters corresponding to ~100ft and ~1600ft
local lowlimit = 30     -- ~100ft
local highlimit = 500   -- ~1600ft

-- Calculate maximum wind shear within the defined height layer
local maxShearData = hitool:VerticalMaximum(windShear, lowlimit, highlimit)

-- Set output parameter and values, then write result to file
result:SetParam(param("WSHR-MAX-KTHFT"))
result:SetValues(maxShearData)
luatool:WriteToFile(result)
