--[[
 
icing max

Find the maximum vertical value of icing index between 0 and 8000m

]]

hitool:SetHeightUnit(HPParameterUnit.kM)
maxicing = hitool:VerticalMaximum(param("ICING-N"),0,8000)

result:SetParam(param("MAXICING-N"))
result:SetValues(maxicing)

logger:Info("Writing source data to file")
luatool:WriteToFile(result)
