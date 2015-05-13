logger:Info("Calculating N at " .. current_level:GetValue() .. " hPa")

nparam = param("N-PRCNT")

hitool:SetHeightUnit(HPParameterUnit.kHPa)
data = hitool:VerticalValue(nparam, current_level:GetValue())

scale = false

if not data then
	nparam = param("N-0TO1")
	logger:Info("Trying N-0TO1")

	data = hitool:VerticalValue(nparam, current_level:GetValue())
	
	if not data then
		logger:Error("Data not found")
		return
	end

	scale = true
end

if scale then
	for i = 1, #data do
		local val = data[i]
		if value then
			data[i] = val * 100 -- scale to percents
		end
	end
end

result:SetParam(nparam)
result:SetValues(data)

logger:Info("Writing results")
luatool:WriteToFile(result)
