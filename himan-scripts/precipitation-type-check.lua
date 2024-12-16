logger:Info("Precipitation type check for Vire data")

par_prec = param('PRECFORM2-N')
par_t = param('T-K')

ftype = forecast_type(HPForecastType.kDeterministic)
l0 = level(HPLevelType.kHeight, 0)
l2 = level(HPLevelType.kHeight, 2)

prec = luatool:Fetch(current_time, l0, par_prec, ftype)
t = luatool:Fetch(current_time, l2, par_t, ftype)


for i=1, #prec do
    t[i] = t[i] - 273.15
    if (prec[i] == 4 or prec[i] == 5) and t[i] > 0 then
        prec[i] = 1
    end
    if (prec[i] == 4 or prec[i] == 5) and t[i] < -10 then
        prec[i] = 3
    end
    if (prec[i] == 1 or prec[i] == 0 or prec[i] == 2) and t[i] < - 0.5 then
        prec[i] = 3
    end
    if (prec[i] == 1 or prec[i] == 0) and (t[i] > 0.5 and t[i] <= 1.5) then
        prec[i] = 2
    end
    if prec[i] == 3 and t[i] > 1 then
        prec[i] = 2
    end
    if (prec[i] == 2 or prec[i] == 3) and t[i] > 1.5 then
        prec[i] = 1
    end
end


result:SetParam(param("PRECFORM-N"))
result:SetValues(prec)

luatool:WriteToFile(result)