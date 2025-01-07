-- Precipitation form check for Vire data
--
-- Changes the form value based on 2 meter temperature value

local par_prec = param('PRECFORM2-N')

if configuration:Exists("param") then
  par_prec = param(configuration:GetValue("param"))
end

local par_t = param('T-K')

local l0 = level(HPLevelType.kHeight, 0)
local l2 = level(HPLevelType.kHeight, 2)

local prec = luatool:Fetch(current_time, l0, par_prec, current_forecast_type)
local t = luatool:Fetch(current_time, l2, par_t, current_forecast_type)

if not prec or not t then
    return
end

for i=1, #prec do
    t[i] = t[i] - 273.15
    -- freezing to rain
    if (prec[i] == 4 or prec[i] == 5) and t[i] > 0 then
        prec[i] = 1
    end
    -- freezing to snow
    if (prec[i] == 4 or prec[i] == 5) and t[i] < -10 then
        prec[i] = 3
    end
    -- drizzle, rain, sleet to snow
    if (prec[i] == 1 or prec[i] == 0 or prec[i] == 2) and t[i] < -0.5 then
        prec[i] = 3
    end
    -- drizzle, rain to sleet
    if (prec[i] == 1 or prec[i] == 0) and (t[i] > -0.5 and t[i] <= 1.5) then
        prec[i] = 2
    end
    -- snow to sleet
    if prec[i] == 3 and t[i] > 1 then
        prec[i] = 2
    end
    -- snow, sleet to rain
    if (prec[i] == 2 or prec[i] == 3) and t[i] > 1.5 then
        prec[i] = 1
    end
end

result:SetParam(param("PRECFORM2-N"))
result:SetValues(prec)

luatool:WriteToFile(result)
