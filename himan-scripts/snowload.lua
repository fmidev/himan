-- Calculating snow load
--
-- Original algorithm by Petri Hoppulan.
--
-- STU-9204

-- accumulation functions
function f_wind_acc(v)
	if v >= 14.0 then
		return 1.5
	end
	return 1.0 + 0.5 * math.sin(v/14.0 * math.pi/2.0)
end

function f_load(l)
	if l == 0 then
		return 0.6
	end
	-- original algorithm states l^0.8 >= 10.0 but this is equivalent while saving expensive exponential operations
	if l >= 17.28 then
		return 1.0
	end

	-- linear interpolation for 0 < l < 17.28
	return 0.6 + l/17.28*0.4
end

function wetsnow(rr,T)
	if T < -0.5 then
		return 0.0
	elseif T <= 0.2 then
		-- linear function between -0.5 deg and 0.2 deg
		return 1.3*rr*(T+0.5)/0.7
	elseif T <= 0.8 then
		-- linear function between 0.2 deg and 0.8 deg
		return (1.3-0.3*(T-0.2)/0.6)*rr
	elseif T <= 1.5 then
		-- linear function between 0.8 deg and 1.5 deg
		return (1.0-(T-0.8)/0.7)*rr
	end
	return 0.0
end

-- decreasing functions
function f_wind_dec(v)
        if v <= 6.0 then
		return 1.0
	elseif v>=24.0 then
		return 0.0
	end
	-- linear interpolation between 6 m/s and 24 m/s
	return (24.0-v)/18.0
end

function f_temperature(T)
	if T <= 0.8 then
		return 1.0
	elseif T >= 15.0 then
		return 0.0
	end
	-- linear interpolation between 0.8 deg and 15.0 deg
	return (15.0-T)/14.2
end

function f_sunrad(phi)
	if phi <= 250.0 then
		return 1.0
	elseif phi >= 3000.0 then
		return 0.0
	end
	-- linear interpolation between 250 and 3000 W/m2
	return (3000.0-phi)/2750.0
end

local Missing = missing

logger:Info("Calculating Snow load")

Wetsnow = {}

result:ResetTime()
while result:NextTime() do
	local curtime = result:GetTime()
	if (curtime:GetStep():Hours() ~= 0) then
		-- fetch input data
		local T = luatool:FetchWithType(curtime, level(HPLevelType.kHeight,2), param("T-K"), current_forecast_type) -- fetch temperature
		local rr = luatool:FetchWithType(curtime, current_level, param("RRR-KGM2"), current_forecast_type) -- fetch precipitation
		local phi = luatool:FetchWithType(curtime, current_level, param("RADGLO-WM2"), current_forecast_type) -- fetch solar radiation
		local v = luatool:FetchWithType(curtime, level(HPLevelType.kHeight,10), param("FF-MS"), current_forecast_type) -- fetch wind speed

		if T and rr and phi and v then
			for i=1, #T do
				-- fill Wetsnow with values 0 at initial timestep
				if table.getn(Wetsnow) < #T  then
						Wetsnow[i] = 0
				end
				T[i] = T[i] - kKelvin -- convert to celsius
				Wetsnow[i] = Wetsnow[i]*f_wind_dec(v[i])*f_temperature(T[i])*f_sunrad(phi[i]) -- snow decrease
				local delta = f_wind_acc(v[i])*f_load(Wetsnow[i])*wetsnow(rr[i],T[i])
				Wetsnow[i] = Wetsnow[i] + delta -- snow accumulation
			end
		end
	end
	result:SetParam(param("SNOWLOAD-KGM2"))
	result:SetMissingValue(Missing)
	result:SetValues(Wetsnow)
end

luatool:WriteToFile(result)
