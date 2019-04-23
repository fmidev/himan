-- accumulated total precipitation of past X hours
-- hours need to be passed as additional configuration argument "accumulation_period" : "X"
-- if beginning of accumulation period would be ahead of analysis time, default value missing is returned

  local acc_period = configuration:GetValue("accumulation_period")
  logger:Info("Calculating accumulated " .. acc_period .. "h total precipitation")


  local currRR = luatool:FetchWithType(current_time, current_level, param("RR-KGM2"), current_forecast_type)
  current_time:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -tonumber(acc_period))
  local prevRR= luatool:FetchWithType(current_time, current_level, param("RR-KGM2"), current_forecast_type)

  if not currRR or not prevRR then
    return
  end

  local acc = {}

  -- accumulated precipitation between two timesteps is the difference of total precipitation values at given 
  for i=1, #currRR do
    acc[i] = currRR[i] - prevRR[i]
  end

  result:SetValues(acc)
  result:SetParam(param("RR" .. acc_period .. "H-KGM2"))
  luatool:WriteToFile(result)
