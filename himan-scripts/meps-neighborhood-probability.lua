-- Calculate probabilities from meps with area and time averaging
--

local MISS = missing

local mask = matrix(33, 33, 1, MISS)
mask:Fill(1)

function produceProbabilities(sourceparam, targetparam, op, limit)

  local datas = {}

  logger:Info("Producing area probabilities for " .. sourceparam:GetName())

  for i=0,9 do -- TODO: get this from database
    local curtime = forecast_time(current_time:GetOriginDateTime(), current_time:GetValidDateTime())

    local ftype = forecast_type(HPForecastType.kEpsControl, i)

    if i > 0 then
      ftype:SetType(HPForecastType.kEpsPerturbation)
    end

    for j=0,3 do -- Look for the past 3 hours
      local data = luatool:FetchInfoWithType(curtime, current_level, sourceparam, ftype)

      if data then
        local reduced = nil
        if op == ">" then
          reduced = ProbLimitGt2D(data:GetData(), mask, limit):GetValues()
        elseif op == "==" then
          reduced = ProbLimitEq2D(data:GetData(), mask, limit):GetValues()
        end
        datas[#datas+1] = reduced
      end
      curtime:GetValidDateTime():Adjust(HPTimeResolution.kHourResolution, -1)

    end
  end

  logger:Info(string.format("Read %d grids", #datas))

  if #datas == 0 then
    return
  end

  local prob = {}

  for i=1,#datas[1] do
    prob[i] = MISS

    local tmp = 0

    for j=1,#datas do
      tmp = tmp + datas[j][i]
    end

    prob[i] = tmp / (#datas)
  end

  result:SetParam(targetparam)
  result:SetValues(prob)

  luatool:WriteToFile(result)

end

local sourceparam = configuration:GetValue("source_param")
local targetparam = configuration:GetValue("target_param")
local op = configuration:GetValue("comparison_op")
local limit = tonumber(configuration:GetValue("limit"))

if sourceparam == "" or targetparam == "" or op == "" or limit == nil then
  logger:Error("Source or target param or limit not specified")
  return
end

logger:Info(string.format("Source: %s Target: %s Operator: %s Limit: %d", sourceparam, targetparam, op, limit))
produceProbabilities(param(sourceparam), param(targetparam), op, limit)
