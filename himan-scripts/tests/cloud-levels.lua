local script = assert(arg[1], "Pass the path to cloud-levels.lua")
local missing = 0 / 0
local function IsMissing(value)
  return value ~= value
end

local function Check(name, bases, covs, expected)
  local heights, tops = {}, {}
  for j = 1, 5 do
    heights[j] = bases[j] and (bases[j] + 1) * 0.3048 or missing
    tops[j] = heights[j] + 1
    covs[j] = covs[j] or missing
  end

  local calls = {
    {"VerticalHeightGreaterThan", heights[1]},
    {"VerticalHeightLessThanGrid", tops[1]},
    {"VerticalHeightLessThanGrid", tops[1]},
    {"VerticalMaximumGrid", 0},
    {"VerticalHeightGreaterThanGrid", missing},
    {"VerticalHeightGreaterThanGrid", missing},
    {"VerticalMaximumGrid", 0},
    {"VerticalMaximumGrid", covs[1]},
    {"VerticalHeightGreaterThanGrid", heights[2]},
    {"VerticalHeightLessThanGrid", tops[2]},
    {"VerticalHeightGreaterThanGrid", missing},
    {"VerticalMaximumGrid", 0},
    {"VerticalMaximumGrid", covs[2]},
    {"VerticalHeightGreaterThanGrid", heights[3]},
    {"VerticalHeightLessThanGrid", tops[3]},
    {"VerticalMaximumGrid", covs[3]},
    {"VerticalHeightGreaterThanGrid", heights[4]},
    {"VerticalHeightLessThanGrid", tops[4]},
    {"VerticalHeightGreaterThanGrid", missing},
    {"VerticalMaximumGrid", 0},
    {"VerticalMaximumGrid", covs[4]},
    {"VerticalHeightGreaterThanGrid", heights[5]},
    {"VerticalHeightLessThanGrid", tops[5]},
    {"VerticalMaximumGrid", covs[5]}
  }
  local nextCall = 0
  local hitool = {}
  for _, method in ipairs({
    "VerticalHeightGreaterThan", "VerticalHeightGreaterThanGrid",
    "VerticalHeightLessThanGrid", "VerticalMaximumGrid"
  }) do
    hitool[method] = function()
      nextCall = nextCall + 1
      local call = assert(calls[nextCall], name .. ": unexpected hitool call")
      assert(call[1] == method, name .. ": unexpected hitool method")
      return {call[2]}
    end
  end

  local outputs = {}
  local result = {
    SetValues = function(self, values) self.values = values end,
    SetParam = function(self, parameter) self.parameter = parameter end
  }
  local environment = setmetatable({
    missing = missing, IsMissing = IsMissing, hitool = hitool,
    param = function(name) return name end,
    result = result,
    luatool = {
      Fetch = function() return {0} end,
      WriteToFile = function(_, output)
        outputs[output.parameter] = output.values[1]
      end
    }
  }, {__index = _G})

  assert(loadfile(script, "t", environment))()
  assert(nextCall == #calls, name .. ": unconsumed hitool calls")
  for j = 1, 3 do
    local idx = expected[j]
    local base = outputs["CL" .. j .. "-FT"]
    local cov = outputs["CLCOV" .. j .. "-0TO1"]
    if idx then
      assert(base == bases[idx], name .. ": incorrect base " .. j)
      assert(cov == covs[idx] or (IsMissing(cov) and IsMissing(covs[idx])),
             name .. ": incorrect coverage " .. j)
    else
      assert(IsMissing(base) and IsMissing(cov), name .. ": expected missing layer " .. j)
    end
    if j > 1 and not IsMissing(base) then
      assert(base - outputs["CL" .. (j - 1) .. "-FT"] >= 300,
             name .. ": separation below 300 ft")
    end
  end
end

Check("separated FEW/BKN", {1000, 2000, 3000, 4000, 5000}, {.2, .6, .7, .4, .8}, {1, 2, 3})
Check("separated BKN", {1000, 2000, 3000, 4000, 5000}, {.6, .6, .7, .4, .8}, {1, 2, 3})
Check("exact dz", {1000, 1300, 1600, 2000, 2500}, {.6, .6, .7, .4, .8}, {1, 2, 3})
Check("case 2a low/low", {1000, 1100, 2000, 3000, 4000}, {.2, .4, .7, .6, .8}, {1, 3, 4})
Check("case 2a low/BKN", {1000, 1100, 2000, 3000, 4000}, {.2, .6, .7, .4, .8}, {2, 3, 4})
Check("case 2a BKN", {1000, 1100, 2000, 3000, 4000}, {.6, .4, .7, .4, .8}, {1, 3, 5})
Check("case 2b low", {1000, 2000, 2100, 3000, 4000}, {.2, .4, .7, .4, .8}, {1, 3, 5})
Check("case 2b BKN", {1000, 2000, 2100, 3000, 4000}, {.2, .6, .7, .6, .8}, {1, 2, 4})
Check("case 6", {1000, 1200, 1400, 3000, 4000}, {.2, .4, .7, .4, .8}, {1, 3, 5})
Check("case 7", {1000, 1100, 1200, 3000, 4000}, {.2, .4, .7, .4, .8}, {3, 4, 5})
Check("case 8", {1000, 1100, 1200, 3000, 4000}, {.2, .6, .7, .4, .8}, {2, 4, 5})
Check("case 9", {1000, 1100, 1200, 3000, 4000}, {.6, .4, .7, .4, .8}, {1, 4, 5})
Check("case 10", {1000, 1200, 1400, 3000, 4000}, {.6, .4, .7, .6, .8}, {1, 3, 4})
Check("missing replacement", {1000, 2000, 2100}, {.2, .4, .7}, {1, 3})
Check("missing replacement coverage", {1000, 2000, 2100, 3000, 4000}, {.2, .4, .7, missing, .8}, {1, 3, 5})
Check("two low layers", {1000, 1100}, {.2, .4}, {1})
Check("two layers prefer BKN", {1000, 1100}, {.2, .6}, {2})
Check("two layers keep lower BKN", {1000, 1100}, {.6, .7}, {1})
Check("one BKN layer", {1000}, {.6}, {1})
Check("no layers", {}, {}, {})
Check("replacement too close", {1000, 1100, 2000, 2100, 4000}, {.2, .4, .7, .6, .8}, {1, 3})
Check("cascading removals", {1000, 1100, 1200, 1300, 1400}, {.2, .4, .7, .4, .8}, {3})
print("All 22 cloud-level regression cases passed")
