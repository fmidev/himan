local U = {}

function U.round(n)
  return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

-- Returns a filter kernel as a matrixf.
-- resolution_km: grid resolution in km
-- radius_km: smoothing radius in km
-- style: "uniform" for mask of 1s, "avg" for averaging (1/count per active cell)
-- shape: "square" or "circle"
function U.create_filter(resolution_km, radius_km, style, shape)
  local shape_weights = {
    square = function(i, j, center, grid_radius) return 1 end,
    circle = function(i, j, center, grid_radius)
      return math.sqrt((i - center)^2 + (j - center)^2) <= grid_radius and 1 or 0
    end,
  }

  if not shape_weights[shape] then
    logger:Error("Invalid shape given to create_filter")
    return
  end

  if style ~= "uniform" and style ~= "avg" then
    logger:Error("Invalid style given to create_filter")
    return
  end

  if type(resolution_km) ~= "number" or resolution_km <= 0 then
    logger:Error("resolution_km must be a positive number")
    return
  end

  if type(radius_km) ~= "number" or radius_km <= 0 then
    logger:Error("radius_km must be a positive number")
    return
  end

  local grid_radius = math.floor(radius_km / resolution_km)
  local size = 2 * grid_radius + 1
  local center = grid_radius
  local kernel = {}
  local weight_sum = 0

  for i = 0, size - 1 do
    for j = 0, size - 1 do
      local w = shape_weights[shape](i, j, center, grid_radius)
      kernel[i * size + j + 1] = w
      weight_sum = weight_sum + w
    end
  end

  if style == "avg" then
    for i = 1, #kernel do
      kernel[i] = kernel[i] / weight_sum
    end
  end

  local f = matrixf(size, size, 1, missing)
  f:SetValues(kernel)
  return f
end

return U