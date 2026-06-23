local U = {}

function U.round(n)
  if n >= 0 then
    return math.floor(n + 0.5)
  else
    return math.ceil(n - 0.5)
  end
end

-- Returns a filter kernel as a matrixf.
-- resolution_km: grid resolution in km
-- radius_km: smoothing radius in km
-- normalize: if true, weights are divided by their sum so the kernel sums to 1
-- shape: "square", "circle", or a function(i, j, center, grid_radius) returning a weight
function U.create_mask(resolution_km, radius_km, shape, normalize)
  local builtins = {
    square = function() return 1 end,
    circle = function(i, j, center, grid_radius)
      return math.sqrt((i - center)^2 + (j - center)^2) <= grid_radius and 1 or 0
    end,
  }

  local weight_fn
  if type(shape) == "function" then
    weight_fn = shape
  elseif builtins[shape] then
    weight_fn = builtins[shape]
  else
    logger:Error("Invalid shape given to create_mask")
    return
  end

  if type(normalize) ~= "boolean" then
    logger:Error("normalize must be a boolean")
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
      local w = weight_fn(i, j, center, grid_radius)
      kernel[i * size + j + 1] = w
      weight_sum = weight_sum + w
    end
  end

  if normalize then
    if weight_sum == 0 then
      logger:Error("create_mask produced zero-sum kernel; cannot normalize")
      return
    end
    for i = 1, #kernel do
      kernel[i] = kernel[i] / weight_sum
    end
  end

  local f = matrixf(size, size, 1, missing)
  f:SetValues(kernel)
  return f
end

return U