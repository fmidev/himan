local U = {}

-- Upper bound for create_mask kernel half-width, in grid cells.
local MAX_GRID_RADIUS = 500

function U.round(n)
  if n >= 0 then
    return math.floor(n + 0.5)
  else
    return math.ceil(n - 0.5)
  end
end

-- Length of one degree of latitude, km. Based on the newbase sphere (r=6371220 m)
-- used throughout himan; see ELLIPS_NEWBASE in earth_shape.h.
local KM_PER_DEGREE = 2 * math.pi * 6371.220 / 360

-- Returns the approximate grid resolution in kilometres.
-- Projected grids (lambert, stereographic, ...) store Di/Dj in metres, but
-- geographic grids store them in degrees, so the conversion depends on grid type.
function U.grid_resolution_km(grid)
  local gtype = grid:GetGridType()

  if gtype == HPGridType.kLatitudeLongitude or gtype == HPGridType.kRotatedLatitudeLongitude then
    -- Dj (north-south) is used because a degree of latitude has a constant length,
    -- whereas a degree of longitude shrinks by cos(latitude).
    return grid:GetDj() * KM_PER_DEGREE
  end

  return grid:GetDi() / 1000
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
      local dx, dy = i - center, j - center
      return (dx * dx + dy * dy) <= (grid_radius * grid_radius) and 1 or 0
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

  -- A kernel this large is never intentional; it usually means resolution_km was
  -- passed in the wrong unit. Without this check the allocation below crashes.
  if grid_radius > MAX_GRID_RADIUS then
    logger:Error(string.format(
      "create_mask: radius %.1f km at resolution %.4f km needs a %dx%d kernel, refusing (max radius %d cells)",
      radius_km, resolution_km, 2 * grid_radius + 1, 2 * grid_radius + 1, MAX_GRID_RADIUS))
    return
  end

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