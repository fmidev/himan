function round(n)
  return n % 1 >= 0.5 and math.ceil(n) or math.floor(n)
end

-- Returns a filter kernel as a matrixf.
-- resolution_km: grid resolution in km
-- radius_km: smoothing radius in km
-- style: "uniform" for mask of 1s, "avg" for averaging (1/count per active cell)
-- shape: "square" or "round"
function create_filter(resolution_km, radius_km, style, shape)
  if shape ~= "square" and shape ~= "round" then
    logger:Error("Invalid shape given to create_filter")
    return
  end
  
  if style ~= "uniform" and style ~= "avg" then
    logger:Error("Invalid style given to create_filter")
    return
  end
  
  local grid_radius = math.floor(radius_km / resolution_km)
  local size = 2 * grid_radius + 1
  local center = grid_radius
  local kernel = {}
  local count = 0

  for i = 0, size - 1 do
    for j = 0, size - 1 do
      if shape == "square" or math.sqrt((i - center)^2 + (j - center)^2) <= grid_radius then
        kernel[i * size + j + 1] = 1
        count = count + 1
      else
        kernel[i * size + j + 1] = 0
      end
    end
  end

  if style == "avg" then
    for i = 1, #kernel do
      if kernel[i] ~= 0 then
        kernel[i] = kernel[i] / count
      end
    end
  end

  local f = matrixf(size, size, 1, missing)
  f:SetValues(kernel)
  return f
end
