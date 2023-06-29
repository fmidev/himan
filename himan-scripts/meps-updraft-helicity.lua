--Updraft helicity
--Defined as integral from 2km-5km over w*(dv/dx-du/dy) dz

local mode = configuration:GetValue("mode")
local dx = result:GetGrid():GetDi()
local dy = result:GetGrid():GetDj()
local ni = result:GetGrid():GetNi()
local nj = result:GetGrid():GetNj()

--Helper function to compute gradients of the velocity field using a central difference scheme.
function ddx(v,dx)
  local res = {}
  for j=0,nj-1 do
    for i=0,ni-1 do
      if i == 0 then
        res[i+j*ni+1] = (v[i+j*ni+2]-v[i+j*ni+1])/dx
      elseif i == ni-1 then
        res[i+j*ni+1] = (v[i+j*ni+1]-v[i+j*ni])/dx
      else
        res[i+j*ni+1] = (v[i+j*ni+2]-v[i+j*ni])/(2*dx)
      end
    end
  end
  return res
end

function ddy(u,dy)
  local res = {}
  for j=0,nj-1 do
    for i=0,ni-1 do
      if j == 0 then
        res[i+j*ni+1] = (u[i+(j+1)*ni+1]-u[i+j*ni+1])/dy
      elseif j == nj-1 then
        res[i+j*ni+1] = (u[i+j*ni+1]-u[i+(j-1)*ni+1])/dy
      else
        res[i+j*ni+1] = (u[i+(j+1)*ni+1]-u[i+(j-1)*ni+1])/(2*dy)
      end
    end
  end
  return res
end

--A pre calculation that computes the integrand of the updraft helicity on the hybrid levels.
--The precalculation is yielding intermediate results that should be written to cache.
if mode == "precalculation" then
  U = luatool:FetchWithType(current_time, current_level, param("U-MS"), current_forecast_type)
  V = luatool:FetchWithType(current_time, current_level, param("V-MS"), current_forecast_type)
  W = luatool:FetchWithType(current_time, current_level, param("VV-MS"), current_forecast_type)

  dudy = ddy(U,dy)
  dvdx = ddx(V,dx)

  local res_pre = {}
  for i=1,#W do
    res_pre[i] = W[i] * (dvdx[i]-dudy[i])
  end
  p = param("UHELPRE-M2S2")

  result:SetValues(res_pre)
  result:SetParam(p)
  luatool:WriteToFile(result)
end

--The final calulation is using the hitool to vertically integrate the function on the interval from 2-5km.
--As Vertical average is defined as Integral f(z) dz / (z1-z0) we can obtain the vertical integral through
--multiplication of the vertical average with the interval, i.e. 3000m
if mode == "finalize" then
  hitool:SetHeightUnit(HPParameterUnit.kM)
  integral = hitool:VerticalAverage(param("UHELPRE-M2S2"),2000,5000)
  for i=1,#integral do
    integral[i] = integral[i] * 3000
  end

  p = param("UHEL-M2S2")

  result:SetValues(integral)
  result:SetParam(p)
  luatool:WriteToFile(result)
end
