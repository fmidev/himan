/*
 * uv_index.h
 */

#ifndef UV_INDEX_H
#define UV_INDEX_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

#include <memory>
#include <vector>

namespace himan
{
namespace plugin
{
// 7-D Lagrange interpolation lookup table (disort_24012003.dat). Loaded once per
// plugin instance and read concurrently by worker threads.
class disort_table
{
   public:
	void Load(const std::string& path);
	double Interpolate(double albedo, double ozone, double atau, double assa, double ctau, double sza,
	                   double alt) const;

   private:
	// Flat storage of the 7-D look-up. Each entry is a clear-sky CIE-weighted
	// UV irradiance ("cie rate") produced by DISORT for the corresponding
	// (alt, sza, ctau, assa, atau, albedo, ozone) point. Indexed via
	// `TableIndex()` (see uv_index.cpp).
	std::vector<double> itsCieRates;
};

// Total-ozone climatology stored as 5 Fourier coefficients per 1°×1° cell on a
// global lat-lon grid. Reconstructed at a given Julian day via:
//   clim(lat, lon, jd) = B0 + B1·cos(θ) + B2·sin(θ) + B3·cos(2θ) + B4·sin(2θ)
// with θ = 2π·jd/365. Bilinear in lat/lon, evaluated lazily per query.
class o3_climatology
{
   public:
	void Load(const std::string& path, std::shared_ptr<const plugin_configuration> conf);
	// Climatological total ozone (Dobson units) at the given lat/lon and Julian day.
	double Evaluate(double latDeg, double lonDeg, int julianDay) const;

   private:
	struct coeffs
	{
		double B[5];
	};
	int itsNLat = 0;
	int itsNLon = 0;
	double itsLatStart = 0.0;
	double itsLonStart = 0.0;
	double itsLatStep = 1.0;
	double itsLonStep = 1.0;
	std::vector<coeffs> itsGrid;  // size = itsNLat * itsNLon, row-major (lat outer, lon inner)
};

class uv_index : public compiled_plugin, private compiled_plugin_base
{
   public:
	uv_index();
	inline virtual ~uv_index() = default;
	uv_index(const uv_index&) = delete;
	uv_index& operator=(const uv_index&) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::uv_index";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short threadIndex);

	enum class mode_t
	{
		// Daily-max UV-index only (UVIMAX-N, solar-noon SZA).
		kUvimax,
		// Instantaneous UV-index only (UVI-N, SZA at valid time).
		kUvi,
		// Both UVIMAX-N and UVI-N in a single pass — shares the input
		// fetches and per-point arithmetic up to the SZA-dependent disort
		// lookup. Useful when both fields are wanted at the same cadence.
		kUv,
		// Total-ozone anomaly vs climatology, output as O3ANOM-PRCNT.
		kO3anom
	};

	mode_t itsMode = mode_t::kUv;
	disort_table itsDisortTable;
	o3_climatology itsO3Clim;
	// Aerosol climatology fields loaded once and stored for use in Calculate().
	std::shared_ptr<info<double>> itsAtauSummer;
	std::shared_ptr<info<double>> itsAtauWinter;
	std::shared_ptr<info<double>> itsAssaSummer;
	std::shared_ptr<info<double>> itsAssaWinter;
};

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<uv_index>(new uv_index());
}
}  // namespace plugin
}  // namespace himan

#endif /* UV_INDEX_H */
