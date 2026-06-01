/**
 * @file uv_index.cpp
 *
 *  ──────────────────────────────  START READING HERE  ──────────────────────────────
 *
 *  What this plugin does
 *  ---------------------
 *  Port of the legacy FMI `uvennuste` / `otsoniano` Fortran-and-C++ tool, fitted into
 *  the himan plugin framework. Operates in one of two modes, selected by the JSON
 *  `mode` option:
 *
 *      mode = "uvimax"  → UVIMAX-N only (clear-sky daily-max UV index,
 *                          solar-noon SZA).
 *      mode = "uvi"     → UVI-N only (clear-sky instantaneous UV index,
 *                          SZA at the forecast valid time).
 *      mode = "uv"      → BOTH UVIMAX-N and UVI-N in one pass — shares the
 *                          input fetches and per-point arithmetic.
 *      mode = "o3anom"  → O3ANOM-PRCNT (100 · (forecast − clim) / clim).
 *
 *  Optional `uvimax_valid_hour` (0..23) gates UVIMAX-N writes to a single
 *  UTC hour of the valid time. UVI-N output is unaffected. Useful when one
 *  config covers both 00 and 12 UTC forecast cycles but UVIMAX is only
 *  wanted at one local time of day.
 *
 *  How the UV mode works (the interesting math)
 *  --------------------------------------------
 *  The expensive part — full atmospheric radiative transfer — was solved OFFLINE
 *  once by the FMI ozone group with the DISORT model in 2003. The result is a
 *  7-dimensional regular table of clear-sky UV irradiance values, stored in
 *  `disort_24012003.dat` (ASCII, ~1.6 MB, 42 336 rows × 4 doubles). The seven
 *  table axes are:
 *
 *      ┌────────┬──────┬─────────────┐
 *      │ axis   │ size │ step / unit │
 *      ├────────┼──────┼─────────────┤
 *      │ albedo │   4  │ 0.3         │
 *      │ ozone  │   7  │ 80 Dobson   │
 *      │ atau   │   8  │ 0.2         │
 *      │ assa   │   3  │ 0.2         │
 *      │ ctau   │   9  │ 15 (cloud)  │
 *      │ sza    │   7  │ 15 degrees  │
 *      │ alt    │   4  │ 1000 m      │
 *      └────────┴──────┴─────────────┘
 *
 *  At runtime, for each grid point we assemble the corresponding input vector,
 *  look up the table by tensor-product Lagrange interpolation (degree ≤ 3 per
 *  axis), multiply by an Earth-Sun-distance correction E0(jd) and a fixed scaling
 *  factor of 40, and obtain the UV index. The cloud axis (ctau) is hard-coded to 0
 *  because we model clear-sky only
 *
 *  Per-point input assembly:
 *      albedo = 0.4 if snow water-equiv > 5 mm else 0.03
 *      ozone  = TOZONE-KGM2 × 46696.24   (kg/m² → Dobson units)
 *      atau   = aerosol optical depth, climatology lookup (hemisphere × season)
 *      assa   = aerosol single-scattering albedo, same source as atau
 *      sza    = solar zenith angle:
 *                 - UVIMAX-N: local-solar-noon  (`SolarNoonZenithAngle`)
 *                 - UVI-N   : at forecast valid time (`metutil::ElevationAngle_`)
 *      alt    = surface geopotential / g
 *


 *  How the anomaly mode works
 *  --------------------------
 *  Reads the TOMS-derived monthly total-ozone climatology stored as 5 Fourier
 *  coefficients per 1°×1° cell. The data ships as a single grib2 file
 *  (`O3clim.grib`) with five messages — one per coefficient B0..B4 — keyed by
 *  `level` 0..4. The plugin bilinear-interpolates the coefficients onto the
 *  target grid, reconstructs the climatological ozone at the forecast Julian
 *  day, and returns the percentage deviation of the forecast field from that
 *  climatology. The grib was produced one-time from the legacy
 *  `O3clim.ascii` by `o3clim_to_grib.py` in the test directory.
 *
 *  Source-of-truth: the plugin is bit-for-bit faithful to the legacy reference
 *  outputs. Where there is a choice between physics improvement and bit-exact
 *  reproduction, we keep the legacy formulation (and note it in the function
 *  comments). The regression tests live under `himan-tests/functional-tests/uv_index/`.
 *
 *  Code map (top to bottom of this file)
 *  -------------------------------------
 *  1. Constants for the disort table axes (kN…, kD…, kTableSize).
 *  2. Free-function helpers in an anonymous namespace:
 *         TableIndex         — flatten 7-D indices into 1-D offset
 *         IntRange           — pick Lagrange stencil for one axis
 *         LagrangeCoeffs     — build per-stencil Lagrange weights
 *         FixValue           — Earth-Sun distance correction
 *         SolarNoonZenithAngle — solar-noon SZA from JD and latitude
 *         LoadAerosolGrib    — read a single-message grib via himan's grib plugin
 *  3. `disort_table` class: loads and interpolates the 7-D UV-irradiance table.
 *  4. `o3_climatology` class: loads `O3clim.grib` (5 messages, level 0..4) and
 *     evaluates clim at (lat,lon,jd).
 *  5. The `uv_index` plugin proper:
 *         Process()   — one-time per run on the main thread; reads options, loads
 *                       static data, declares output params, hands off to Start().
 *         Calculate() — runs on worker threads, once per (time, level, ftype) slot;
 *                       fetches forecast inputs, walks the target grid, writes
 *                       the output param vector(s).
 *
 *  ──────────────────────────────────────────────────────────────────────────────
 */

#include "uv_index.h"
#include "forecast_time.h"
#include "grib.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "plugin_factory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace himan;
using namespace himan::plugin;

namespace
{
const string itsName("uv_index");

// Disort table axis sizes and step sizes — fixed by disort_24012003.dat layout.
constexpr int kNAlbedo = 4;
constexpr int kNOzone = 7;
constexpr int kNAtau = 8;
constexpr int kNAssa = 3;
constexpr int kNCtau = 9;
constexpr int kNSza = 7;
constexpr int kNAlt = 4;
constexpr double kDAlbedo = 0.3;
constexpr double kDOzone = 80.0;
constexpr double kDAtau = 0.2;
constexpr double kDAssa = 0.2;
constexpr double kDCtau = 15.0;
constexpr double kDSza = 15.0;
constexpr double kDAlt = 1000.0;

constexpr size_t kTableSize = static_cast<size_t>(kNAlbedo) * kNOzone * kNAtau * kNAssa * kNCtau * kNSza * kNAlt;

// ──────────────────────  Physical / unit-conversion constants  ──────────────────────

// ECMWF total ozone (TOZONE-KGM2) is in kg/m². 1 Dobson Unit ≡ 2.1414·10⁻⁵
// kg/m², so 1 / 2.1414·10⁻⁵ ≈ 46696.24 converts kg/m² to Dobson units.
constexpr double kOzoneKgm2ToDobson = 46696.240;

// ECMWF SD-TM2 (m of water equivalent) × 100000 → legacy
// `WaterEquivalentOfSnow` units, against which `kSnowAlbedoThreshold` below
// was calibrated. The 1e5 factor itself comes from the legacy
// `OtsoniLumiHakuEcmwf.par` chain — see the project notes for the history.
constexpr double kSnowTm2ToLegacy = 100000.0;

// Days per year used in every Julian-day Fourier expansion in this plugin
// (Earth–Sun distance, solar declination, ozone climatology). 365.0 matches
// the legacy code; the resulting ≤ 0.25 % per-leap-year drift is below the
// disort table's own discretisation noise.
constexpr double kDaysPerYear = 365.0;

// Julian day used as the reference distance in `FixValue` (E0(jd) / E0(jdRef)).
// 185 ≈ early-July aphelion baseline picked by the legacy `uv_tools::fix_value`.
constexpr int kE0RefJulianDay = 185;

// ──────────────────────  Albedo / snow decision  ──────────────────────────────

// Snow water-equivalent threshold (in `kSnowTm2ToLegacy` units) above which
// the surface is treated as snow-covered.
constexpr double kSnowAlbedoThreshold = 5.0;

// Surface albedo presented to the disort table.
constexpr double kAlbedoSnow = 0.4;
constexpr double kAlbedoBare = 0.03;

// ──────────────────────  Disort-table input offsets  ──────────────────────────

// Cloud optical depth presented to the table. Always 0 here: this plugin
// produces clear-sky UV only; cloud effects are layered on downstream.
constexpr double kClearSkyCtau = 0.0;

// The legacy C++ wrapper subtracts these offsets from ozone (Dobson) and
// single-scattering albedo before calling `cie_interp`. The table was
// generated around the offset values, so removing them would shift the
// lookups to the mostly-empty corners of the axes.
constexpr double kOzoneTableOffset = 100.0;
constexpr double kAssaTableOffset = 0.6;

// Maps the dimensionless disort output to the WHO UV-index scale.
constexpr double kUvIndexScale = 40.0;

// Ratio → percent for the O3ANOM-PRCNT output.
constexpr double kPercent = 100.0;

// ──────────────────────  Input-range guards  ──────────────────────────────────
// Mirror the legacy `call_rutine` guards. Any forecast point outside these
// brackets gets a MissingValue output.
constexpr double kAlbedoMin = 0.0;
constexpr double kAlbedoMax = 1.0;
constexpr double kOzoneMinDU = 10.0;
constexpr double kOzoneMaxDU = 800.0;
constexpr double kAtauMin = 0.0;
constexpr double kAtauMax = 100.0;
constexpr double kAssaMin = 0.0;
constexpr double kAssaMax = 1.0;
constexpr double kSzaMin = 0.0;
constexpr double kSzaMax = 180.0;  // physical max is 90; legacy was lenient
constexpr double kAltMaxM = 10000.0;

// Flatten the seven (iAlt, iSza, iCtau, iAssa, iAtau, iAlbedo, iOzone) indices
// into the 1-D storage offset used by `disort_table::itsCieRates`.
//
// Why this exact ordering? It mirrors the column-major memory layout of the
// Fortran array `table(n_alt, n_sza, n_ctau, n_assa, n_atau, n_albedo, n_ozone)`
// in cieInterp.for, with `iAlt` varying fastest. Keeping the same layout means
// the file-read loop in `disort_table::Load` can just walk the file in order
// and drop each value into the right slot, and the interpolation sum hits
// memory in a cache-friendly stride for the innermost altitude axis.
inline size_t TableIndex(int iAlt, int iSza, int iCtau, int iAssa, int iAtau, int iAlbedo, int iOzone)
{
	return (
	    (((((static_cast<size_t>(iOzone) * kNAlbedo + iAlbedo) * kNAtau + iAtau) * kNAssa + iAssa) * kNCtau + iCtau) *
	         kNSza +
	     iSza) *
	        kNAlt +
	    iAlt);
}

// Lagrange interpolation stencil along one axis, precomputed once per axis
// and reused for each (val, axis[first..first+pdeg]) sample. `first` is the
// starting index, `pdeg` the polynomial degree (so `pdeg + 1` stencil points,
// capped at 4 — `cieInterp.for` does the same). `denom[i]` is the precomputed
// Lagrange denominator Π_{k ≠ i} (axis[first+i] − axis[first+k]).
struct lagrange_stencil
{
	int first = 0;
	int pdeg = 0;
	std::array<double, 4> denom{};
};

// Pick a Lagrange interpolation stencil for `val` along the 1-D axis `axis`.
//
// The returned stencil covers `axis[first .. first + pdeg]` with `pdeg ≤ 3`.
// When `val` falls outside the axis range, `pdeg` is reduced to 1 and `first`
// shifted to the nearest edge — matching the legacy linear-extrapolation
// behaviour of `cieInterp.for::int_range`. Splitting `denom` precomputation
// from the per-query Lagrange coefficient evaluation in `LagrangeCoeffs`
// saves the (axis-only) products there.
//
// Templated on the axis length `N` so the caller passes a `std::array`
// directly (no raw pointer / explicit length pair).
template <size_t N>
lagrange_stencil IntRange(double val, const std::array<double, N>& axis)
{
	const int lastel = static_cast<int>(N) - 1;
	int inrange = 0;

	int f = 0;
	while (f < lastel && (axis[f] - val) < 0.0)
	{
		f++;
	}
	if ((axis[f] - val) < 0.0)
	{
		inrange = 1;
	}

	int p;
	if (f >= 2)
	{
		p = 2 - inrange;
	}
	else
	{
		p = f;
		if (f == 0)
		{
			inrange = -1;
		}
	}
	f = f - p;
	if (f + p != lastel)
	{
		p = p + 1;
	}

	lagrange_stencil s{};
	s.first = f;
	s.pdeg = p;

	for (int i = 0; i <= p; ++i)
	{
		s.denom[i] = 1.0;
		for (int k = 0; k <= p; ++k)
		{
			if (k != i)
			{
				s.denom[i] *= (axis[f + i] - axis[f + k]);
			}
		}
	}
	return s;
}

// Build the Lagrange coefficients for `val` against the stencil chosen by
// `IntRange`. Output p[i] satisfies
//   Σ_i p[i] · f(axis[first+i]) ≈ f(val)
// for any function f. Computed as
//   p[i] = (1 / denom[i]) · Π_{k ≠ i} (val − axis[first+k])
// — `denom` was precomputed once by `IntRange` so this loop avoids redoing
// the same axis-only products on every query.
template <size_t N>
std::array<double, 4> LagrangeCoeffs(double val, const std::array<double, N>& axis, const lagrange_stencil& s)
{
	std::array<double, 4> diff{};
	for (int i = 0; i <= s.pdeg; ++i)
	{
		diff[i] = val - axis[s.first + i];
	}
	std::array<double, 4> p{};
	for (int i = 0; i <= s.pdeg; ++i)
	{
		p[i] = 1.0 / s.denom[i];
		for (int k = 0; k <= s.pdeg; ++k)
		{
			if (k != i)
			{
				p[i] *= diff[k];
			}
		}
	}
	return p;
}

// Earth–Sun distance correction E0(jd) / E0(jd_ref), normalised to Julian day
// 185 (≈ 4-July aphelion reference used by the legacy `uv_tools::fix_value`).
//
// Why: solar irradiance at the top of the atmosphere varies by ~7 % over the
// year as the Earth–Sun distance changes. The DISORT look-up table in
// `disort_24012003.dat` was computed at the reference distance, so the UV
// index must be scaled by this ratio. Keeping the same JD-185 normalisation
// reproduces the legacy bit-for-bit.
double FixValue(int julianDay)
{
	double angle = 2.0 * M_PI * (kE0RefJulianDay - 1.0) / kDaysPerYear;
	const double e0_basic = 1.000110 + 0.034221 * cos(angle) + 0.001280 * sin(angle) + 0.000719 * cos(2 * angle) +
	                        0.000077 * sin(2 * angle);

	angle = 2.0 * M_PI * (julianDay - 1.0) / kDaysPerYear;
	const double e0 = 1.000110 + 0.034221 * cos(angle) + 0.001280 * sin(angle) + 0.000719 * cos(2 * angle) +
	                  0.000077 * sin(2 * angle);

	return e0 / e0_basic;
}

// Solar zenith angle (degrees) at local solar noon for the given latitude and
// Julian day. The result equals |lat − δ| with δ the solar declination from a
// 5-term Fourier expansion (Spencer 1971 form, same coefficients as the
// legacy `uv_tools::max_angle`).
//
// Why a custom helper instead of `metutil::ElevationAngle_`?
//   1. We want solar-noon (hour angle H = 0) explicitly, not the angle at an
//      arbitrary wall-clock time — daily-max UV index is by construction the
//      value at solar noon.
//   2. The legacy reference outputs were generated with this exact 5-term
//      declination + cos(H) = 1 formulation; matching it keeps the regression
//      bit-identical.
// `metutil::ElevationAngle_` uses a 7-term declination plus an
// equation-of-time correction — slightly more accurate in absolute terms but
// would force us to regenerate the reference data.
double SolarNoonZenithAngle(double latDeg, int julianDay)
{
	const double kappa = 2.0 * M_PI * (julianDay - 1.0) / kDaysPerYear;
	const double delta = 0.006918 - 0.399912 * cos(kappa) + 0.070257 * sin(kappa) - 0.006758 * cos(2 * kappa) +
	                     0.000907 * sin(2 * kappa);
	const double latRad = latDeg * constants::kDeg;
	const double argum = sin(latRad) * sin(delta) + cos(latRad) * cos(delta);  // H=0 → cos(H)=1
	const double clampedArgum = max(-1.0, min(1.0, argum));
	const double newargum = sqrt(1.0 - clampedArgum * clampedArgum);
	return atan2(newargum, clampedArgum) * constants::kRad;
}

// Load a single-message grib file directly through the himan `grib` plugin
// and return the resulting `info<double>`.
//
// Used for the seasonal aerosol climatologies (atau / assa, summer / winter,
// per hemisphere). Those gribs all carry placeholder
// paramId / discipline / category / number = 0 (see `aerosol-grib.conf` in
// the test directory) so the standard radon-driven Fetch() path cannot
// distinguish them by name. We instead identify each file by its path and
// pass `forceCaching = true` so the grib plugin returns the only message in
// the file regardless of the (empty) `search_options` we hand it.
//
// Why route through `grib::FromFile` rather than reading via raw eccodes or
// newbase? It keeps the plugin newbase-free (the old version used
// `NFmiQueryData` / `NFmiFastQueryInfo` to load `.fqd` files; we converted
// them to grib2 in the test directory) and delivers the data as a himan-
// native `info<double>`, so the hot loop in `Calculate()` can use plain
// `VEC(info)[i]` indexing.
shared_ptr<info<double>> LoadAerosolGrib(const string& path, shared_ptr<const plugin_configuration> conf)
{
	file_information fi;
	fi.file_location = path;
	fi.file_type = kGRIB;
	fi.storage_type = HPFileStorageType::kLocalFileSystem;

	search_options opts(forecast_time(raw_time(), raw_time()), param(), level(), producer(),
	                    forecast_type(kDeterministic), conf);

	auto gribPlugin = GET_PLUGIN(grib);
	auto infos = gribPlugin->FromFile<double>(fi, opts, false, true);

	if (infos.empty() || !infos.front())
	{
		throw runtime_error(fmt::format("uv_index: failed to read aerosol grib {}", path));
	}
	return infos.front();
}

}  // namespace

// Read the disort look-up table from `path` (typically `disort_24012003.dat`,
// a 1.6 MB ASCII file) into `itsCieRates`.
//
// The file lists 4 doubles per line — the altitude axis at fixed
// (sza, ctau, assa, atau, albedo, ozone). The seven nested loops below walk
// the same axis order the Fortran writer used, so reading the file
// sequentially fills `itsCieRates` at exactly the offsets `TableIndex` later
// expects in `Interpolate`.
//
// Throws if the file cannot be opened or runs out of values before the loop
// completes (catches truncated files / wrong axis sizes early).
void disort_table::Load(const string& path)
{
	itsCieRates.resize(kTableSize);

	ifstream in(path);
	if (!in)
	{
		throw runtime_error(fmt::format("uv_index: cannot open disort table {}", path));
	}

	// The file's value order (slowest axis first: ozone, albedo, atau, assa,
	// ctau, sza, alt) matches the linear-index order of `TableIndex` by
	// construction — see the comment on `TableIndex` above.
	for (size_t i = 0; i < kTableSize; ++i)
	{
		if (!(in >> itsCieRates[i]))
		{
			throw runtime_error(fmt::format("uv_index: short read on {}", path));
		}
	}
}

// Evaluate the 7-D clear-sky UV irradiance look-up table at the supplied
// inputs via tensor-product Lagrange interpolation (degree ≤ 3 per axis).
//
// Inputs and units (must match the legacy `cieInterp.for` calling convention):
//   albedo  : surface albedo, dimensionless, 0..1
//   ozone   : total ozone in Dobson units MINUS 100 (the legacy C++ wrapper
//             applies the −100 offset before calling Fortran; we keep that)
//   atau    : aerosol optical depth, dimensionless
//   assa    : single-scattering albedo MINUS 0.6 (likewise legacy convention)
//   ctau    : cloud optical depth — hard-coded to 0 by the caller (clear sky)
//   sza     : solar zenith angle in degrees, 0..90
//   alt     : surface altitude in metres, 0..10000
// Output is the table value in DISORT's internal units; the caller multiplies
// by the Earth–Sun correction (`FixValue`) and 40 to produce the UV index.
//
// Why tensor-product Lagrange rather than e.g. trilinear? Because the legacy
// reference outputs were generated this way; this routine is a direct port of
// the Fortran `cie_interp` so the regression test can compare bit-for-bit.
double disort_table::Interpolate(double albedo, double ozone, double atau, double assa, double ctau, double sza,
                                 double alt) const
{
	std::array<double, kNAlbedo> albedoAxis;
	std::array<double, kNOzone> ozoneAxis;
	std::array<double, kNAtau> atauAxis;
	std::array<double, kNAssa> assaAxis;
	std::array<double, kNCtau> ctauAxis;
	std::array<double, kNSza> szaAxis;
	std::array<double, kNAlt> altAxis;

	for (int i = 0; i < kNAlbedo; ++i)
	{
		albedoAxis[i] = i * kDAlbedo;
	}
	for (int i = 0; i < kNOzone; ++i)
	{
		ozoneAxis[i] = i * kDOzone;
	}
	for (int i = 0; i < kNAtau; ++i)
	{
		atauAxis[i] = i * kDAtau;
	}
	for (int i = 0; i < kNAssa; ++i)
	{
		assaAxis[i] = i * kDAssa;
	}
	for (int i = 0; i < kNCtau; ++i)
	{
		ctauAxis[i] = i * kDCtau;
	}
	for (int i = 0; i < kNSza; ++i)
	{
		szaAxis[i] = i * kDSza;
	}
	for (int i = 0; i < kNAlt; ++i)
	{
		altAxis[i] = i * kDAlt;
	}

	const auto sAlbedo = IntRange(albedo, albedoAxis);
	const auto sOzone = IntRange(ozone, ozoneAxis);
	const auto sAtau = IntRange(atau, atauAxis);
	const auto sAssa = IntRange(assa, assaAxis);
	const auto sCtau = IntRange(ctau, ctauAxis);
	const auto sSza = IntRange(sza, szaAxis);
	const auto sAlt = IntRange(alt, altAxis);

	const auto pAlbedo = LagrangeCoeffs(albedo, albedoAxis, sAlbedo);
	const auto pOzone = LagrangeCoeffs(ozone, ozoneAxis, sOzone);
	const auto pAtau = LagrangeCoeffs(atau, atauAxis, sAtau);
	const auto pAssa = LagrangeCoeffs(assa, assaAxis, sAssa);
	const auto pCtau = LagrangeCoeffs(ctau, ctauAxis, sCtau);
	const auto pSza = LagrangeCoeffs(sza, szaAxis, sSza);
	const auto pAlt = LagrangeCoeffs(alt, altAxis, sAlt);

	double sum = 0.0;
	for (int iOzone = 0; iOzone <= sOzone.pdeg; ++iOzone)
	{
		for (int iAlbedo = 0; iAlbedo <= sAlbedo.pdeg; ++iAlbedo)
		{
			for (int iAtau = 0; iAtau <= sAtau.pdeg; ++iAtau)
			{
				for (int iAssa = 0; iAssa <= sAssa.pdeg; ++iAssa)
				{
					for (int iCtau = 0; iCtau <= sCtau.pdeg; ++iCtau)
					{
						for (int iSza = 0; iSza <= sSza.pdeg; ++iSza)
						{
							for (int iAlt = 0; iAlt <= sAlt.pdeg; ++iAlt)
							{
								sum += pAlbedo[iAlbedo] * pOzone[iOzone] * pAtau[iAtau] * pAssa[iAssa] * pCtau[iCtau] *
								       pSza[iSza] * pAlt[iAlt] *
								       itsCieRates[TableIndex(iAlt + sAlt.first, iSza + sSza.first, iCtau + sCtau.first,
								                              iAssa + sAssa.first, iAtau + sAtau.first,
								                              iAlbedo + sAlbedo.first, iOzone + sOzone.first)];
							}
						}
					}
				}
			}
		}
	}

	return sum;
}

// Load the TOMS total-ozone climatology from a 5-message grib file (one
// message per Fourier coefficient B0..B4, distinguished by `level` 0..4).
//
// The conversion script `o3clim_to_grib.py` in the test directory turns the
// legacy `O3clim.ascii` text format into this grib. The grib has no real
// meteorological paramId — it carries a placeholder so the himan grib plugin
// will accept it; we identify the file by path and the messages by level.
//
// On success, populates `itsGrid` with 5 coefficients per (lat, lon) cell
// in row-major order (lat outer, lon inner), plus the grid metadata fields
// used by `Evaluate` for bilinear lookups.
//
// Throws on file open / read failure, on a wrong message count, or if any
// level outside 0..4 turns up.
void o3_climatology::Load(const string& path, shared_ptr<const plugin_configuration> conf)
{
	file_information fi;
	fi.file_location = path;
	fi.file_type = kGRIB;
	fi.storage_type = HPFileStorageType::kLocalFileSystem;

	search_options opts(forecast_time(raw_time(), raw_time()), param(), level(), producer(),
	                    forecast_type(kDeterministic), conf);

	auto gribPlugin = GET_PLUGIN(grib);
	auto infos = gribPlugin->FromFile<double>(fi, opts, /*readPackedData=*/false, /*forceCaching=*/true);

	if (infos.size() != 5)
	{
		throw runtime_error(
		    fmt::format("uv_index: expected 5 messages in O3 climatology grib {}, got {}", path, infos.size()));
	}

	// Sort messages by level so we don't depend on the file's message order
	// (the conversion script writes them 0..4, but be defensive).
	std::sort(infos.begin(), infos.end(),
	          [](const shared_ptr<info<double>>& a, const shared_ptr<info<double>>& b)
	          {
		          a->template First<himan::level>();
		          b->template First<himan::level>();
		          return a->Level().Value() < b->Level().Value();
	          });

	for (int k = 0; k < 5; ++k)
	{
		infos[k]->template First<himan::level>();
		const double lv = infos[k]->Level().Value();
		if (static_cast<int>(lv) != k)
		{
			throw runtime_error(fmt::format("uv_index: O3 climatology grib {} missing level {} (got {})", path, k, lv));
		}
	}

	// All 5 messages share the same grid; pull metadata from B0.
	infos[0]->template First<forecast_time>();
	infos[0]->template First<himan::level>();
	infos[0]->template First<param>();
	const auto& g = dynamic_cast<const himan::latitude_longitude_grid&>(*infos[0]->Grid());
	itsNLon = static_cast<int>(g.Ni());
	itsNLat = static_cast<int>(g.Nj());
	itsLonStart = g.FirstPoint().X();
	itsLatStart = g.FirstPoint().Y();
	itsLonStep = g.Di();
	itsLatStep = g.Dj();

	// Re-pack the 5 fields into the (B0..B4)-per-cell layout used by Evaluate.
	itsGrid.assign(static_cast<size_t>(itsNLat) * static_cast<size_t>(itsNLon), coeffs{});
	for (int k = 0; k < 5; ++k)
	{
		infos[k]->template First<forecast_time>();
		infos[k]->template First<himan::level>();
		infos[k]->template First<param>();
		const auto& vec = VEC(infos[k]);
		if (vec.size() != itsGrid.size())
		{
			throw runtime_error(fmt::format("uv_index: O3 climatology grib {} level {} has {} cells, expected {}", path,
			                                k, vec.size(), itsGrid.size()));
		}
		for (size_t i = 0; i < vec.size(); ++i)
		{
			itsGrid[i].B[k] = vec[i];
		}
	}
}

// Climatological total ozone in Dobson units at the requested (lat, lon) on
// Julian day `julianDay`.
//
// Two-step process:
//   1. Bilinear-interpolate the 5 Fourier coefficients from the 4 climatology
//      cells nearest (lat, lon).
//   2. Reconstruct the climatology value at this Julian day from the
//      interpolated coefficients:
//        clim = B0 + B1·cos θ + B2·sin θ + B3·cos 2θ + B4·sin 2θ,
//        θ = 2π · jd / 365
//
// Longitude is wrapped into the grid's native range, so callers can pass
// longitudes in any convention. The latitude is clamped to the grid edges
// (the climatology covers -89.5 .. +89.5, callers may query poles).
//
// Why pre-interp the coefficients and reconstruct, rather than reconstructing
// each cell's clim value first and then bilinear-interpolating those? Same
// result for a linear-in-coefficient model, plus this matches the legacy
// `otsoniano.cpp` flow so we stay reference-compatible.
double o3_climatology::Evaluate(double latDeg, double lonDeg, int julianDay) const
{
	// Wrap longitude to the grid's range.
	const double lonSpan = itsLonStep * itsNLon;
	double lon = lonDeg;
	while (lon < itsLonStart)
	{
		lon += lonSpan;
	}
	while (lon >= itsLonStart + lonSpan)
	{
		lon -= lonSpan;
	}

	const double latIdx = (latDeg - itsLatStart) / itsLatStep;
	const double lonIdx = (lon - itsLonStart) / itsLonStep;

	int i0 = static_cast<int>(floor(latIdx));
	int j0 = static_cast<int>(floor(lonIdx));
	double wi = latIdx - i0;
	double wj = lonIdx - j0;

	if (i0 < 0)
	{
		i0 = 0;
		wi = 0.0;
	}
	if (i0 >= itsNLat - 1)
	{
		i0 = itsNLat - 2;
		wi = 1.0;
	}

	const int j1 = (j0 + 1) % itsNLon;
	j0 = ((j0 % itsNLon) + itsNLon) % itsNLon;
	const int i1 = i0 + 1;

	auto cell = [&](int i, int j) -> const coeffs& { return itsGrid[static_cast<size_t>(i) * itsNLon + j]; };

	coeffs c{};
	for (int k = 0; k < 5; ++k)
	{
		const double v00 = cell(i0, j0).B[k];
		const double v10 = cell(i1, j0).B[k];
		const double v01 = cell(i0, j1).B[k];
		const double v11 = cell(i1, j1).B[k];
		c.B[k] = (1 - wi) * (1 - wj) * v00 + wi * (1 - wj) * v10 + (1 - wi) * wj * v01 + wi * wj * v11;
	}

	const double theta = 2.0 * M_PI * julianDay / kDaysPerYear;
	return c.B[0] + c.B[1] * cos(theta) + c.B[2] * sin(theta) + c.B[3] * cos(2 * theta) + c.B[4] * sin(2 * theta);
}

uv_index::uv_index()
{
	itsLogger = logger(itsName);
}

// Plugin entry point. Called once per plugin invocation from himan-bin.
//
// Reads the JSON `mode` option to decide which output parameter(s) and which
// static data files to load:
//   mode = "uv" (default)  → BOTH UVIMAX-N (daily-max, solar-noon SZA) and
//                            UVI-N (instantaneous, SZA at valid time) in one
//                            pass; loads disort table + 4 aerosol gribs.
//   mode = "anomaly"       → O3ANOM-PRCNT (forecast vs. TOMS climatology);
//                            loads only the total-ozone climatology GRIB
//                            file (`o3_climatology`).
//
// All static data is loaded here (Process runs once on the main thread)
// rather than in Calculate (which runs per worker thread), so the heavy I/O
// and parsing happens once. The loaded data is then shared read-only across
// all Calculate threads.
//
// Finally hands off to `Start()` which spawns the worker threads that call
// `Calculate` for each (time, level, forecast_type) combination.
void uv_index::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	const string mode = itsConfiguration->Exists("mode") ? itsConfiguration->GetValue("mode") : "uv";

	const auto requireOpt = [this](const char* key)
	{
		if (!itsConfiguration->Exists(key))
		{
			throw runtime_error(fmt::format("uv_index: missing required option '{}'", key));
		}
		return itsConfiguration->GetValue(key);
	};

	if (mode == "uvimax")
	{
		itsMode = mode_t::kUvimax;
		SetParams({param("UVIMAX-N")});
	}
	else if (mode == "uvi")
	{
		itsMode = mode_t::kUvi;
		SetParams({param("UVI-N")});
	}
	else if (mode == "uv")
	{
		itsMode = mode_t::kUv;
		// UVIMAX-N first, UVI-N second — Calculate() relies on this ordering
		// when navigating the target info via Find<param>().
		SetParams({param("UVIMAX-N"), param("UVI-N")});
	}
	else if (mode == "o3anom")
	{
		itsMode = mode_t::kO3anom;
		SetParams({param("O3ANOM-PRCNT")});
	}
	else
	{
		throw runtime_error(
		    fmt::format("uv_index: unknown mode '{}' (expected 'uvimax', 'uvi', 'uv' or 'o3anom')", mode));
	}

	if (itsConfiguration->Exists("uvimax_valid_hour"))
	{
		itsUvimaxValidHour = stoi(itsConfiguration->GetValue("uvimax_valid_hour"));
		if (itsUvimaxValidHour < 0 || itsUvimaxValidHour > 23)
		{
			throw runtime_error(fmt::format("uv_index: 'uvimax_valid_hour' must be 0..23, got {}", itsUvimaxValidHour));
		}
	}

	if (itsMode == mode_t::kO3anom)
	{
		itsO3Clim.Load(requireOpt("o3_climatology"), itsConfiguration);
		itsLogger.Info("Loaded total-ozone climatology");
	}
	else
	{
		itsDisortTable.Load(requireOpt("disort_table"));
		itsAtauSummer = LoadAerosolGrib(requireOpt("atau_summer"), itsConfiguration);
		itsAtauWinter = LoadAerosolGrib(requireOpt("atau_winter"), itsConfiguration);
		itsAssaSummer = LoadAerosolGrib(requireOpt("assa_summer"), itsConfiguration);
		itsAssaWinter = LoadAerosolGrib(requireOpt("assa_winter"), itsConfiguration);
		itsLogger.Info("Loaded disort table and seasonal aerosol climatology");
	}

	Start();
}

// Anomaly mode: fetch total ozone, compute (forecast − climatology) /
// climatology × 100 per grid point and write the result to O3ANOM-PRCNT.
//
// UV mode: fetch total ozone, snow water-equivalent and surface geopotential
// once. For each grid point build the shared 7-element input vector
// (albedo, ozone − 100, atau, assa − 0.6, ctau = 0 for clear-sky, sza, alt)
// and call the disort table twice: once with the solar-noon SZA from
// `SolarNoonZenithAngle` to write UVIMAX-N, and once with the SZA at valid
// time from `metutil::ElevationAngle_` to write UVI-N. Computing both in a
// single pass amortises the input fetches and the per-point arithmetic that
// is independent of SZA (albedo, alt, scaling); only the table lookup runs
// twice.
void uv_index::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger(fmt::format("{}Thread #{}", itsName, threadIndex));

	const forecast_time forecastTime = myTargetInfo->Time();
	const forecast_type forecastType = myTargetInfo->ForecastType();
	const level forecastLevel = myTargetInfo->Level();

	myThreadedLogger.Info(fmt::format("Calculating time {} level {}", static_cast<string>(forecastTime.ValidDateTime()),
	                                  static_cast<string>(forecastLevel)));

	const level surface(kHeight, 0, "HEIGHT");

	auto ozoneInfo = Fetch(forecastTime, surface, param("TOZONE-KGM2"), forecastType, false);

	if (itsMode == mode_t::kO3anom)
	{
		if (!ozoneInfo)
		{
			myThreadedLogger.Warning(
			    fmt::format("Skipping step {}: TOZONE-KGM2 missing", static_cast<string>(forecastTime.Step())));
			return;
		}

		const int jd = stoi(forecastTime.ValidDateTime().String("%j"));
		const auto& ozoneVec = VEC(ozoneInfo);
		auto& outVec = VEC(myTargetInfo);

		for (size_t i = 0; i < outVec.size(); ++i)
		{
			const double rawOzone = ozoneVec[i];
			if (IsMissing(rawOzone))
			{
				outVec[i] = MissingDouble();
				continue;
			}
			const double forecastDU = rawOzone * kOzoneKgm2ToDobson;
			const point ll = myTargetInfo->Grid()->LatLon(i);
			const double climDU = itsO3Clim.Evaluate(ll.Y(), ll.X(), jd);
			if (climDU == 0.0)
			{
				outVec[i] = MissingDouble();
				continue;
			}
			outVec[i] = (forecastDU - climDU) / climDU * kPercent;
		}

		myThreadedLogger.Info(fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(),
		                                  myTargetInfo->Data().Size()));
		return;
	}

	auto snowInfo = Fetch(forecastTime, surface, param("SD-TM2"), forecastType, false);

	// Topography is time-invariant; fetch at analysis time (step 0).
	const forecast_time analysisTime(forecastTime.OriginDateTime(), forecastTime.OriginDateTime());
	auto topoInfo = Fetch(analysisTime, surface, param("Z-M2S2"), forecastType, false);

	if (!ozoneInfo || !snowInfo || !topoInfo)
	{
		myThreadedLogger.Warning(
		    fmt::format("Skipping step {}: missing real-time input", static_cast<string>(forecastTime.Step())));
		return;
	}

	// Pick seasonal aerosol files by valid-time month (Apr–Sep → summer).
	// The four aerosol gribs were pre-interpolated to the per-hemisphere
	// target geometries when the legacy fqd files were generated, so we
	// access them directly by index (no re-interpolation needed here).
	const int month = stoi(forecastTime.ValidDateTime().String("%m"));
	const bool isSummer = (month >= 4 && month <= 9);
	auto atauInfo = isSummer ? itsAtauSummer : itsAtauWinter;
	auto assaInfo = isSummer ? itsAssaSummer : itsAssaWinter;
	atauInfo->First<forecast_time>();
	atauInfo->First<level>();
	atauInfo->First<param>();
	assaInfo->First<forecast_time>();
	assaInfo->First<level>();
	assaInfo->First<param>();
	const auto& atauVec = VEC(atauInfo);
	const auto& assaVec = VEC(assaInfo);

	const int julianDay = stoi(forecastTime.ValidDateTime().String("%j"));
	const double e0 = FixValue(julianDay);

	const auto& ozoneVec = VEC(ozoneInfo);
	const auto& snowVec = VEC(snowInfo);
	const auto& topoVec = VEC(topoInfo);

	bool produceMax = (itsMode == mode_t::kUvimax || itsMode == mode_t::kUv);
	const bool produceInst = (itsMode == mode_t::kUvi || itsMode == mode_t::kUv);

	// Optional filter: only write UVIMAX-N when the forecast valid time's UTC
	// hour matches `uvimax_valid_hour`. Lets a single config drive both the
	// 00 and 12 UTC forecast cycles producing UVI hourly and UVIMAX-N only at
	// 12 UTC valid time. UVI-N is unaffected.
	if (produceMax && itsUvimaxValidHour >= 0)
	{
		const int validHour = stoi(forecastTime.ValidDateTime().String("%H"));
		if (validHour != itsUvimaxValidHour)
		{
			produceMax = false;
		}
	}

	// Bind one value vector per requested output param. Modes that don't
	// produce a given param leave the corresponding pointer null and skip
	// the per-point write.
	std::vector<double>* uvMaxVec = nullptr;
	std::vector<double>* uvInstVec = nullptr;
	if (produceMax)
	{
		myTargetInfo->Find<param>(param("UVIMAX-N"));
		uvMaxVec = &VEC(myTargetInfo);
	}
	if (produceInst)
	{
		myTargetInfo->Find<param>(param("UVI-N"));
		uvInstVec = &VEC(myTargetInfo);
	}

	const size_t n = produceMax ? uvMaxVec->size() : uvInstVec->size();

	// Local helper: run the disort lookup for a given SZA and produce the
	// final UV-index value, or MissingDouble() if any guard fails.
	const auto uvIndexAt = [&](double albedo, double ozone, double atau, double assa, double sza, double alt) -> double
	{
		if (albedo < kAlbedoMin || albedo > kAlbedoMax || ozone < kOzoneMinDU || ozone > kOzoneMaxDU ||
		    atau < kAtauMin || atau > kAtauMax || assa < kAssaMin || assa > kAssaMax || sza < kSzaMin ||
		    sza > kSzaMax || alt > kAltMaxM)
		{
			return MissingDouble();
		}
		const double cieRate = itsDisortTable.Interpolate(albedo, ozone - kOzoneTableOffset, atau,
		                                                  assa - kAssaTableOffset, kClearSkyCtau, sza, alt);
		return cieRate * e0 * kUvIndexScale;
	};

	for (size_t i = 0; i < n; ++i)
	{
		const double ozoneRaw = ozoneVec[i];  // kg/m²
		const double snowRaw = snowVec[i];    // m of water equivalent
		const double geop = topoVec[i];       // m²/s²
		const double atau = atauVec[i];
		const double assa = assaVec[i];

		if (IsMissing(ozoneRaw) || IsMissing(snowRaw) || IsMissing(geop) || IsMissing(atau) || IsMissing(assa))
		{
			if (produceMax)
			{
				(*uvMaxVec)[i] = MissingDouble();
			}
			if (produceInst)
			{
				(*uvInstVec)[i] = MissingDouble();
			}
			continue;
		}

		const double ozone = ozoneRaw * kOzoneKgm2ToDobson;
		const double snowWE = snowRaw * kSnowTm2ToLegacy;
		const double albedo = (snowWE > kSnowAlbedoThreshold) ? kAlbedoSnow : kAlbedoBare;
		double alt = geop * constants::kIg;
		if (alt < 0.0)
		{
			alt = 0.0;
		}

		const point latlon = myTargetInfo->Grid()->LatLon(i);

		if (produceMax)
		{
			// UVIMAX-N: solar zenith at local solar noon (daily maximum).
			const double szaMax = SolarNoonZenithAngle(latlon.Y(), julianDay);
			(*uvMaxVec)[i] = uvIndexAt(albedo, ozone, atau, assa, szaMax, alt);
		}

		if (produceInst)
		{
			// UVI-N: instantaneous SZA at the forecast valid time.
			double szaInst = 90.0 - metutil::ElevationAngle_(latlon, forecastTime.ValidDateTime());
			if (szaInst < kSzaMin)
			{
				szaInst = kSzaMin;
			}
			if (szaInst > 90.0)
			{
				szaInst = 90.0;
			}
			(*uvInstVec)[i] = uvIndexAt(albedo, ozone, atau, assa, szaInst, alt);
		}
	}

	myThreadedLogger.Info(
	    fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}

void uv_index::WriteToFile(const shared_ptr<info<double>> targetInfo, write_options opts)
{
	// Force the writer to drop all-missing grids. With `uvimax_valid_hour`
	// set, UVIMAX-N at non-matching hours is fully missing and we don't
	// want a placeholder grib file landing on disk / in radon. Other plugins
	// can still produce all-missing fields and expect them to be written, so
	// we only flip the default here, not framework-wide.
	opts.write_empty_grid = false;
	compiled_plugin_base::WriteToFile(targetInfo, opts);
}
