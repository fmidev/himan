#pragma once

#ifdef __CUDACC__
#define SQRT __dsqrt_rn
#else
#define SQRT sqrt
#endif

// Required source and target parameters and levels

const himan::param TParam("T-K");
const himan::param TDParam("TD-K");
const himan::param HParam("Z-M2S2");
const himan::params PParam({himan::param("P-HPA"), himan::param("P-PA")});
const himan::param SIParam("SI-N");
const himan::param LIParam("LI-N");
const himan::param BSParam("WSH-MS");
const himan::param EBSParam("EWSH-MS");
const himan::param SRHParam("HLCY-M2S2");
const himan::param TPEParam("TPE-K");
const himan::param EHIParam("EHI-N");
const himan::param BRNParam("BRN-N");
const himan::param FFParam("FF-MS");
const himan::param UParam("U-MS");
const himan::param VParam("V-MS");
const himan::param RHParam("RH-PRCNT");
const himan::param QParam("Q-KGKG");
const himan::param CAPESParam("CAPES-JKG");

const himan::level P850Level(himan::kPressure, 850);
const himan::level P700Level(himan::kPressure, 700);
const himan::level P500Level(himan::kPressure, 500);
const himan::level SixKMLevel(himan::kHeightLayer, 6000, 0);
const himan::level OneKMLevel(himan::kHeightLayer, 1000, 0);
const himan::level ThreeKMLevel(himan::kHeightLayer, 3000, 0);
const himan::level EuropeanMileLevel(himan::kHeight, 1500);
const himan::level Height0Level(himan::kHeight, 0);
const himan::level HalfKMLevel(himan::kHeightLayer, 500, 0);

namespace STABILITY
{
/**
 * See eq 1 from
 * https://www.weather.gov/media/unr/soo/scm/BKZTW00.pdf
 */

CUDA_DEVICE
inline void UVId(double u_shr, double v_shr, double u_avg, double v_avg, double& u_id, double& v_id)
{
	const double mag = SQRT(u_shr * u_shr + v_shr * v_shr);
	const double u_unit = u_shr / mag;
	const double v_unit = v_shr / mag;

	u_id = fma(v_unit, 7.5, u_avg);  // x*y+z
	v_id = fma(-u_unit, 7.5, v_avg);
}

/**
 * @brief Bulk richardson number
 *
 * CAPE needs to be at least 500 J/ms and wind shear 10 m/s
 */

CUDA_DEVICE
inline double BRN(double CAPE, double U6, double V6, double U05, double V05)
{
	const double Ud = U6 - U05;
	const double Vd = V6 - V05;
	const double m = SQRT(Ud * Ud + Vd * Vd);

	double ret = himan::MissingDouble();

	if (CAPE >= 500 && m >= 10)
	{
		ret = CAPE / (0.5 * m * m);
	}

	return ret;
}

}  // namespace STABILITY

#ifdef HAVE_CUDA
#include "info_simple.h"

namespace himan
{
namespace plugin
{
class hitool;

namespace stability_cuda
{
extern level itsBottomLevel;

struct options
{
	info_simple* si;
	info_simple* li;
	info_simple* bs01;
	info_simple* bs03;
	info_simple* bs06;
	info_simple* ebs;
	info_simple* capes;
	info_simple* srh01;
	info_simple* srh03;
	info_simple* ehi;
	info_simple* brn;
	info_simple* thetae3;

	size_t N;

	std::shared_ptr<himan::plugin::hitool> h;
	std::shared_ptr<const himan::plugin_configuration> conf;
	std::shared_ptr<himan::info> myTargetInfo;

	options()
	    : si(0),
	      li(0),
	      bs01(0),
	      bs03(0),
	      bs06(0),
	      ebs(0),
	      capes(0),
	      srh01(0),
	      srh03(0),
	      ehi(0),
	      brn(0),
	      thetae3(0),
	      N(0),
	      h(),
	      conf(),
	      myTargetInfo()
	{
	}
};

void Process(options& opts);

extern himan::level itsBottomLevel;

}  // namespace stability_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
