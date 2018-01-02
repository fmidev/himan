#include "stability.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "plugin_factory.h"
#include "stability.cuh"
#include "util.h"
#include <algorithm>  // for std::transform

#include "fetcher.h"
#include "hitool.h"
#include "radon.h"
#include "writer.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

static level itsBottomLevel;

typedef vector<double> vec;

pair<vec, vec> GetSRHSourceData(const shared_ptr<info>& myTargetInfo, shared_ptr<hitool> h);
void RunCuda(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, shared_ptr<hitool>& h);

namespace STABILITY
{
vec Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par, const vec& lowerHeight,
          const vec& upperHeight)
{
	const auto lowerValues = h->VerticalValue(par, lowerHeight);
	const auto upperValues = h->VerticalValue(par, upperHeight);

	vec ret(lowerValues.size(), himan::MissingDouble());

	for (size_t i = 0; i < lowerValues.size(); i++)
	{
		ret[i] = upperValues[i] - lowerValues[i];
	}

	return ret;
}

vec Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par, double lowerHeight, double upperHeight,
          size_t N)
{
	const vec lower(N, lowerHeight);
	const vec upper(N, upperHeight);

	return Shear(h, par, lower, upper);
}

himan::info_t Fetch(std::shared_ptr<const plugin_configuration>& conf, std::shared_ptr<himan::info>& myTargetInfo,
                    const himan::level& lev, const himan::param& par)
{
	const forecast_time forecastTime = myTargetInfo->Time();
	const forecast_type forecastType = myTargetInfo->ForecastType();
	const bool useCuda = conf->UseCudaForPacking();

	auto f = GET_PLUGIN(fetcher);

	return f->Fetch(conf, forecastTime, lev, par, forecastType, useCuda);
}
}

stability::stability()
{
	itsLogger = logger("stability");
}
void stability::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	auto r = GET_PLUGIN(radon);

	itsBottomLevel = level(kHybrid, stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer().Id(),
	                                                                      "last hybrid level number")));

#ifdef HAVE_CUDA
	stability_cuda::itsBottomLevel = itsBottomLevel;
#endif

	PrimaryDimension(kTimeDimension);

	itsInfo->LevelIterator().Clear();

	SetParams({EBSParam, KIParam, CTIParam, VTIParam, TTIParam, LIParam, SIParam}, {Height0Level});
	SetParams({BSParam}, {OneKMLevel, ThreeKMLevel, SixKMLevel});
	SetParams({SRHParam}, {OneKMLevel, ThreeKMLevel});
	SetParams({TPEParam}, {ThreeKMLevel});
	SetParams({EHIParam}, {OneKMLevel});
	SetParams({BRNParam}, {SixKMLevel});
	SetParams({FFParam}, {EuropeanMileLevel});

	Start();
}

vec CalculateStormRelativeHelicity(shared_ptr<const plugin_configuration> conf, info_t& myTargetInfo,
                                   shared_ptr<hitool> h, double stopHeight, const pair<vec, vec>& UVId)
{
	auto Uid = UVId.first;
	auto Vid = UVId.second;

	vec SRH(Uid.size(), 0);

	auto prevUInfo = STABILITY::Fetch(conf, myTargetInfo, itsBottomLevel, UParam);
	auto prevVInfo = STABILITY::Fetch(conf, myTargetInfo, itsBottomLevel, VParam);
	auto prevZInfo = STABILITY::Fetch(conf, myTargetInfo, itsBottomLevel, param("HL-M"));

	vector<bool> found(SRH.size(), false);

	level curLevel = itsBottomLevel;

	while (curLevel.Value() > 34)
	{
		curLevel.Value(curLevel.Value() - 1);

		auto UInfo = STABILITY::Fetch(conf, myTargetInfo, curLevel, UParam);
		auto VInfo = STABILITY::Fetch(conf, myTargetInfo, curLevel, VParam);
		auto ZInfo = STABILITY::Fetch(conf, myTargetInfo, curLevel, param("HL-M"));

		const auto& U = VEC(UInfo);
		const auto& V = VEC(VInfo);
		const auto& Z = VEC(ZInfo);
		const auto& prevU = VEC(prevUInfo);
		const auto& prevV = VEC(prevVInfo);
		const auto& prevZ = VEC(prevZInfo);

		for (size_t i = 0; i < SRH.size(); i++)
		{
			if (found[i])
				continue;
			const double _Uid = Uid[i];
			const double _Vid = Vid[i];

			const double _pU = prevU[i];
			const double _pV = prevV[i];

			double _U = U[i];
			double _V = V[i];

			if (Z[i] > stopHeight)
			{
				ASSERT(prevZ[i] < stopHeight);

				_U = numerical_functions::interpolation::Linear(stopHeight, prevZ[i], Z[i], _pU, _U);
				_V = numerical_functions::interpolation::Linear(stopHeight, prevZ[i], Z[i], _pV, _V);

				found[i] = true;
			}

			const double res = ((_Uid - _pU) * (_pV - _V)) - ((_Vid - _pV) * (_pU - _U));

			if (!IsMissing(res))
			{
				SRH[i] -= res;
			}
		}

		if (found.size() == static_cast<size_t>(count(found.begin(), found.end(), true)))
		{
			break;
		}

		prevUInfo = UInfo;
		prevVInfo = VInfo;
		prevZInfo = ZInfo;
	}

	for (auto& v : SRH)
	{
		if (v == 0)
			v = MissingDouble();
	}

	return SRH;
}

vec CalculateBulkRichardsonNumber(shared_ptr<const plugin_configuration> conf, info_t& myTargetInfo,
                                  shared_ptr<hitool> h)
{
	const forecast_time forecastTime = myTargetInfo->Time();
	const forecast_type forecastType = myTargetInfo->ForecastType();
	const bool useCuda = conf->UseCudaForPacking();

	auto f = GET_PLUGIN(fetcher);

	auto CAPEInfo = f->Fetch(conf, forecastTime, level(kHeightLayer, 500, 0), param("CAPE-JKG"), forecastType, useCuda);
	const auto& CAPE = VEC(CAPEInfo);

	auto U6 = h->VerticalAverage(UParam, 10, 6000);
	auto V6 = h->VerticalAverage(VParam, 10, 6000);

	auto U05 = h->VerticalAverage(UParam, 10, 500);
	auto V05 = h->VerticalAverage(VParam, 10, 500);

	vec BRN(CAPE.size(), MissingDouble());

	for (size_t i = 0; i < BRN.size(); i++)
	{
		if (CAPE[i] < 500)
		{
			continue;
		}
		const double Ud = (U6[i] - U05[i]);
		const double Vd = (V6[i] - V05[i]);
		const double m = sqrt(Ud * Ud + Vd * Vd);

		if (m < 10)
		{
			continue;
		}
		BRN[i] = CAPE[i] / (0.5 * m * m);
		ASSERT(BRN[i] < 1000);
	}

	return BRN;
}

vec CalculateEnergyHelicityIndex(shared_ptr<const plugin_configuration> conf, info_t& myTargetInfo)
{
	const forecast_time forecastTime = myTargetInfo->Time();
	const forecast_type forecastType = myTargetInfo->ForecastType();
	const bool useCuda = conf->UseCudaForPacking();

	auto f = GET_PLUGIN(fetcher);

	auto CAPEInfo = f->Fetch(conf, forecastTime, level(kHeightLayer, 500, 0), param("CAPE-JKG"), forecastType, useCuda);
	const auto& CAPE = VEC(CAPEInfo);
	const auto& SRH = VEC(myTargetInfo);

	vec EHI(CAPE.size(), MissingDouble());

	for (size_t i = 0; i < EHI.size(); i++)
	{
		EHI[i] = CAPE[i] * SRH[i] / 160000.;
	}

	return EHI;
}

void CalculateHelicityIndices(shared_ptr<const plugin_configuration> conf, info_t& myTargetInfo, shared_ptr<hitool> h)
{
	auto UVId = GetSRHSourceData(myTargetInfo, h);

	myTargetInfo->Param(SRHParam);

	myTargetInfo->Level(ThreeKMLevel);
	const auto SRH03 = CalculateStormRelativeHelicity(conf, myTargetInfo, h, 3000, UVId);
	myTargetInfo->Data().Set(SRH03);

	myTargetInfo->Level(OneKMLevel);
	const auto SRH01 = CalculateStormRelativeHelicity(conf, myTargetInfo, h, 1000, UVId);
	myTargetInfo->Data().Set(SRH01);

	const auto EHI01 = CalculateEnergyHelicityIndex(conf, myTargetInfo);
	myTargetInfo->Param(EHIParam);
	myTargetInfo->Data().Set(EHI01);

	myTargetInfo->Level(SixKMLevel);
	myTargetInfo->Param(BRNParam);
	const auto BRN = CalculateBulkRichardsonNumber(conf, myTargetInfo, h);
	myTargetInfo->Data().Set(BRN);
}

void CalculateStaticIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo)
{
	auto T850Info = STABILITY::Fetch(conf, myTargetInfo, P850Level, TParam);
	auto T700Info = STABILITY::Fetch(conf, myTargetInfo, P700Level, TParam);
	auto T500Info = STABILITY::Fetch(conf, myTargetInfo, P500Level, TParam);
	auto TD850Info = STABILITY::Fetch(conf, myTargetInfo, P850Level, TDParam);
	auto TD700Info = STABILITY::Fetch(conf, myTargetInfo, P700Level, TDParam);

	myTargetInfo->Level(Height0Level);

	myTargetInfo->Param(KIParam);
	auto& KI = VEC(myTargetInfo);

	myTargetInfo->Param(CTIParam);
	auto& CTI = VEC(myTargetInfo);

	myTargetInfo->Param(VTIParam);
	auto& VTI = VEC(myTargetInfo);

	myTargetInfo->Param(TTIParam);
	auto& TTI = VEC(myTargetInfo);

	const auto& t850 = VEC(T850Info);
	const auto& td850 = VEC(TD850Info);
	const auto& t700 = VEC(T700Info);
	const auto& td700 = VEC(TD700Info);
	const auto& t500 = VEC(T500Info);

	for (size_t i = 0; i < TTI.size(); i++)
	{
		KI[i] = STABILITY::KI(t850[i], t700[i], t500[i], td850[i], td700[i]);
		CTI[i] = STABILITY::CTI(t500[i], td850[i]);
		VTI[i] = STABILITY::VTI(t850[i], t500[i]);
		TTI[i] = STABILITY::TTI(t850[i], t500[i], td850[i]);
	}
}

tuple<vec, vec, vec, info_t, info_t, info_t> GetDynamicIndicesSourceData(shared_ptr<const plugin_configuration>& conf,
                                                                        info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	auto T500 = h->VerticalAverage(TParam, 0, 500.);
	auto P500 = h->VerticalAverage(PParam, 0., 500.);

	vec TD500;

	try
	{
		TD500 = h->VerticalAverage(param("TD-K"), 0, 500.);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			TD500 = h->VerticalAverage(RHParam, 0, 500.);

			for (auto&& tup : zip_range(TD500, T500))
			{
				double& res = tup.get<0>();
				double t = tup.get<1>();

				res = metutil::DewPointFromRH_(t, res);
			}
		}
	}

	if (P500[0] < 1500)
	{
		transform(P500.begin(), P500.end(), P500.begin(), bind1st(multiplies<double>(), 100.));  // hPa to Pa
	}

	auto f = GET_PLUGIN(fetcher);

	const forecast_type forecastType = myTargetInfo->ForecastType();
	const forecast_time forecastTime = myTargetInfo->Time();

	auto T850Info = f->Fetch(conf, forecastTime, P850Level, TParam, forecastType, conf->UseCudaForPacking());
	auto TD850Info = f->Fetch(conf, forecastTime, P850Level, TDParam, forecastType, conf->UseCudaForPacking());
	auto T500Info = f->Fetch(conf, forecastTime, P850Level, TParam, forecastType, conf->UseCudaForPacking());

	return make_tuple(T500, TD500, P500, T850Info, TD850Info, T500Info);
}

vec CalculateThetaE(shared_ptr<hitool>& h, double startHeight, double stopHeight)
{
	auto Tstop = h->VerticalValue(TParam, stopHeight);
	auto RHstop = h->VerticalValue(RHParam, stopHeight);
	auto Pstop = h->VerticalValue(PParam, stopHeight);

	auto Tstart = h->VerticalValue(TParam, startHeight);
	auto RHstart = h->VerticalValue(RHParam, startHeight);
	auto Pstart = h->VerticalValue(PParam, startHeight);

	vec ret(Tstop.size());

	for (size_t i = 0; i < Tstart.size(); i++)
	{
		const double ThetaEstart = metutil::smarttool::ThetaE_(Tstart[i], RHstart[i], Pstart[i] * 100);
		const double ThetaEstop = metutil::smarttool::ThetaE_(Tstop[i], RHstop[i], Pstop[i] * 100);

		ret[i] = ThetaEstart - ThetaEstop;
	}

	return ret;
}

void CalculateThetaEIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	vec thetae = CalculateThetaE(h, 2, 3000);

	myTargetInfo->Level(ThreeKMLevel);
	myTargetInfo->Param(TPEParam);
	myTargetInfo->Data().Set(thetae);
}

void CalculateDynamicIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	auto src = GetDynamicIndicesSourceData(conf, myTargetInfo, h);

	myTargetInfo->Param(LIParam);
	auto& LI = VEC(myTargetInfo);

	myTargetInfo->Param(SIParam);
	auto& SI = VEC(myTargetInfo);

	const auto& t500m = get<0>(src);
	const auto& td500m = get<1>(src);
	const auto& p500m = get<2>(src);
	const auto& t500 = VEC(get<3>(src));
	const auto& t850 = VEC(get<4>(src));
	const auto& td850 = VEC(get<5>(src));

	vec t_li_lifted(t500.size(), MissingDouble());
	vec t_si_lifted(t500.size(), MissingDouble());

	vec p500(t500.size(), 50000), p850(t500.size(), 85000);

	metutil::Lift(p500m.data(), t500m.data(), td500m.data(), p500.data(), t_li_lifted.data(), p500.size());
	metutil::Lift(p850.data(), t850.data(), td850.data(), p500.data(), t_si_lifted.data(), p500.size());

	for (size_t i = 0; i < t500.size(); i++)
	{
		LI[i] = t500[i] - t_li_lifted[i];
		SI[i] = t500[i] - t_si_lifted[i];
	}
}

vec CalculateBulkShear(info_t& myTargetInfo, shared_ptr<hitool>& h, double stopHeight)
{
	const auto U = STABILITY::Shear(h, param("U-MS"), 10, stopHeight, myTargetInfo->SizeLocations());
	const auto V = STABILITY::Shear(h, param("V-MS"), 10, stopHeight, myTargetInfo->SizeLocations());

	vec BS(U.size(), MissingDouble());

	for (size_t i = 0; i < U.size(); i++)
	{
		BS[i] = sqrt(U[i] * U[i] + V[i] * V[i]);
	}

	return BS;
}

vec CalculateEffectiveBulkShear(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                                shared_ptr<hitool>& h)
{
	const forecast_time forecastTime = myTargetInfo->Time();
	const forecast_type forecastType = myTargetInfo->ForecastType();
	const bool useCuda = conf->UseCudaForPacking();

	auto f = GET_PLUGIN(fetcher);

	auto ELInfo = f->Fetch(conf, forecastTime, level(kMaximumThetaE, 0), param("EL-LAST-M"), forecastType, useCuda);
	auto LPLInfo = f->Fetch(conf, forecastTime, level(kMaximumThetaE, 0), param("LPL-M"), forecastType, useCuda);

	const auto& EL = VEC(ELInfo);
	const auto& LPL = VEC(LPLInfo);

	vec Midway(EL.size(), MissingDouble());

	for (auto&& tup : zip_range(Midway, LPL, EL))
	{
		double& mid = tup.get<0>();
		const double lpl = tup.get<1>();
		const double el = tup.get<2>();

		mid = 0.5 * (el - lpl) + lpl;
	}

	const auto U = STABILITY::Shear(h, param("U-MS"), LPL, Midway);
	const auto V = STABILITY::Shear(h, param("V-MS"), LPL, Midway);

	vec BS(U.size(), MissingDouble());

	for (size_t i = 0; i < U.size(); i++)
	{
		BS[i] = sqrt(U[i] * U[i] + V[i] * V[i]);
	}

	return BS;
}

void CalculateBulkShearIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                               shared_ptr<hitool>& h)
{
	myTargetInfo->Param(BSParam);

	myTargetInfo->Level(OneKMLevel);
	const auto BS01 = CalculateBulkShear(myTargetInfo, h, 1000);
	myTargetInfo->Data().Set(BS01);

	myTargetInfo->Level(ThreeKMLevel);
	const auto BS03 = CalculateBulkShear(myTargetInfo, h, 3000);
	myTargetInfo->Data().Set(BS03);

	myTargetInfo->Level(SixKMLevel);
	const auto BS06 = CalculateBulkShear(myTargetInfo, h, 6000);
	myTargetInfo->Data().Set(BS06);

	myTargetInfo->Param(EBSParam);
	myTargetInfo->Level(Height0Level);
	const auto EBS = CalculateEffectiveBulkShear(conf, myTargetInfo, h);
	myTargetInfo->Data().Set(EBS);
}

void stability::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{
	auto myThreadedLogger = logger("stabilityThread #" + to_string(theThreadIndex));

	const forecast_time forecastTime = myTargetInfo->Time();
	const level forecastLevel = myTargetInfo->Level();
	const forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(forecastTime);
	h->ForecastType(forecastType);

	try
	{
		vec FF1500 = h->VerticalValue(param("FF-MS"), 1500);

		myTargetInfo->Param(FFParam);
		myTargetInfo->Level(EuropeanMileLevel);
		myTargetInfo->Data().Set(FF1500);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
		}
	}

	string deviceType = "CPU";

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		RunCuda(itsConfiguration, myTargetInfo, h);
	}
	else
#endif
	{
		try
		{
			CalculateStaticIndices(itsConfiguration, myTargetInfo);
			myThreadedLogger.Info("Static index calculation finished");
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("Static index calculation failed");

			if (e != kFileDataNotFound)
			{
			}
		}
		try
		{
			CalculateDynamicIndices(itsConfiguration, myTargetInfo, h);
			myThreadedLogger.Info("Dynamic index calculation finished");
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("Dynamic index calculation failed");

			if (e != kFileDataNotFound)
			{
			}
		}
		try
		{
			CalculateThetaEIndices(itsConfiguration, myTargetInfo, h);
			myThreadedLogger.Info("ThetaE index calculation finished");
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("ThetaE index calculation failed");

			if (e != kFileDataNotFound)
			{
			}
		}
		try
		{
			CalculateBulkShearIndices(itsConfiguration, myTargetInfo, h);
			myThreadedLogger.Info("Bulk shear calculation finished");
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("Bulk shear calculation failed");

			if (e != kFileDataNotFound)
			{
			}
		}

		try
		{
			CalculateHelicityIndices(itsConfiguration, myTargetInfo, h);
			myThreadedLogger.Info("Storm relative helicity calculation finished");
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("Storm relative helicity calculation failed");

			if (e != kFileDataNotFound)
			{
			}
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing: " + to_string(util::MissingPercent(*myTargetInfo)) + "%");
}

pair<vec, vec> GetSRHSourceData(const shared_ptr<info>& myTargetInfo, shared_ptr<hitool> h)
{
	// NOTES COPIED FROM SMARTTOOLS-LIBRARY

	/* // **********  SRH calculation help from Pieter Groenemeijer ******************

	Some tips here on how tyo calculate storm-relative helciity

	How to calculate storm-relative helicity

	Integrate the following from p = p_surface to p = p_top (or in case of height coordinates from h_surface to h_top):

	storm_rel_helicity -= ((u_ID-u[p])*(v[p]-v[p+1]))-((v_ID - v[p])*(u[p]-u[p+1]));

	Here, u_ID and v_ID are the forecast storm motion vectors calculated with the so-called ID-method. These can be calculated as follows:

	where

	/average wind
	u0_6 = average 0_6 kilometer u-wind component
	v0_6 = average 0_6 kilometer v-wind component
	(you should use a pressure-weighted average in case you work with height coordinates)

	/shear
	shr_0_6_u = u_6km - u_surface;
	shr_0_6_v = v_6km - v_surface;

	/ shear unit vector
	shr_0_6_u_n = shr_0_6_u / ((shr_0_6_u^2 + shr_0_6_v^2)**0.5);
	shr_0_6_v_n = shr_0_6_v / ((shr_0_6_u^2 + shr_0_6_v^2)** 0.5);

	/id-vector components
	u_ID = u0_6 + shr_0_6_v_n * 7.5;
	v_ID = v0_6 - shr_0_6_u_n * 7.5;

	(7.5 are meters per second... watch out when you work with knots instead)

	*/  // **********  SRH calculation help from Pieter Groenemeijer ******************

	// average wind
	auto Uavg = h->VerticalAverage(param("U-MS"), 10, 6000);
	auto Vavg = h->VerticalAverage(param("V-MS"), 10, 6000);

	// shear
	auto Ushear = STABILITY::Shear(h, param("U-MS"), 10, 6000, Uavg.size());
	auto Vshear = STABILITY::Shear(h, param("V-MS"), 10, 6000, Uavg.size());

	// U & V id vectors
	vec Uid(Ushear.size(), MissingDouble());
	auto Vid = Uid;

	for (size_t i = 0; i < Ushear.size(); i++)
	{
		STABILITY::UVId(Ushear[i], Vshear[i], Uavg[i], Vavg[i], Uid[i], Vid[i]);
	}

	return make_pair(Uid, Vid);
}

void RunCuda(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	unique_ptr<stability_cuda::options> opts(new stability_cuda::options);

	myTargetInfo->Level(Height0Level);
	myTargetInfo->Param(KIParam);
	opts->ki = myTargetInfo->ToSimple();

	myTargetInfo->Param(VTIParam);
	opts->vti = myTargetInfo->ToSimple();

	myTargetInfo->Param(CTIParam);
	opts->cti = myTargetInfo->ToSimple();

	myTargetInfo->Param(TTIParam);
	opts->tti = myTargetInfo->ToSimple();

	myTargetInfo->Param(LIParam);
	opts->li = myTargetInfo->ToSimple();

	myTargetInfo->Param(SIParam);
	opts->si = myTargetInfo->ToSimple();

	myTargetInfo->Param(BSParam);
	myTargetInfo->Level(OneKMLevel);
	opts->bs01 = myTargetInfo->ToSimple();

	myTargetInfo->Level(ThreeKMLevel);
	opts->bs03 = myTargetInfo->ToSimple();

	myTargetInfo->Level(SixKMLevel);
	opts->bs06 = myTargetInfo->ToSimple();

	myTargetInfo->Param(EBSParam);
	myTargetInfo->Level(Height0Level);
	opts->ebs = myTargetInfo->ToSimple();

	myTargetInfo->Param(SRHParam);
	myTargetInfo->Level(OneKMLevel);
	opts->srh01 = myTargetInfo->ToSimple();

	myTargetInfo->Level(ThreeKMLevel);
	opts->srh03 = myTargetInfo->ToSimple();

	myTargetInfo->Param(TPEParam);
	opts->thetae3 = myTargetInfo->ToSimple();

	myTargetInfo->Param(EHIParam);
	myTargetInfo->Level(OneKMLevel);
	opts->ehi = myTargetInfo->ToSimple();

	myTargetInfo->Param(BRNParam);
	myTargetInfo->Level(SixKMLevel);
	opts->brn = myTargetInfo->ToSimple();

	opts->N = myTargetInfo->SizeLocations();
	opts->h = h;
	opts->conf = conf;
	opts->myTargetInfo = make_shared<info>(*myTargetInfo);

	stability_cuda::Process(*opts);
}

void stability::RunTimeDimension(info_t myTargetInfo, unsigned short threadIndex)
{
	myTargetInfo->FirstLevel();
	while (NextExcludingLevel(*myTargetInfo))
	{
		if (itsConfiguration->UseDynamicMemoryAllocation())
		{
			AllocateMemory(*myTargetInfo);
		}

		ASSERT(myTargetInfo->Data().Size() > 0);

		Calculate(myTargetInfo, threadIndex);

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
		}
	}
	WriteToFile(myTargetInfo);
}

void stability::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);

	// writing might modify iterator positions --> create a copy

	auto tempInfo = make_shared<info>(*targetInfo);

	tempInfo->ResetLevel();

	while (tempInfo->NextLevel())
	{
		for (tempInfo->ResetParam(); tempInfo->NextParam();)
		{
			if (!tempInfo->IsValidGrid())
			{
				continue;
			}

			if (itsConfiguration->FileWriteOption() == kDatabase ||
			    itsConfiguration->FileWriteOption() == kMultipleFiles)
			{
				aWriter->ToFile(tempInfo, itsConfiguration);
			}
			else
			{
				aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
			}
		}
	}
	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(*targetInfo);
	}
}
