#include "stability.h"
#include "forecast_time.h"
#include "level.h"
#include "lift.h"
#include "logger.h"
#include "numerical_functions.h"
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

pair<vec, vec> GetSRHSourceData(const shared_ptr<info<double>>& myTargetInfo, shared_ptr<hitool> h);
vec CalculateBulkShear(const vec& U, const vec& V);
vec CalculateBulkShear(info_t& myTargetInfo, shared_ptr<hitool>& h, double stopHeight);
vec CalculateEffectiveBulkShear(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                                shared_ptr<hitool>& h, const level& sourceLevel, const level& targetLevel);

#ifdef HAVE_CUDA
namespace stabilitygpu
{
void Process(shared_ptr<const plugin_configuration> conf, info_t myTargetInfo);
}
#endif

namespace STABILITY
{
himan::info_t Fetch(std::shared_ptr<const plugin_configuration>& conf,
                    std::shared_ptr<himan::info<double>>& myTargetInfo, const himan::level& lev,
                    const himan::param& par, bool returnPacked = false);

vec Shear(const vec& lowerValues, const vec& upperValues)
{
	vec ret(lowerValues.size());

	for (size_t i = 0; i < lowerValues.size(); i++)
	{
		ret[i] = upperValues[i] - lowerValues[i];
	}

	return ret;
}

vec Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par, const vec& lowerHeight,
          const vec& upperHeight)
{
	const auto lowerValues = h->VerticalValue<double>(par, lowerHeight);
	const auto upperValues = h->VerticalValue<double>(par, upperHeight);

	return Shear(lowerValues, upperValues);
}

vec Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par, double lowerHeight, double upperHeight,
          size_t N)
{
	const vec lower(N, lowerHeight);
	const vec upper(N, upperHeight);

	return Shear(h, par, lower, upper);
}

himan::info_t Fetch(std::shared_ptr<const plugin_configuration>& conf,
                    std::shared_ptr<himan::info<double>>& myTargetInfo, const himan::level& lev,
                    const himan::param& par, bool useCuda)
{
	const forecast_time forecastTime = myTargetInfo->Time();
	const forecast_type forecastType = myTargetInfo->ForecastType();

	auto f = GET_PLUGIN(fetcher);

	return f->Fetch(conf, forecastTime, lev, par, forecastType, useCuda && conf->UseCudaForPacking());
}

pair<vec, vec> GetEBSLevelData(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                               shared_ptr<hitool>& h, const level& sourceLevel, const level& targetLevel)
{
	auto ELInfo = STABILITY::Fetch(conf, myTargetInfo, sourceLevel, param("EL-LAST-M"));
	const auto& EL = VEC(ELInfo);

	vec LPL;

	// LPL only exists for Max theta e level
	if (sourceLevel == MaxThetaELevel)
	{
		auto LPLInfo = STABILITY::Fetch(conf, myTargetInfo, sourceLevel, param("LPL-M"));
		LPL = VEC(LPLInfo);
	}
	else
	{
		LPL.resize(EL.size(), 2);
	}

	vec Top(EL.size(), MissingDouble());

	if (targetLevel == Height0Level)
	{
		for (auto&& tup : zip_range(Top, LPL, EL))
		{
			double& top = tup.get<0>();
			const double lpl = tup.get<1>();
			const double el = tup.get<2>();

			top = 0.5 * (el - lpl) + lpl;
		}

		return make_pair(LPL, Top);
	}
	else if (targetLevel == MaxWindLevel)
	{
		vec Bottom = Top;

		// Finding maximum wind between levels LPL + 0.5*(0.6EL - LPL) and 0.6E
		// Hence, if LPL is on the surface, being ~0m, the search limits for maximum wind become 0.3EL ... 0.6EL

		for (auto&& tup : zip_range(Bottom, Top, LPL, EL))
		{
			double& btm = tup.get<0>();
			double& top = tup.get<1>();
			const double lpl = tup.get<2>();
			const double el = tup.get<3>();

			if (el - lpl >= 3000.)
			{
				btm = 0.5 * (0.6 * el - lpl) + lpl;
				top = 0.6 * el;
			}

			ASSERT(btm < top || (IsMissing(btm) && IsMissing(top)));
		}

		const auto maxWind = h->VerticalMaximum(FFParam, Bottom, Top);
		const auto maxWindHeight = h->VerticalHeight(FFParam, Bottom, Top, maxWind);

		return make_pair(LPL, maxWindHeight);
	}

	throw runtime_error("Invalid target level: " + static_cast<string>(targetLevel));
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

	itsBottomLevel = level(kHybrid, stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->TargetProducer().Id(),
	                                                                      "last hybrid level number")));

#ifdef HAVE_CUDA
	stability_cuda::itsBottomLevel = itsBottomLevel;
#endif

	itsThreadDistribution = ThreadDistribution::kThreadForForecastTypeAndTime;

	itsLevelIterator.Clear();

	SetParams({CSIParam, LIParam, SIParam, CAPESParam}, {Height0Level});
	SetParams({BSParam}, {OneKMLevel, ThreeKMLevel, SixKMLevel});
	SetParams({SRHParam}, {OneKMLevel, ThreeKMLevel});
	SetParams({TPEParam}, {ThreeKMLevel});
	SetParams({EHIParam}, {OneKMLevel});
	SetParams({BRNParam}, {SixKMLevel});
	SetParams({FFParam}, {EuropeanMileLevel});
	SetParams({QParam}, {HalfKMLevel});
	SetParams({EBSParam}, {MaxWindLevel});

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
	auto prevZInfo = STABILITY::Fetch(conf, myTargetInfo, itsBottomLevel, HLParam);

	vector<bool> found(SRH.size(), false);

	level curLevel = itsBottomLevel;

	while (curLevel.Value() > 0)
	{
		curLevel.Value(curLevel.Value() - 1);

		auto UInfo = STABILITY::Fetch(conf, myTargetInfo, curLevel, UParam);
		auto VInfo = STABILITY::Fetch(conf, myTargetInfo, curLevel, VParam);
		auto ZInfo = STABILITY::Fetch(conf, myTargetInfo, curLevel, HLParam);

		const auto& U = VEC(UInfo);
		const auto& V = VEC(VInfo);
		const auto& Z = VEC(ZInfo);
		const auto& prevU = VEC(prevUInfo);
		const auto& prevV = VEC(prevVInfo);
		const auto& prevZ = VEC(prevZInfo);

		for (size_t i = 0; i < SRH.size(); i++)
		{
			if (found[i])
			{
				continue;
			}
			const double _Uid = Uid[i];
			const double _Vid = Vid[i];

			const double _pU = prevU[i];
			const double _pV = prevV[i];

			double _U = U[i];
			double _V = V[i];

			if (Z[i] > stopHeight)
			{
				ASSERT(prevZ[i] < stopHeight);

				_U = numerical_functions::interpolation::Linear<double>(stopHeight, prevZ[i], Z[i], _pU, _U);
				_V = numerical_functions::interpolation::Linear<double>(stopHeight, prevZ[i], Z[i], _pV, _V);

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
		{
			v = MissingDouble();
		}
	}

	return SRH;
}

vec CalculateBulkRichardsonNumber(shared_ptr<const plugin_configuration> conf, info_t& myTargetInfo,
                                  shared_ptr<hitool> h)
{
	auto CAPEInfo = STABILITY::Fetch(conf, myTargetInfo, level(kHeightLayer, 500, 0), param("CAPE-JKG"));
	const auto& CAPE = VEC(CAPEInfo);

	auto U6 = h->VerticalAverage<double>(UParam, 10, 6000);
	auto V6 = h->VerticalAverage<double>(VParam, 10, 6000);

	auto U05 = h->VerticalAverage<double>(UParam, 10, 500);
	auto V05 = h->VerticalAverage<double>(VParam, 10, 500);

	vec BRN(CAPE.size(), MissingDouble());

	for (size_t i = 0; i < BRN.size(); i++)
	{
		BRN[i] = STABILITY::BRN(CAPE[i], U6[i], V6[i], U05[i], V05[i]);
	}

	return BRN;
}

vec CalculateEnergyHelicityIndex(shared_ptr<const plugin_configuration> conf, info_t& myTargetInfo)
{
	auto CAPEInfo = STABILITY::Fetch(conf, myTargetInfo, level(kHeightLayer, 500, 0), param("CAPE-JKG"));

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

	myTargetInfo->Find<param>(SRHParam);

	myTargetInfo->Find<level>(ThreeKMLevel);
	const auto SRH03 = CalculateStormRelativeHelicity(conf, myTargetInfo, h, 3000, UVId);
	myTargetInfo->Data().Set(SRH03);

	myTargetInfo->Find<level>(OneKMLevel);
	const auto SRH01 = CalculateStormRelativeHelicity(conf, myTargetInfo, h, 1000, UVId);
	myTargetInfo->Data().Set(SRH01);

	const auto EHI01 = CalculateEnergyHelicityIndex(conf, myTargetInfo);
	myTargetInfo->Find<param>(EHIParam);
	myTargetInfo->Data().Set(EHI01);

	myTargetInfo->Find<level>(SixKMLevel);
	myTargetInfo->Find<param>(BRNParam);
	const auto BRN = CalculateBulkRichardsonNumber(conf, myTargetInfo, h);
	myTargetInfo->Data().Set(BRN);
}

tuple<vec, vec, vec, info_t, info_t, info_t> GetLiftedIndicesSourceData(shared_ptr<const plugin_configuration>& conf,
                                                                        info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	auto T500 = h->VerticalAverage<double>(TParam, 0., 500.);
	auto P500 = h->VerticalAverage<double>(PParam, 0., 500.);

	vec TD500;

	try
	{
		TD500 = h->VerticalAverage<double>(param("TD-K"), 0., 500.);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			TD500 = h->VerticalAverage<double>(RHParam, 0., 500.);

			for (auto&& tup : zip_range(TD500, T500))
			{
				double& res = tup.get<0>();
				double t = tup.get<1>();

				res = metutil::DewPointFromRH_<double>(t, res);
			}
		}
	}

	if (P500[0] < 1500)
	{
		transform(P500.begin(), P500.end(), P500.begin(), bind1st(multiplies<double>(), 100.));  // hPa to Pa
	}

	auto T850Info = STABILITY::Fetch(conf, myTargetInfo, P850Level, TParam);
	auto TD850Info = STABILITY::Fetch(conf, myTargetInfo, P850Level, TDParam);
	auto T500Info = STABILITY::Fetch(conf, myTargetInfo, P500Level, TParam);

	return make_tuple(T500, TD500, P500, T850Info, TD850Info, T500Info);
}

vec CalculateThetaE(shared_ptr<hitool>& h, double startHeight, double stopHeight)
{
	auto Tstop = h->VerticalValue<double>(TParam, stopHeight);
	auto RHstop = h->VerticalValue<double>(RHParam, stopHeight);
	auto Pstop = h->VerticalValue<double>(PParam, stopHeight);

	auto Tstart = h->VerticalValue<double>(TParam, startHeight);
	auto RHstart = h->VerticalValue<double>(RHParam, startHeight);
	auto Pstart = h->VerticalValue<double>(PParam, startHeight);

	vec ret(Tstop.size());

	for (size_t i = 0; i < Tstart.size(); i++)
	{
		const double ThetaEstart = metutil::smarttool::ThetaE_<double>(Tstart[i], RHstart[i], Pstart[i] * 100);
		const double ThetaEstop = metutil::smarttool::ThetaE_<double>(Tstop[i], RHstop[i], Pstop[i] * 100);

		ret[i] = ThetaEstart - ThetaEstop;
	}

	return ret;
}

void CalculateThetaEIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	vec thetae = CalculateThetaE(h, 2, 3000);

	myTargetInfo->Find<level>(ThreeKMLevel);
	myTargetInfo->Find<param>(TPEParam);
	myTargetInfo->Data().Set(thetae);
}

void CalculateLiftedIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, shared_ptr<hitool>& h)
{
	auto src = GetLiftedIndicesSourceData(conf, myTargetInfo, h);

	myTargetInfo->Find<level>(Height0Level);

	myTargetInfo->Find<param>(LIParam);
	auto& LI = VEC(myTargetInfo);

	myTargetInfo->Find<param>(SIParam);
	auto& SI = VEC(myTargetInfo);

	const auto& t500m = get<0>(src);
	const auto& td500m = get<1>(src);
	const auto& p500m = get<2>(src);
	const auto& t850 = VEC(get<3>(src));
	const auto& td850 = VEC(get<4>(src));
	const auto& t500 = VEC(get<5>(src));

	vec p500(t500.size(), 50000.);
	vec p850(t500.size(), 85000.);

	for (size_t i = 0; i < t500.size(); i++)
	{
		// Lift parcel from lowest 500m average pressure to 500hPa
		const double t_li = metutil::Lift_<double>(p500m[i], t500m[i], td500m[i], p500[i]);

		// Lift parcel from 850hPa to 500hPa
		const double t_si = metutil::Lift_<double>(p850[i], t850[i], td850[i], p500[i]);

		LI[i] = t500[i] - t_li;
		SI[i] = t500[i] - t_si;
	}
}

vec CalculateCapeShear(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo, const vec& EBS)
{
	auto CAPEInfo = STABILITY::Fetch(conf, myTargetInfo, MaxThetaELevel, param("CAPE-JKG"));

	const auto& CAPE = VEC(CAPEInfo);

	vec ret(EBS.size());

	for (size_t i = 0; i < EBS.size(); i++)
	{
		ret[i] = EBS[i] * sqrt(CAPE[i]);
	}

	return ret;
}

void CalculateBulkShearIndices(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                               shared_ptr<hitool>& h)
{
	myTargetInfo->Find<param>(BSParam);

	myTargetInfo->Find<level>(OneKMLevel);
	const auto BS01 = CalculateBulkShear(myTargetInfo, h, 1000);
	myTargetInfo->Data().Set(BS01);

	myTargetInfo->Find<level>(ThreeKMLevel);
	const auto BS03 = CalculateBulkShear(myTargetInfo, h, 3000);
	myTargetInfo->Data().Set(BS03);

	myTargetInfo->Find<level>(SixKMLevel);
	const auto BS06 = CalculateBulkShear(myTargetInfo, h, 6000);
	myTargetInfo->Data().Set(BS06);

	// CAPE shear is calculated here too
	const auto normEBS = CalculateEffectiveBulkShear(conf, myTargetInfo, h, MaxThetaELevel, Height0Level);
	const auto CAPES = CalculateCapeShear(conf, myTargetInfo, normEBS);
	myTargetInfo->Find<level>(Height0Level);
	myTargetInfo->Find<param>(CAPESParam);
	myTargetInfo->Data().Set(CAPES);

	// Calculate maximum EBS
	const auto muMaxEBS = CalculateEffectiveBulkShear(conf, myTargetInfo, h, MaxThetaELevel, MaxWindLevel);
	myTargetInfo->Find<param>(EBSParam);
	myTargetInfo->Find<level>(MaxWindLevel);
	myTargetInfo->Data().Set(muMaxEBS);
}

void CalculateConvectiveSeverityIndex(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                                      shared_ptr<hitool>& h)
{
	// For CSI we need mixed layer maximum EBS too, which is not needed anywhere else
	const auto mlEBS = CalculateEffectiveBulkShear(conf, myTargetInfo, h, HalfKMLevel, MaxWindLevel);

	auto muCAPEInfo = STABILITY::Fetch(conf, myTargetInfo, MaxThetaELevel, param("CAPE-JKG"));
	auto muLPLInfo = STABILITY::Fetch(conf, myTargetInfo, MaxThetaELevel, param("LPL-M"));
	auto mlCAPEInfo = STABILITY::Fetch(conf, myTargetInfo, HalfKMLevel, param("CAPE-JKG"));

	myTargetInfo->Find<param>(EBSParam);
	myTargetInfo->Find<level>(MaxWindLevel);

	const auto& muEBS = VEC(myTargetInfo);
	const auto& muLPL = VEC(muLPLInfo);
	const auto& muCAPE = VEC(muCAPEInfo);
	const auto& mlCAPE = VEC(mlCAPEInfo);

	myTargetInfo->Find<param>(CSIParam);
	myTargetInfo->Find<level>(Height0Level);

	auto& CSI = VEC(myTargetInfo);

	for (size_t i = 0; i < muEBS.size(); i++)
	{
		auto cape = MissingDouble();
		auto ebs = MissingDouble();

		if (muLPL[i] >= 250. && muCAPE[i] > 10.)
		{
			cape = muCAPE[i];
			ebs = muEBS[i];
		}
		else if (muLPL[i] < 250. && mlCAPE[i] > 10.)
		{
			cape = mlCAPE[i];
			ebs = mlEBS[i];
		}

		CSI[i] = (ebs * sqrt(2 * cape)) * 0.1;

		if (ebs <= 15.)
		{
			CSI[i] += 0.025 * cape * (-0.06666 * ebs + 1);
		}
	}
}

void stability::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short theThreadIndex)
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
		vec FF1500 = h->VerticalValue<double>(FFParam, 1500);

		myTargetInfo->Find<param>(FFParam);
		myTargetInfo->Find<level>(EuropeanMileLevel);
		myTargetInfo->Data().Set(FF1500);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
		}
	}

	try
	{
		vec Q500 = h->VerticalAverage<double>(QParam, 0, 500);

		myTargetInfo->Find<param>(QParam);
		myTargetInfo->Find<level>(HalfKMLevel);
		myTargetInfo->Data().Set(Q500);
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

		stabilitygpu::Process(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		try
		{
			CalculateLiftedIndices(itsConfiguration, myTargetInfo, h);
			myThreadedLogger.Info("Lifted index calculation finished");
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("Lifted index calculation failed");

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
		try
		{
			CalculateConvectiveSeverityIndex(itsConfiguration, myTargetInfo, h);
		}
		catch (const HPExceptionType& e)
		{
			itsLogger.Warning("Convective stability index calculation failed");

			if (e != kFileDataNotFound)
			{
			}
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing: " + to_string(util::MissingPercent(*myTargetInfo)) + "%");
}

pair<vec, vec> GetSRHSourceData(const shared_ptr<info<double>>& myTargetInfo, shared_ptr<hitool> h)
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
	auto Uavg = h->VerticalAverage<double>(UParam, 10, 6000);
	auto Vavg = h->VerticalAverage<double>(VParam, 10, 6000);

	// shear
	auto Ushear = STABILITY::Shear(h, UParam, 10, 6000, Uavg.size());
	auto Vshear = STABILITY::Shear(h, VParam, 10, 6000, Uavg.size());

	// U & V id vectors
	vec Uid(Ushear.size(), MissingDouble());
	auto Vid = Uid;

	for (size_t i = 0; i < Ushear.size(); i++)
	{
		STABILITY::UVId(Ushear[i], Vshear[i], Uavg[i], Vavg[i], Uid[i], Vid[i]);
	}

	return make_pair(Uid, Vid);
}

void stability::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);

	// writing might modify iterator positions --> create a copy

	auto tempInfo = make_shared<info<double>>(*targetInfo);

	tempInfo->Reset<level>();

	while (tempInfo->Next<level>())
	{
		for (tempInfo->Reset<param>(); tempInfo->Next<param>();)
		{
			if (!tempInfo->IsValidGrid())
			{
				continue;
			}

			if (itsConfiguration->WriteMode() == kSingleGridToAFile)
			{
				aWriter->ToFile(tempInfo, itsConfiguration);
			}
			else
			{
				aWriter->ToFile(tempInfo, itsConfiguration);
			}
		}
	}
	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(*targetInfo);
	}
}

vec CalculateBulkShear(const vec& U, const vec& V)
{
	vec BS(U.size());

	for (size_t i = 0; i < U.size(); i++)
	{
		BS[i] = hypot(U[i], V[i]);
	}

	return BS;
}

vec CalculateBulkShear(info_t& myTargetInfo, shared_ptr<hitool>& h, double stopHeight)
{
	const auto U = STABILITY::Shear(h, UParam, 10, stopHeight, myTargetInfo->SizeLocations());
	const auto V = STABILITY::Shear(h, VParam, 10, stopHeight, myTargetInfo->SizeLocations());

	return CalculateBulkShear(U, V);
}

vec CalculateEffectiveBulkShear(shared_ptr<const plugin_configuration>& conf, info_t& myTargetInfo,
                                shared_ptr<hitool>& h, const level& sourceLevel, const level& targetLevel)
{
	const auto Levels = STABILITY::GetEBSLevelData(conf, myTargetInfo, h, sourceLevel, targetLevel);

	const auto U = STABILITY::Shear(h, UParam, Levels.first, Levels.second);
	const auto V = STABILITY::Shear(h, VParam, Levels.first, Levels.second);

	return CalculateBulkShear(U, V);
}
