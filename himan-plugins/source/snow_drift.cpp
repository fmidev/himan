#include "snow_drift.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "util.h"

#include "fetcher.h"
#include "hitool.h"
#include "radon.h"
#include "writer.h"

using namespace himan;
using namespace himan::plugin;

const param SDIParam("SNOWDRIFT-N");
const param SAParam("SNACC-H");
const param DAParam("SNDACC-N");
const double SFLimit = 0.09;
const double GustFactor = 2.0;

double DriftMagnitude(double ff, double mi);
double DriftIndex(double sv);
double MobilityIndex(double da, double sa);

void CalculateSnowDriftIndex(info_t& myTargetInfo, const std::vector<double>& T, const std::vector<double>& FF,
                             const std::vector<double>& FFG, const std::vector<double>& SF,
                             const std::vector<double>& pSA, const std::vector<double>& pDA,
                             const std::vector<bool>& mask)
{
	myTargetInfo->Find<param>(SDIParam);
	auto& SI = VEC(myTargetInfo);
	myTargetInfo->Find<param>(SAParam);
	auto& SA = VEC(myTargetInfo);
	myTargetInfo->Find<param>(DAParam);
	auto& DA = VEC(myTargetInfo);

	for (auto&& tup : zip_range(SI, SA, DA, T, FF, FFG, SF, pSA, pDA, mask))
	{
		auto& si = tup.get<0>();        // snow drift index
		auto& sa = tup.get<1>();        // accumulated snow cover age
		auto& da = tup.get<2>();        // accumulated snowdrift value
		const auto t = tup.get<3>();    // temperature
		const auto ff = tup.get<4>();   // wind speed
		const auto ffg = tup.get<5>();  // wind gust speed
		const auto sf = tup.get<6>();   // snow fall rate (mm/h)
		const auto psa = tup.get<7>();  // previous accumulated snow cover age
		const auto pda = tup.get<8>();  // previous accumulated snowdrift value
		const auto m = tup.get<9>();    // mask value for this grid point (true: process grid point)

		if (IsMissing(t) || IsMissing(ff) || IsMissing(sf) || IsMissing(psa) || IsMissing(pda) || !m)
		{
			continue;
		}

		if (t > 273.15)
		{
			// no possibility for snow drift until next snow fall
			si = 0;
			sa = 0;
			da = 0;
			continue;
		}

		/*
		        this is not present in the fortran code either
		        if (ff < 6.0)
		        {
		            si = 0;
		        }
		*/

		double eff = ff;  // effective wind speed, factors in gustyness

		if ((ffg / ff) > GustFactor)
		{
			eff = ((ffg / ff) - GustFactor) * ff + ff;
		}

		if (sf > SFLimit)
		{
			// during snow fall
			si = DriftIndex(DriftMagnitude(eff, 1.0));
			sa = 0;
			da = 0;
		}
		else
		{
			// after snow fall
			sa = psa + 1;  // add one hour to snowcover age

			if (eff < 6)
			{
				si = 0;
				da = pda;
			}
			else
			{
				const double sv = DriftMagnitude(eff, MobilityIndex(pda, sa));
				da = pda + sv;
				si = DriftIndex(sv);
			}
		}
	}
}

snow_drift::snow_drift()
{
	itsLogger = logger("snow_drift");
}

void snow_drift::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({SDIParam, SAParam, DAParam});

	itsThreadDistribution = ThreadDistribution::kThreadForForecastTypeAndLevel;

	Start();
}

void snow_drift::Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short theThreadIndex)
{
	auto myThreadedLogger = logger("snow_driftThread #" + std::to_string(theThreadIndex));

	const auto forecastLevel = myTargetInfo->Level();
	const auto forecastType = myTargetInfo->ForecastType();
	const auto prod = myTargetInfo->Producer();

	const std::string deviceType = "CPU";

	info_t pSAInfo, pDAInfo;  // data from previous time step

	for (myTargetInfo->Reset<forecast_time>(); myTargetInfo->Next<forecast_time>();)
	{
		const auto forecastTime = myTargetInfo->Time();

		if (forecastTime.Step().Hours() == 0 && prod.Id() != 107)
		{
			// We only calculate analysis hour for LAPS
			continue;
		}

		myThreadedLogger.Info("Calculating time " + static_cast<std::string>(forecastTime.ValidDateTime()) + " level " +
		                      static_cast<std::string>(forecastLevel));

		auto TInfo = Fetch(forecastTime, level(kHeight, 2), param("T-K"), forecastType, false);
		auto FFInfo = Fetch(forecastTime, level(kHeight, 10), param("FF-MS"), forecastType, false);
		auto FFGInfo = Fetch(forecastTime, level(kHeight, 10), param("FFG-MS"), forecastType, false);
		auto SFInfo = Fetch(forecastTime, level(kHeight, 0), param("SNR-KGM2"), forecastType, false);

		if (!TInfo || !FFInfo || !SFInfo)
		{
			continue;
		}

		// Use ice coverage information to mask out points on the sea where there is no ice
		auto iceTime = forecastTime;
		iceTime.OriginDateTime().Adjust(kHourResolution, -std::stoi(iceTime.OriginDateTime().String("%H")));
		iceTime.OriginDateTime().Adjust(kHourResolution, 12);
		iceTime.ValidDateTime(iceTime.OriginDateTime());

		const producer vanadis(105, 86, 11, "MTLICE");

		auto MaskInfo = Fetch(iceTime, level(kHeight, 0), param("ICNCT-PRCNT"), forecast_type(kDeterministic),
		                      itsConfiguration->SourceGeomNames(), vanadis, false);

		if (!MaskInfo)
		{
			iceTime.OriginDateTime().Adjust(kHourResolution, -24);
			iceTime.ValidDateTime().Adjust(kHourResolution, -24);

			MaskInfo = Fetch(iceTime, level(kHeight, 0), param("ICNCT-PRCNT"), forecast_type(kDeterministic),
			                 itsConfiguration->SourceGeomNames(), vanadis, false);
		}

		if (!MaskInfo)
		{
			// if ice coverage information is not found, use land-sea mask
			MaskInfo = Fetch(forecastTime, level(kHeight, 0), param("LC-0TO1"), forecastType, false);
		}

		// Is mask value is true, grid point will be processed
		// --> If no mask is found (Ice cover OR land sea mask), all grid points are accepted
		std::vector<bool> mask(myTargetInfo->SizeLocations(), true);

		if (MaskInfo)
		{
			const double threshold = (MaskInfo->Param().Name() == "ICNCT-PRCNT") ? 70 : 0.5;

			const auto& vec = VEC(MaskInfo);

			// grid point is accepted if: value of mask is missing (ie land point if ice cover data), or
			// if value is over given threshold
			std::transform(vec.begin(), vec.end(), mask.begin(),
			               [&](double d) { return IsMissing(d) || d >= threshold; });
		}

		auto prevTime = forecastTime;
		prevTime.ValidDateTime().Adjust(kHourResolution, -1);

		// In the start of the forecast fetch the latest sa and da
		// values from LAPS producer.

		if (prod.Id() == 107 || forecastTime.Step().Hours() == 1)
		{
			prevTime.OriginDateTime(prevTime.ValidDateTime());

			const std::vector<std::string> lapsGeom({"LAPS3000"});
			const producer lapsProd(107, 86, 107, "LAPSFIN");

			// Previous data can be max three hours old
			for (int i = 0; i < 3; i++)
			{
				pSAInfo =
				    Fetch(prevTime, level(kHeight, 0), SAParam, forecast_type(kAnalysis), lapsGeom, lapsProd, false);
				pDAInfo =
				    Fetch(prevTime, level(kHeight, 0), DAParam, forecast_type(kAnalysis), lapsGeom, lapsProd, false);

				if (pSAInfo && pDAInfo)
				{
					break;
				}

				prevTime.ValidDateTime().Adjust(kHourResolution, -1);
				prevTime.OriginDateTime(prevTime.ValidDateTime());
			}
		}
		else if (!pSAInfo || !pDAInfo)
		{
			// Calculation started "mid" timeseries
			pSAInfo = Fetch(prevTime, level(kHeight, 0), SAParam, forecastType, false);
			pDAInfo = Fetch(prevTime, level(kHeight, 0), DAParam, forecastType, false);
		}

		if (!pSAInfo || !pDAInfo)
		{
			if (forecastTime.Step().Hours() == 0)
			{
				// If we don't have a history of sa & da, start accumulating
				// it but do not write snow index.

				std::vector<double> pSA(myTargetInfo->Grid()->Size(), 0.0);
				std::vector<double> pDA = pSA;

				myThreadedLogger.Warning("Spinup phase, producing only DA and SA");

				// LAPS does not provide FFG
				std::vector<double> FFG(pDA.size(), MissingDouble());
				CalculateSnowDriftIndex(myTargetInfo, VEC(TInfo), VEC(FFInfo), FFG, VEC(SFInfo), pSA, pDA, mask);

				myTargetInfo->Find<param>(SDIParam);
				myTargetInfo->Data().Fill(MissingDouble());
			}
			else
			{
				// strict mode: if previous forecasted sa & da is not found,
				// stop calculation
				break;
			}
		}
		else
		{
			std::vector<double> FFG;

			if (FFGInfo)
			{
				FFG = VEC(FFGInfo);
			}
			else
			{
				FFG.resize(myTargetInfo->SizeLocations(), MissingDouble());
			}

			CalculateSnowDriftIndex(myTargetInfo, VEC(TInfo), VEC(FFInfo), FFG, VEC(SFInfo), VEC(pSAInfo), VEC(pDAInfo),
			                        mask);
		}

		myTargetInfo->Find<param>(SAParam);
		pSAInfo = std::make_shared<info<double>>(*myTargetInfo);
		myTargetInfo->Find<param>(DAParam);
		pDAInfo = std::make_shared<info<double>>(*myTargetInfo);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing: " + std::to_string(util::MissingPercent(*myTargetInfo)) + "%");
}

double DriftMagnitude(double ff, double mi)
{
	return (ff * ff * ff / 1728.0) * mi;
}

double DriftIndex(double sv)
{
	if (sv < 0.09)
	{
		return 0;  // no drift
	}
	else if (sv < 0.21)
	{
		return 1;  // low
	}
	else if (sv < 0.5)
	{
		return 2;  // moderate
	}
	else
	{
		return 3;  // high
	}
}

double MobilityIndex(double da, double sa)
{
	double mi = 1.0;

	if (da < 2.0 && sa < 24.0)
	{
		mi = 1.0;
	}
	else if (da >= 2.0 && da <= 6.0)
	{
		mi = 0.6;
	}
	else if (da > 6.0)
	{
		mi = 0.3;
	}

	if (sa >= 24.0 && mi == 1.0)
	{
		mi = 0.6;
	}

	return mi;
}
