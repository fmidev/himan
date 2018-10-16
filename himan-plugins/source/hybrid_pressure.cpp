/**
 * @file hybrid_pressure.cpp
 *
 */

#include "hybrid_pressure.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"

#include "plugin_factory.h"
#include "util.h"

#include "cache.h"
#include "writer.h"

using namespace std;
using namespace himan::plugin;

mutex lnspMutex, mySingleFileWriteMutex;

hybrid_pressure::hybrid_pressure()
{
	itsLogger = logger("hybrid_pressure");
}

void hybrid_pressure::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param p("P-HPA", 1, 0, 3, 0);
	p.Unit(kHPa);

	SetParams({p});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void hybrid_pressure::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short theThreadIndex)
{
	params PParam{param("P-PA"), param("P-HPA")};
	const param TParam("T-K");
	level PLevel(himan::kHeight, 0, "HEIGHT");

	bool isECMWF = (itsConfiguration->TargetProducer().Id() == 240 || itsConfiguration->TargetProducer().Id() == 243);

	auto myThreadedLogger = logger("hybrid_pressureThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	info_t PInfo;

	double PScale = 1;

	if (isECMWF)
	{
		// For EC we calculate surface pressure from LNSP parameter

		PLevel = level(himan::kHybrid, 1);
		PParam = {param("LNSP-N")};

		// To make calculation more efficient we calculate surface
		// pressure once from LNSP and store it to cache as LNSP-HPA

		// Double-check pattern

		PInfo = Fetch(forecastTime, PLevel, param("LNSP-HPA"), forecastType, false);

		if (!PInfo)
		{
			lock_guard<mutex> lock(lnspMutex);

			PInfo = Fetch(forecastTime, PLevel, param("LNSP-HPA"), forecastType, false);

			if (!PInfo)
			{
				auto lnspn = Fetch(forecastTime, PLevel, PParam, forecastType, false);

				if (!lnspn)
				{
					myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
					                         static_cast<string>(forecastLevel));
					return;
				}

				myThreadedLogger.Info("Transforming LNSP to HPa for step " + to_string(forecastTime.Step()));

				auto newInfo = make_shared<info<double>>(*lnspn);
				newInfo->Set<param>(param("LNSP-HPA"));
				newInfo->Create(lnspn->Base());

				for (auto& val : VEC(newInfo))
				{
					val = 0.01 * exp(val);
				}

				auto c = GET_PLUGIN(cache);
				c->Insert(newInfo, true);
			}
		}

		PInfo = Fetch(forecastTime, PLevel, param("LNSP-HPA"), forecastType, false);
		PScale = 100;
	}
	else
	{
		PInfo = Fetch(forecastTime, PLevel, PParam, forecastType, false);
	}

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, false);

	if (!PInfo || !TInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	SetAB(myTargetInfo, TInfo);

	/*
	 * Vertical coordinates for full hybrid levels.
	 * For Hirlam data, coefficients A and B are already interpolated to full level coefficients in the grib-file.
	 * For Harmonie and ECMWF interpolation is done, when reading data from the grib-file. (NFmiGribMessage::PV)
	 */

	vector<double> ab = TInfo->Level().AB();

	double A, B;

	if (ab.size() == 2)
	{
		A = ab[0];
		B = ab[1];
	}
	else
	{
		const size_t levelValue = static_cast<size_t>(forecastLevel.Value());
		ASSERT(levelValue <= ab.size());

		A = (ab[levelValue - 1] + ab[levelValue]) * 0.5;

		const size_t halfsize = static_cast<size_t>(static_cast<double>(ab.size()) * 0.5);

		B = (ab[halfsize + levelValue - 1] + ab[halfsize + levelValue]) * 0.5;
	}

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(PInfo)))
	{
		double& result = tup.get<0>();
		double P = tup.get<1>();

		result = 0.01 * (A + P * PScale * B);
	}

	myThreadedLogger.Info("[CPU] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                      to_string(myTargetInfo->Data().Size()));
}

void hybrid_pressure::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	writeOptions.write_empty_grid = false;

	compiled_plugin_base::WriteToFile(targetInfo, writeOptions);
}
