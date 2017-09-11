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
map<string, himan::info_t> lnspInfos;

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

void hybrid_pressure::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{
	params PParam{param("P-PA"), param("P-HPA")};
	const param TParam("T-K");
	level PLevel(himan::kHeight, 0, "HEIGHT");

	bool isECMWF = (itsConfiguration->SourceProducer().Id() == 131 || itsConfiguration->SourceProducer().Id() == 134);

	auto myThreadedLogger = logger("hybrid_pressureThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	info_t PInfo;

	if (isECMWF)
	{
		// For EC we calculate surface pressure from LNSP parameter

		PLevel = level(himan::kHybrid, 1);
		PParam = {param("LNSP-N")};

		// To make calculation more efficient we calculate surface
		// pressure once from LNSP and store it to cache as LNSP-PA

		// Double-check pattern

		const auto key = static_cast<string>(forecastType) + "_" + to_string(forecastTime.Step());

		if (lnspInfos.find(key) == lnspInfos.end())
		{
			lock_guard<mutex> lock(lnspMutex);

			if (lnspInfos.find(key) == lnspInfos.end())
			{
				PInfo = Fetch(forecastTime, PLevel, PParam, forecastType, false);

				if (!PInfo)
				{
					myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
					                         static_cast<string>(forecastLevel));
					return;
				}

				myThreadedLogger.Info("Transforming LNSP to Pa");

				for (auto& val : VEC(PInfo))
				{
					val = exp(val);
				}

				PInfo->SetParam(param("LNSP-PA"));

				auto c = GET_PLUGIN(cache);
				c->Insert(*PInfo, true);

				lnspInfos[key] = PInfo;
			}
		}

		PParam = {param("LNSP-PA")};
		PInfo = Fetch(forecastTime, PLevel, PParam, forecastType, false);
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

	std::vector<double> ab = TInfo->Grid()->AB();

	double A = MissingDouble(), B = MissingDouble();

	if (ab.size() == 2)
	{
		A = ab[0];
		B = ab[1];
	}
	else
	{
		const size_t levelValue = static_cast<size_t>(forecastLevel.Value());
		assert(levelValue <= ab.size());

		A = (ab[levelValue - 1] + ab[levelValue]) * 0.5;

		const size_t halfsize = static_cast<size_t>(static_cast<double>(ab.size()) * 0.5);

		B = (ab[halfsize + levelValue - 1] + ab[halfsize + levelValue]) * 0.5;
	}

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(PInfo)))
	{
		double& result = tup.get<0>();
		double P = tup.get<1>();

		result = 0.01 * (A + P * B);
	}

	myThreadedLogger.Info("[CPU] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                      to_string(myTargetInfo->Data().Size()));
}

void hybrid_pressure::WriteToFile(const info& targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	writeOptions.write_empty_grid = false;

	aWriter->WriteOptions(writeOptions);

	// writing might modify iterator positions --> create a copy

	auto tempInfo = targetInfo;

	tempInfo.ResetParam();

	while (tempInfo.NextParam())
	{
		if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
		{
			aWriter->ToFile(tempInfo, itsConfiguration);
		}
		else
		{
			lock_guard<mutex> lock(mySingleFileWriteMutex);

			aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
		}
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(targetInfo);
	}
}
