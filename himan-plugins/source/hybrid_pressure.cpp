/**
 * @file hybrid_pressure.cpp
 *
 *  @date: Mar 23, 2013
 *  @author aaltom
 */

#include "hybrid_pressure.h"
#include "forecast_time.h"
#include "level.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

#include "plugin_factory.h"
#include "util.h"

#include "cache.h"
#include "writer.h"

using namespace std;
using namespace himan::plugin;

mutex lnspMutex, mySingleFileWriteMutex;
map<int, himan::info_t> lnspInfos;

hybrid_pressure::hybrid_pressure()
{
	// Vertkoord_A and Vertkoord_B refer to full hybrid-level coefficients
	itsClearTextFormula = "P = Vertkoord_A + P0 * Vertkoord_B";

	itsLogger = logger_factory::Instance()->GetLog("hybrid_pressure");
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

	bool isECMWF =
	    (itsConfiguration->SourceProducer().Id() == 131);  // Note! This only checks the *current* source producer

	auto myThreadedLogger =
	    logger_factory::Instance()->GetLog("hybrid_pressureThread #" + boost::lexical_cast<string>(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
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

		if (lnspInfos.find(forecastTime.Step()) == lnspInfos.end())
		{
			lock_guard<mutex> lock(lnspMutex);

			if (lnspInfos.find(forecastTime.Step()) == lnspInfos.end())
			{
				PInfo = Fetch(forecastTime, PLevel, PParam, forecastType, false);

				if (!PInfo)
				{
					myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string>(forecastTime.Step()) +
					                          ", level " + static_cast<string>(forecastLevel));
					return;
				}

				myThreadedLogger->Info("Transforming LNSP to Pa");

				for (auto& val : VEC(PInfo))
				{
					val = exp(val);
				}

				PInfo->SetParam(param("LNSP-PA"));

				auto c = GET_PLUGIN(cache);
				c->Insert(*PInfo, true);

				lnspInfos[forecastTime.Step()] = PInfo;
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
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string>(forecastTime.Step()) + ", level " +
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

	double A = ab[0];
	double B = ab[1];

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(PInfo)))
	{
		double& result = tup.get<0>();
		double P = tup.get<1>();

		if (P == kFloatMissing)
		{
			continue;
		}

		result = 0.01 * (A + P * B);
	}

	myThreadedLogger->Info("[CPU] Missing values: " + boost::lexical_cast<string>(myTargetInfo->Data().MissingCount()) +
	                       "/" + boost::lexical_cast<string>(myTargetInfo->Data().Size()));
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
