#include "hybrid_height.h"
#include "logger.h"
#include "plugin_factory.h"
#include <future>

#include "cache.h"
#include "radon.h"
#include "util.h"
#include "writer.h"

using namespace std;
using namespace himan::plugin;

extern mutex singleFileWriteMutex;

const string itsName("hybrid_height");
const himan::param PParam("P-HPA");
const himan::param TParam("T-K");
const himan::param HParam("HL-M");
const himan::param ZParam("Z-M2S2");

hybrid_height::hybrid_height() : itsBottomLevel(kHPMissingInt), itsUseGeopotential(true)
{
	itsLogger = logger(itsName);
}

void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	HPDatabaseType dbtype = conf->DatabaseType();

	if (dbtype == kRadon)
	{
		auto r = GET_PLUGIN(radon);

		itsBottomLevel =
		    stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->TargetProducer().Id(), "last hybrid level number"));
	}

	if ((itsConfiguration->TargetProducer().Id() == 240 || itsConfiguration->TargetProducer().Id() == 243) ||
	    itsConfiguration->TargetProducer().Id() == 270)
	{
		// Workaround for MNWC which doesn't have geopotential for sub-hour data (as of 2019-06-12)
		itsUseGeopotential = false;

		itsThreadDistribution = ThreadDistribution::kThreadForForecastTypeAndTime;
	}

	SetParams({param(HParam)});

	Start<float>();
}

void hybrid_height::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger(itsName + "Thread #" + to_string(threadIndex));

	myThreadedLogger.Info("Calculating time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()) + " level " +
	                      static_cast<string>(myTargetInfo->Level()) + " forecast type " +
	                      static_cast<string>(myTargetInfo->ForecastType()));

	bool ret;

	if (itsUseGeopotential)
	{
		ret = WithGeopotential(myTargetInfo);
	}
	else
	{
		ret = WithHypsometricEquation(myTargetInfo);
	}

	if (!ret)
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(myTargetInfo->Time().Step()) + ", level " +
		                         static_cast<string>(myTargetInfo->Level()));
		return;
	}

	myThreadedLogger.Info("[CPU] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                      to_string(myTargetInfo->Data().Size()));
}

bool hybrid_height::WithGeopotential(shared_ptr<himan::info<float>>& myTargetInfo)
{
	const himan::level H0(himan::kHeight, 0);

	auto GPInfo =
	    Fetch<float>(myTargetInfo->Time(), myTargetInfo->Level(), ZParam, myTargetInfo->ForecastType(), false);
	auto zeroGPInfo = Fetch<float>(myTargetInfo->Time(), H0, ZParam, myTargetInfo->ForecastType(), false);

	if (!GPInfo || !zeroGPInfo)
	{
		return false;
	}

	SetAB(myTargetInfo, GPInfo);

	auto& target = VEC(myTargetInfo);
	const auto& zeroGP = VEC(zeroGPInfo);
	const auto& GP = VEC(GPInfo);

	for (auto&& tup : zip_range(target, GP, zeroGP))
	{
		float& result = tup.get<0>();
		const float gp = tup.get<1>();
		const float zerogp = tup.get<2>();

		result = (gp - zerogp) * static_cast<float>(himan::constants::kIg);
	}

	WriteSingleGridToFile(myTargetInfo);
	return true;
}

shared_ptr<himan::info<float>> hybrid_height::GetSurfacePressure(shared_ptr<himan::info<float>>& myTargetInfo)
{
	const auto forecastTime = myTargetInfo->Time();
	const auto forecastType = myTargetInfo->ForecastType();

	shared_ptr<info<float>> ret;

	if (myTargetInfo->Producer().Id() == 240 || myTargetInfo->Producer().Id() == 243)
	{
		// LNSP is always at level 1 for ECMWF

		ret = Fetch<float>(forecastTime, level(himan::kHybrid, 1), param("LNSP-HPA"), forecastType, false);

		if (!ret)
		{
			ret = Fetch<float>(forecastTime, level(himan::kHybrid, 1), param("LNSP-N"), forecastType, false);

			if (ret)
			{
				// LNSP to regular pressure

				auto newInfo = make_shared<info<float>>(*ret);
				newInfo->Set<param>(param("LNSP-HPA"));
				newInfo->Create(ret->Base());

				auto& target = VEC(newInfo);
				for (float& val : target)
				{
					val = 0.01f * exp(val);
					ASSERT(isfinite(val));
				}

				ret = newInfo;
			}
		}
	}
	else
	{
		ret = Fetch<float>(forecastTime, level(himan::kHeight, 0), param("P-PA"), forecastType, false);

		if (ret)
		{
			auto newInfo = make_shared<info<float>>(*ret);
			newInfo->Set<param>(param("P-HPA"));
			newInfo->Create(ret->Base());

			auto& v = VEC(newInfo);
			transform(v.begin(), v.end(), v.begin(), [](float vv) { return vv * 0.01f; });

			ret = newInfo;
		}
	}

	return ret;
}

bool hybrid_height::WithHypsometricEquation(shared_ptr<himan::info<float>>& myTargetInfo)
{
	/*
	 * Processing is done in two passes:
	 *
	 * First pass: each grid is processes in a multithreaded fashion, only the height of the specific air
	 *             slab is calculated.
	 * Second pass: Each thickness is summed so that we get for each level the height from ground
	 *
	 */

	const auto forecastTime = myTargetInfo->Time();
	const auto forecastType = myTargetInfo->ForecastType();

	shared_ptr<info<float>> prevPInfo, prevTInfo;

	bool firstLevel;

	bool topToBottom = false;

	if (myTargetInfo->Size<level>() > 1)
	{
		auto first = itsLevelIterator.At(0), second = itsLevelIterator.At(1);

		if (first.Value() < second.Value())
		{
			topToBottom = true;
		}
	}

	if (topToBottom == false)
	{
		firstLevel = myTargetInfo->Peek<level>(0).Value() == itsBottomLevel;
	}
	else
	{
		firstLevel = myTargetInfo->Peek<level>(myTargetInfo->Size<level>() - 1).Value() == itsBottomLevel;
	}

	if (firstLevel)
	{
		prevPInfo = GetSurfacePressure(myTargetInfo);

		prevTInfo = Fetch<float>(forecastTime, level(himan::kHeight, 2), TParam, forecastType, false);

		if (!prevTInfo)
		{
			prevTInfo = Fetch<float>(forecastTime, level(himan::kHybrid, itsBottomLevel), TParam, forecastType, false);
		}
	}
	else
	{
		level prevLevel(myTargetInfo->Level());
		prevLevel.Value(myTargetInfo->Level().Value() + 1);

		prevTInfo = Fetch<float>(forecastTime, prevLevel, TParam, forecastType, false);
		prevPInfo = Fetch<float>(forecastTime, prevLevel, PParam, forecastType, false);
	}

	vector<future<void>> pool;

	// First pass

	topToBottom ? myTargetInfo->Last<level>() : myTargetInfo->First<level>();

	while (true)
	{
		if (itsConfiguration->UseDynamicMemoryAllocation())
		{
			AllocateMemory(*myTargetInfo);
		}

		ASSERT(myTargetInfo->Data().Size() > 0);

		auto PInfo = Fetch<float>(forecastTime, myTargetInfo->Level(), PParam, forecastType, false);
		auto TInfo = Fetch<float>(forecastTime, myTargetInfo->Level(), TParam, forecastType, false);

		if (!prevTInfo || !prevPInfo || !PInfo || !TInfo)
		{
			itsLogger.Error("Source data missing for level " + to_string(myTargetInfo->Level().Value()) + " step " +
			                static_cast<string>(myTargetInfo->Time().Step()) + ", stopping processing");
			return false;
		}

		SetAB(myTargetInfo, TInfo);

		// Launch async thread to process the grid
		// We have to give shared_ptr's as arguments to make sure they don't go
		// out of scope and memory free'd while processing is still in progress

		pool.push_back(async(launch::async,
		                     [&](shared_ptr<himan::info<float>> _myTargetInfo, shared_ptr<himan::info<float>> _PInfo,
		                         shared_ptr<himan::info<float>> _prevPInfo, shared_ptr<himan::info<float>> _TInfo,
		                         shared_ptr<himan::info<float>> _prevTInfo) {
			                     auto& target = VEC(_myTargetInfo);
			                     const auto& PVec = VEC(_PInfo);
			                     const auto& prevPVec = VEC(_prevPInfo);
			                     const auto& TVec = VEC(_TInfo);
			                     const auto& prevTVec = VEC(_prevTInfo);

			                     for (auto&& tup : zip_range(target, PVec, prevPVec, TVec, prevTVec))
			                     {
				                     float& result = tup.get<0>();

				                     const float P = tup.get<1>();
				                     const float prevP = tup.get<2>();
				                     const float T = tup.get<3>();
				                     const float prevT = tup.get<4>();

				                     result = 14.628f * (prevT + T) * log(prevP / P);

				                     ASSERT(isfinite(result));
			                     }
		                     },
		                     make_shared<info<float>>(*myTargetInfo), PInfo, prevPInfo, TInfo, prevTInfo));

		if (pool.size() == 10)
		{
			for (auto& f : pool)
			{
				f.get();
			}

			pool.clear();
		}

		prevPInfo = PInfo;
		prevTInfo = TInfo;

		const bool levelsRemaining = topToBottom ? myTargetInfo->Previous<level>() : myTargetInfo->Next<level>();

		if (levelsRemaining == false)
		{
			break;
		}
	}

	for (auto& f : pool)
	{
		f.get();
	}

	// Second pass

	// Using separate writer threads is efficient when we are calculating with iteration (ECMWF) and if
	// we are using external packing like gzip or if the grid size is large (several million grid points)
	// In those conditions spawning a separate thread to write the results should give according to initial tests
	// a ~30%-50% increase in total calculation speed.

	vector<future<void>> writers;

	topToBottom ? myTargetInfo->Last<level>() : myTargetInfo->First<level>();

	while (true)
	{
		// Check if we have data in grid. If all values are missing, it is impossible to continue
		// processing any level above this one.
		if (myTargetInfo->Data().Size() == myTargetInfo->Data().MissingCount())
		{
			itsLogger.Error("All data missing for level " + to_string(myTargetInfo->Level().Value()) + " step " +
			                static_cast<string>(myTargetInfo->Time().Step()) + ", stopping processing");
			return false;
		}

		if (myTargetInfo->Level().Value() != itsBottomLevel)
		{
			auto& cur = VEC(myTargetInfo);

			bool isFirst = topToBottom ? myTargetInfo->Next<level>() : myTargetInfo->Previous<level>();

			if (isFirst)
			{
				const auto& prev = VEC(myTargetInfo);

				transform(cur.begin(), cur.end(), prev.begin(), cur.begin(), plus<float>());

				topToBottom ? myTargetInfo->Previous<level>() : myTargetInfo->Next<level>();
			}
			else
			{
				// First level calculated is not the lowest level
				level prevLevel(myTargetInfo->Level());
				prevLevel.Value(prevLevel.Value() + 1);

				auto prevH = Fetch<float>(forecastTime, prevLevel, HParam, forecastType, false);
				if (!prevH)
				{
					itsLogger.Error("Unable to get height of level below level " +
					                static_cast<string>(myTargetInfo->Level()));
					return false;
				}

				const auto& prev = VEC(prevH);

				transform(cur.begin(), cur.end(), prev.begin(), cur.begin(), plus<float>());
			}
		}

		if (itsConfiguration->WriteMode() == kSingleGridToAFile)
		{
			writers.push_back(async(launch::async,
			                        [this](shared_ptr<info<float>> tempInfo) { WriteSingleGridToFile(tempInfo); },
			                        make_shared<info<float>>(*myTargetInfo)));
		}
		else
		{
			WriteSingleGridToFile(myTargetInfo);
		}

		const bool levelsRemaining = topToBottom ? myTargetInfo->Previous<level>() : myTargetInfo->Next<level>();

		if (levelsRemaining == false)
		{
			break;
		}
	}

	for (auto& f : writers)
	{
		f.get();
	}

	return true;
}

void hybrid_height::WriteToFile(const shared_ptr<info<float>> targetInfo, write_options writeOptions)
{
}

void hybrid_height::WriteSingleGridToFile(const shared_ptr<info<float>> targetInfo)
{
	auto aWriter = GET_PLUGIN(writer);

	if (!targetInfo->IsValidGrid())
	{
		return;
	}

	if (itsConfiguration->WriteMode() == kSingleGridToAFile)
	{
		aWriter->ToFile<float>(targetInfo, itsConfiguration);
	}
	else
	{
		lock_guard<mutex> lock(singleFileWriteMutex);

		aWriter->ToFile<float>(targetInfo, itsConfiguration);
	}
}
