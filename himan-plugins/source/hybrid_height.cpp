#include "hybrid_height.h"
#include "logger.h"
#include "plugin_factory.h"
#include <boost/thread.hpp>
#include <future>

#include "cache.h"
#include "radon.h"
#include "util.h"

using namespace std;
using namespace himan::plugin;

const string itsName("hybrid_height");
const himan::param PParam("P-HPA");
const himan::param TParam("T-K");
const himan::param HParam("HL-M");
const himan::param ZParam("Z-M2S2");

hybrid_height::hybrid_height() : itsBottomLevel(kHPMissingInt), itsUseGeopotential(true)
{
	itsLogger = logger(itsName);
}

hybrid_height::~hybrid_height()
{
}
void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	HPDatabaseType dbtype = conf->DatabaseType();

	if (dbtype == kRadon)
	{
		auto r = GET_PLUGIN(radon);

		itsBottomLevel =
		    stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
	}

	if (itsConfiguration->Info()->Producer().Id() == 240 || itsConfiguration->Info()->Producer().Id() == 243)
	{
		itsUseGeopotential = false;

		// Using separate writer threads is only efficient when we are calculating with iteration (ECMWF)
		// and if we are using external packing like gzip or if the grid size is large (several million grid points)
		// In those conditions spawning a separate thread to write the results should give according to initial tests
		// a ~30% increase in total calculation speed.

		PrimaryDimension(kTimeDimension);

		/*
		 * With iteration method, we must start from the lowest level.
		 */

		if (itsInfo->SizeLevels() > 1)
		{
			auto first = itsInfo->PeekLevel(0), second = itsInfo->PeekLevel(1);

			if (first.Value() < second.Value())
			{
				itsInfo->LevelOrder(kBottomToTop);
			}
		}
	}

	SetParams({param(HParam)});

	Start();
}

void hybrid_height::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
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
		myThreadedLogger.Warning("Skipping step " + to_string(myTargetInfo->Time().Step()) + ", level " +
		                         static_cast<string>(myTargetInfo->Level()));
		return;
	}

	myThreadedLogger.Info("[CPU] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                      to_string(myTargetInfo->Data().Size()));
}

bool hybrid_height::WithGeopotential(info_t& myTargetInfo)
{
	const himan::level H0(himan::kHeight, 0);

	auto GPInfo = Fetch(myTargetInfo->Time(), myTargetInfo->Level(), ZParam, myTargetInfo->ForecastType(), false);
	auto zeroGPInfo = Fetch(myTargetInfo->Time(), H0, ZParam, myTargetInfo->ForecastType(), false);

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
		double& result = tup.get<0>();
		const double gp = tup.get<1>();
		const double zerogp = tup.get<2>();

		result = (gp - zerogp) * himan::constants::kIg;
	}

	return true;
}

himan::info_t hybrid_height::GetSurfacePressure(himan::info_t& myTargetInfo)
{
	const auto forecastTime = myTargetInfo->Time();
	const auto forecastType = myTargetInfo->ForecastType();

	info_t ret;

	if (myTargetInfo->Producer().Id() == 240 || myTargetInfo->Producer().Id() == 243)
	{
		// LNSP is always at level 1 for ECMWF

		ret = Fetch(forecastTime, level(himan::kHybrid, 1), param("LNSP-HPA"), forecastType, false);

		if (!ret)
		{
			ret = Fetch(forecastTime, level(himan::kHybrid, 1), param("LNSP-N"), forecastType, false);

			if (ret)
			{
				// LNSP to regular pressure

				auto newInfo = make_shared<info>(*ret);
				newInfo->SetParam(param("LNSP-HPA"));
				newInfo->Create(ret->Grid());

				auto& target = VEC(newInfo);
				for (double& val : target)
				{
					val = 0.01 * exp(val);
					ASSERT(isfinite(val));
				}

				ret = newInfo;
			}
		}
	}
	else
	{
		ret = Fetch(forecastTime, level(himan::kHeight, 0), PParam, forecastType, false);
	}

	return ret;
}

bool hybrid_height::WithHypsometricEquation(info_t& myTargetInfo)
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

	info_t prevPInfo, prevTInfo;

	bool firstLevel;

	if (myTargetInfo->LevelOrder() == kTopToBottom)
	{
		firstLevel = myTargetInfo->PeekLevel(0).Value() == itsBottomLevel;
	}
	else
	{
		firstLevel = myTargetInfo->PeekLevel(myTargetInfo->SizeLevels() - 1).Value() == itsBottomLevel;
	}

	if (firstLevel)
	{
		prevPInfo = GetSurfacePressure(myTargetInfo);

		prevTInfo = Fetch(forecastTime, level(himan::kHeight, 2), TParam, forecastType, false);

		if (!prevTInfo)
		{
			prevTInfo = Fetch(forecastTime, level(himan::kHybrid, itsBottomLevel), TParam, forecastType, false);
		}
	}
	else
	{
		level prevLevel(myTargetInfo->Level());
		prevLevel.Value(myTargetInfo->Level().Value() + 1);

		prevTInfo = Fetch(forecastTime, prevLevel, TParam, forecastType, false);
		prevPInfo = Fetch(forecastTime, prevLevel, PParam, forecastType, false);
	}

	vector<future<void>> pool;

	// First pass

	for (myTargetInfo->ResetLevel(); myTargetInfo->NextLevel();)
	{
		if (itsConfiguration->UseDynamicMemoryAllocation())
		{
			AllocateMemory(*myTargetInfo);
		}

		ASSERT(myTargetInfo->Data().Size() > 0);

		auto PInfo = Fetch(forecastTime, myTargetInfo->Level(), PParam, forecastType, false);
		auto TInfo = Fetch(forecastTime, myTargetInfo->Level(), TParam, forecastType, false);

		if (!prevTInfo || !prevPInfo || !PInfo || !TInfo)
		{
			itsLogger.Error("Source data missing for level " + to_string(myTargetInfo->Level().Value()) + " step " +
			                to_string(myTargetInfo->Time().Step()) + ", stopping processing");
			return false;
		}

		SetAB(myTargetInfo, TInfo);

		// Launch async thread to process the grid
		// We have to give shared_ptr's as arguments to make sure they don't go
		// out of scope and memory free'd while processing is still in progress

		pool.push_back(async(launch::async,
		                     [&](info_t myTargetInfo, info_t PInfo, info_t prevPInfo, info_t TInfo, info_t prevTInfo) {
			                     auto& target = VEC(myTargetInfo);
			                     const auto& PVec = VEC(PInfo);
			                     const auto& prevPVec = VEC(prevPInfo);
			                     const auto& TVec = VEC(TInfo);
			                     const auto& prevTVec = VEC(prevTInfo);

			                     for (auto&& tup : zip_range(target, PVec, prevPVec, TVec, prevTVec))
			                     {
				                     double& result = tup.get<0>();

				                     const double P = tup.get<1>();
				                     const double prevP = tup.get<2>();
				                     const double T = tup.get<3>();
				                     const double prevT = tup.get<4>();

				                     result = 14.628 * (prevT + T) * log(prevP / P);

				                     ASSERT(isfinite(result));
			                     }
			                 },
		                     make_shared<info>(*myTargetInfo), PInfo, prevPInfo, TInfo, prevTInfo));

		if (pool.size() == 8)
		{
			for (auto& f : pool)
			{
				f.get();
			}

			pool.clear();
		}

		prevPInfo = PInfo;
		prevTInfo = TInfo;
	}

	for (auto& f : pool)
	{
		f.get();
	}

	// Second pass

	vector<future<void>> writers;

	for (myTargetInfo->ResetLevel(); myTargetInfo->NextLevel();)
	{
		// Check if we have data in grid. If all values are missing, it is impossible to continue
		// processing any level above this one.
		if (myTargetInfo->Data().Size() == myTargetInfo->Data().MissingCount())
		{
			itsLogger.Error("All data missing for level " + to_string(myTargetInfo->Level().Value()) + " step " +
			                to_string(myTargetInfo->Time().Step()) + ", stopping processing");
			return false;
		}

		if (myTargetInfo->Level().Value() != itsBottomLevel)
		{
			auto& cur = VEC(myTargetInfo);

			if (myTargetInfo->PreviousLevel())
			{
				const auto& prev = VEC(myTargetInfo);

				transform(cur.begin(), cur.end(), prev.begin(), cur.begin(), plus<double>());

				myTargetInfo->NextLevel();
			}
			else
			{
				// First level calculated is not the lowest level
				level prevLevel(myTargetInfo->Level());
				prevLevel.Value(prevLevel.Value() + 1);

				auto prevH = Fetch(forecastTime, prevLevel, HParam, forecastType, false);
				if (!prevH)
				{
					itsLogger.Error("Unable to get height of level below level " +
					                static_cast<string>(myTargetInfo->Level()));
					return false;
				}

				const auto& prev = VEC(prevH);

				transform(cur.begin(), cur.end(), prev.begin(), cur.begin(), plus<double>());
			}
		}

		if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
		{
			writers.push_back(async(launch::async, [this](info_t myTargetInfo) { WriteToFile(myTargetInfo); },
			                        make_shared<info>(*myTargetInfo)));
		}
		else
		{
			WriteToFile(myTargetInfo);
		}

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
		}
	}

	for (auto& f : writers)
	{
		f.get();
	}

	return true;
}

void hybrid_height::RunTimeDimension(info_t myTargetInfo, unsigned short threadIndex)
{
	myTargetInfo->FirstLevel();

	while (NextExcludingLevel(*myTargetInfo))
	{
		Calculate(myTargetInfo, threadIndex);
	}
}
