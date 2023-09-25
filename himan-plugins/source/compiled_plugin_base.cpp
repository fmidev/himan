/**
 *
 * @file compiled_plugin_base.cpp
 *
 */

#include "compiled_plugin_base.h"
#include "cuda_helper.h"
#include "logger.h"
#include "plugin_factory.h"
#include "statistics.h"
#include "util.h"
#include <mutex>
#include <thread>

#include "cache.h"
#include "fetcher.h"
#include "radon.h"
#include "writer.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

mutex dimensionMutex, writeStatusMutex;

template <typename T>
bool compiled_plugin_base::Next(info<T>& myTargetInfo)
{
	lock_guard<mutex> lock(dimensionMutex);

	if (!itsDimensionsRemaining)
	{
		return false;
	}

	if (itsThreadDistribution == ThreadDistribution::kThreadForAny ||
	    itsThreadDistribution == ThreadDistribution::kThreadForForecastTypeAndLevel ||
	    itsThreadDistribution == ThreadDistribution::kThreadForTimeAndLevel ||
	    itsThreadDistribution == ThreadDistribution::kThreadForLevel)
	{
		if (itsLevelIterator.Next())
		{
			bool ret = myTargetInfo.template Find<level>(itsLevelIterator.At());
			ASSERT(ret);
			ret = myTargetInfo.template Find<forecast_time>(itsTimeIterator.At());
			ASSERT(ret);
			ret = myTargetInfo.template Find<forecast_type>(itsForecastTypeIterator.At());
			ASSERT(ret);

			return ret;
		}

		// No more levels at this forecast type/time combination; rewind level iterator

		itsLevelIterator.First();
	}

	if (itsThreadDistribution == ThreadDistribution::kThreadForAny ||
	    itsThreadDistribution == ThreadDistribution::kThreadForForecastTypeAndTime ||
	    itsThreadDistribution == ThreadDistribution::kThreadForTimeAndLevel ||
	    itsThreadDistribution == ThreadDistribution::kThreadForTime)
	{
		if (itsTimeIterator.Next())
		{
			bool ret = myTargetInfo.template Find<forecast_time>(itsTimeIterator.At());
			ASSERT(ret);
			ret = myTargetInfo.template Find<level>(itsLevelIterator.At());
			ASSERT(ret);
			ret = myTargetInfo.template Find<forecast_type>(itsForecastTypeIterator.At());
			ASSERT(ret);

			return ret;
		}

		// No more times at this forecast type; rewind time iterator, level iterator is
		// already at first place

		itsTimeIterator.First();
	}

	if (itsThreadDistribution == ThreadDistribution::kThreadForAny ||
	    itsThreadDistribution == ThreadDistribution::kThreadForForecastTypeAndTime ||
	    itsThreadDistribution == ThreadDistribution::kThreadForForecastTypeAndLevel ||
	    itsThreadDistribution == ThreadDistribution::kThreadForForecastType)
	{
		if (itsForecastTypeIterator.Next())
		{
			bool ret = myTargetInfo.template Find<forecast_time>(itsTimeIterator.At());
			ASSERT(ret);
			ret = myTargetInfo.template Find<level>(itsLevelIterator.At());
			ASSERT(ret);
			ret = myTargetInfo.template Find<forecast_type>(itsForecastTypeIterator.At());
			ASSERT(ret);

			return ret;
		}
	}
	// future threads calling for new dimensions aren't getting any

	itsDimensionsRemaining = false;

	return false;
}

template bool compiled_plugin_base::Next<double>(info<double>&);
template bool compiled_plugin_base::Next<float>(info<float>&);

bool compiled_plugin_base::SetAB(const shared_ptr<info<double>>& myTargetInfo,
                                 const shared_ptr<info<double>>& sourceInfo)
{
	return SetAB<double>(myTargetInfo, sourceInfo);
}

template <typename T>
bool compiled_plugin_base::SetAB(const shared_ptr<info<T>>& myTargetInfo, const shared_ptr<info<T>>& sourceInfo)
{
	if (myTargetInfo->template Level().Type() == kHybrid)
	{
		const size_t paramIndex = myTargetInfo->template Index<param>();

		for (myTargetInfo->template Reset<param>(); myTargetInfo->template Next<param>();)
		{
			myTargetInfo->Level().AB(sourceInfo->Level().AB());
		}

		myTargetInfo->template Index<param>(paramIndex);
	}

	return true;
}

template bool compiled_plugin_base::SetAB<double>(const shared_ptr<info<double>>&, const shared_ptr<info<double>>&);
template bool compiled_plugin_base::SetAB<float>(const shared_ptr<info<float>>&, const shared_ptr<info<float>>&);

void compiled_plugin_base::WriteToFile(const shared_ptr<info<double>> targetInfo, write_options writeOptions)
{
	return WriteToFile<double>(targetInfo, writeOptions);
}

void compiled_plugin_base::WriteToFile(const shared_ptr<info<float>> targetInfo, write_options writeOptions)
{
	return WriteToFile<float>(targetInfo, writeOptions);
}

template <typename T>
void compiled_plugin_base::WriteToFile(const shared_ptr<info<T>> targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);

	// writing might modify iterator positions --> create a copy

	auto tempInfo = make_shared<info<T>>(*targetInfo);

	auto WriteParam = [&]()
	{
		tempInfo->template Reset<param>();

		while (tempInfo->template Next<param>())
		{
			if (!tempInfo->IsValidGrid())
			{
				continue;
			}

			const HPWriteStatus status = aWriter->ToFile(tempInfo, itsConfiguration);
			const std::string uName = util::UniqueName<T>(*tempInfo);

			// status of all writes is recorded, although currently we are
			// only interested of pending writes

			{
				lock_guard<mutex> lock(writeStatusMutex);
				itsWriteStatuses.push_back(make_pair(uName, status));
			}
		}
	};

	// TODO: should work for all thread distribution types
	switch (itsThreadDistribution)
	{
		case ThreadDistribution::kThreadForForecastTypeAndTime:
			tempInfo->template Reset<level>();

			while (tempInfo->template Next<level>())
			{
				WriteParam();
			}
			break;

		case ThreadDistribution::kThreadForForecastTypeAndLevel:
			tempInfo->template Reset<forecast_time>();

			while (tempInfo->template Next<forecast_time>())
			{
				WriteParam();
			}
			break;

		case ThreadDistribution::kThreadForTimeAndLevel:
			tempInfo->template Reset<forecast_type>();

			while (tempInfo->template Next<forecast_type>())
			{
				WriteParam();
			}
			break;

		case ThreadDistribution::kThreadForForecastType:
			for (tempInfo->template Reset<forecast_time>(); tempInfo->template Next<forecast_time>();)
			{
				for (tempInfo->template Reset<level>(); tempInfo->template Next<level>();)
				{
					WriteParam();
				}
			}
			break;

		case ThreadDistribution::kThreadForTime:
			for (tempInfo->template Reset<forecast_type>(); tempInfo->template Next<forecast_type>();)
			{
				for (tempInfo->template Reset<level>(); tempInfo->template Next<level>();)
				{
					WriteParam();
				}
			}
			break;

		case ThreadDistribution::kThreadForLevel:
			for (tempInfo->template Reset<forecast_type>(); tempInfo->template Next<forecast_type>();)
			{
				for (tempInfo->template Reset<forecast_time>(); tempInfo->template Next<forecast_time>();)
				{
					WriteParam();
				}
			}
			break;

		case ThreadDistribution::kThreadForAny:
			WriteParam();
			break;
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(*targetInfo);
	}
}

template void compiled_plugin_base::WriteToFile<double>(const shared_ptr<info<double>>, write_options);
template void compiled_plugin_base::WriteToFile<float>(const shared_ptr<info<float>>, write_options);

void compiled_plugin_base::Start()
{
	return Start<double>();
}

void compiled_plugin_base::SetInitialIteratorPositions()
{
	itsParamIterator.First();

	switch (itsThreadDistribution)
	{
		case ThreadDistribution::kThreadForAny:
			itsLevelIterator.Reset();
			itsTimeIterator.First();
			itsForecastTypeIterator.First();
			break;
		case ThreadDistribution::kThreadForForecastTypeAndTime:
			itsLevelIterator.First();  // dimension ignored by Next(), set for convenience
			itsTimeIterator.Reset();
			itsForecastTypeIterator.First();
			break;
		case ThreadDistribution::kThreadForForecastTypeAndLevel:
		case ThreadDistribution::kThreadForTimeAndLevel:
			itsLevelIterator.Reset();
			itsTimeIterator.First();
			itsForecastTypeIterator.First();
			break;
		case ThreadDistribution::kThreadForLevel:
			itsLevelIterator.Reset();
			itsTimeIterator.First();
			itsForecastTypeIterator.First();
			break;
		case ThreadDistribution::kThreadForForecastType:
			itsLevelIterator.First();
			itsTimeIterator.First();
			itsForecastTypeIterator.Reset();
			break;
		case ThreadDistribution::kThreadForTime:
			itsLevelIterator.First();
			itsTimeIterator.Reset();
			itsForecastTypeIterator.First();
			break;
	}
}

void compiled_plugin_base::SetThreadCount()
{
	const auto ftypes = itsForecastTypeIterator.Size();
	const auto times = itsTimeIterator.Size();
	const auto lvls = itsLevelIterator.Size();

	size_t dims = 12;

	switch (itsThreadDistribution)
	{
		case ThreadDistribution::kThreadForAny:
			dims = ftypes * times * lvls;
			break;
		case ThreadDistribution::kThreadForForecastTypeAndTime:
			dims = ftypes * times;
			break;
		case ThreadDistribution::kThreadForForecastTypeAndLevel:
			dims = ftypes * lvls;
			break;
		case ThreadDistribution::kThreadForTimeAndLevel:
			dims = times * lvls;
			break;
		case ThreadDistribution::kThreadForLevel:
			dims = lvls;
			break;
		case ThreadDistribution::kThreadForForecastType:
			dims = ftypes;
			break;
		case ThreadDistribution::kThreadForTime:
			dims = times;
			break;
	}

	const auto cnfCount = itsConfiguration->ThreadCount();
	itsThreadCount = (cnfCount == -1) ? static_cast<short>(std::min(12, static_cast<int>(dims))) : cnfCount;
	itsConfiguration->Statistics()->UsedThreadCount(itsThreadCount);
}

template <typename T>
std::string TypeToName();

template <>
string TypeToName<double>()
{
	return string("double");
}

template <>
string TypeToName<float>()
{
	return string("float");
}

template <typename T>
void compiled_plugin_base::Start()
{
	if (!itsPluginIsInitialized)
	{
		itsBaseLogger.Error("Start() called before Init()");
		return;
	}

	/*
	 * From the timing perspective at this point plugin initialization is
	 * considered to be done
	 */

	if (itsConfiguration->StatisticsEnabled())
	{
		itsConfiguration->Statistics()->UsedThreadCount(itsThreadCount);
		itsTimer.Stop();
		itsConfiguration->Statistics()->AddToInitTime(itsTimer.GetTime());
		// Start process timing
		itsTimer.Start();
	}

	auto baseInfo = make_shared<info<T>>(itsForecastTypeIterator.Values(), itsTimeIterator.Values(),
	                                     itsLevelIterator.Values(), itsParamIterator.Values());
	baseInfo->Producer(itsConfiguration->TargetProducer());

	baseInfo->template First<forecast_type>();
	baseInfo->template First<forecast_time>();
	baseInfo->template First<level>();
	baseInfo->template Reset<param>();

	const auto gr = itsConfiguration->BaseGrid();

	while (baseInfo->Next())
	{
		for (const auto& both : itsLevelParams)
		{
			if (baseInfo->Param() == both.second && baseInfo->Level() == both.first)
			{
				auto b = make_shared<base<T>>();
				b->grid = shared_ptr<grid>(gr->Clone());

				if (itsConfiguration->UseDynamicMemoryAllocation() == false)
				{
					if (b->grid->Class() == kRegularGrid)
					{
						const regular_grid* regGrid(dynamic_cast<const regular_grid*>(b->grid.get()));
						b->data.Resize(regGrid->Ni(), regGrid->Nj());
					}
					else if (b->grid->Class() == kIrregularGrid)
					{
						b->data.Resize(b->grid->Size(), 1, 1);
					}
				}
				baseInfo->Base(b);
			}
		}
	}

	SetThreadCount();
	SetInitialIteratorPositions();

	itsBaseLogger.Trace("Plugin is using data type: " + TypeToName<T>());

	vector<thread> threads;

	for (short i = 0; i < itsThreadCount; i++)
	{
		itsBaseLogger.Info("Thread " + to_string(i) + " starting");
		threads.emplace_back(thread(&compiled_plugin_base::Run<T>, this, make_shared<info<T>>(*baseInfo), i + 1));
	}

	for (auto& t : threads)
	{
		t.join();
	}

	Finish();
}

template void compiled_plugin_base::Start<double>();
template void compiled_plugin_base::Start<float>();

void compiled_plugin_base::Init(const shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = conf;

	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer.Start();
	}

	itsForecastTypeIterator = forecast_type_iter(itsConfiguration->ForecastTypes());
	itsTimeIterator = time_iter(itsConfiguration->Times());
	itsLevelIterator = level_iter(itsConfiguration->Levels());

	itsPluginIsInitialized = true;
}

template <typename T>
void compiled_plugin_base::Run(shared_ptr<info<T>> myTargetInfo, unsigned short threadIndex)
{
	while (Next(*myTargetInfo))
	{
		myTargetInfo->FirstValidGrid();

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

		WriteToFile(myTargetInfo);
	}
}

template void compiled_plugin_base::Run<double>(shared_ptr<info<double>>, unsigned short);
template void compiled_plugin_base::Run<float>(shared_ptr<info<float>>, unsigned short);

void compiled_plugin_base::Finish()
{
	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer.Stop();
		itsConfiguration->Statistics()->AddToProcessingTime(itsTimer.GetTime());
	}

	// Check if we have pending writes in write statuses

	vector<string> pending;

	util::transform_if(
	    itsWriteStatuses.cbegin(), itsWriteStatuses.cend(), back_inserter(pending),
	    [](const pair<string, HPWriteStatus>& element) { return element.second == HPWriteStatus::kPending; },
	    [](const pair<string, HPWriteStatus>& element) { return element.first; });

	itsBaseLogger.Trace(fmt::format("Pending write status for {} infos", pending.size()));

	if (pending.empty() == false)
	{
		writer::AddToPending(pending);
	}
}

void compiled_plugin_base::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	itsBaseLogger.Fatal("Top level Calculate<double>() called");
	exit(1);
}

void compiled_plugin_base::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	itsBaseLogger.Fatal("Top level Calculate<float>() called");
	exit(1);
}

void compiled_plugin_base::SetParams(initializer_list<param> params, bool paramInformationExists)
{
	vector<param> paramVec;

	for (auto it = params.begin(); it != params.end(); ++it)
	{
		paramVec.push_back(*it);
	}

	SetParams(paramVec, paramInformationExists);
}

void compiled_plugin_base::SetParams(initializer_list<param> params, initializer_list<level> levels,
                                     bool paramInformationExists)
{
	vector<param> paramVec(params);
	vector<level> levelVec(levels);

	SetParams(paramVec, levelVec, paramInformationExists);
}

void compiled_plugin_base::SetParams(std::vector<param>& params, const vector<level>& levels,
                                     bool paramInformationExists)
{
	if (params.empty())
	{
		itsBaseLogger.Fatal("Size of target parameter vector is zero");
		himan::Abort();
	}

	if (levels.empty())
	{
		itsBaseLogger.Fatal("Size of target level vector is zero");
		himan::Abort();
	}

	if (itsConfiguration->DatabaseType() == kRadon && paramInformationExists == false)
	{
		auto r = GET_PLUGIN(radon);

		for (const auto& lvl : levels)
		{
			for (auto& par : params)
			{
				if (par.Name() == "DUMMY")
				{
					// special placeholder parameter which is replaced later
					continue;
				}

				par = util::GetParameterInfoFromDatabaseName(itsConfiguration->TargetProducer(), par, lvl);
			}
		}
	}

	// Create a vector that contains a union of current levels and new levels
	vector<level> alllevels;

	for (itsLevelIterator.Reset(); itsLevelIterator.Next();)
	{
		alllevels.push_back(itsLevelIterator.At());
	}

	for (const auto& lvl : levels)
	{
		if (find(alllevels.begin(), alllevels.end(), lvl) == alllevels.end())
		{
			alllevels.push_back(lvl);
		}
	}

	// Create a vector that contains a union of current params and new params
	vector<param> allparams;

	for (itsParamIterator.Reset(); itsParamIterator.Next();)
	{
		allparams.push_back(itsParamIterator.At());
	}

	for (const auto p : params)
	{
		if (find(allparams.begin(), allparams.end(), p) == allparams.end())
		{
			allparams.push_back(p);
		}
	}

	if (itsLevelIterator.Size() < alllevels.size())
	{
		itsLevelIterator = level_iter(alllevels);
	}

	if (itsParamIterator.Size() < allparams.size())
	{
		itsParamIterator = param_iter(allparams);
	}

	if (!itsConfiguration->UseDynamicMemoryAllocation())
	{
		itsBaseLogger.Trace("Using static memory allocation");
	}
	else
	{
		itsBaseLogger.Trace("Using dynamic memory allocation");
	}

	for (const auto& l : levels)
	{
		for (const auto& p : params)
		{
			itsLevelParams.push_back(make_pair(l, p));
		}
	}
}

void compiled_plugin_base::SetParams(std::vector<param>& params, bool paramInformationExists)
{
	vector<level> levels;

	for (size_t i = 0; i < itsLevelIterator.Size(); i++)
	{
		levels.push_back(itsLevelIterator.At(i));
	}

	SetParams(params, levels, paramInformationExists);
}

bool compiled_plugin_base::IsMissingValue(initializer_list<double> values) const
{
	return IsMissingValue<double>(values);
}

template <typename T>
bool compiled_plugin_base::IsMissingValue(initializer_list<T> values) const
{
	for (auto it = values.begin(); it != values.end(); ++it)
	{
		if (IsMissing(*it))
		{
			return true;
		}
	}

	return false;
}

template bool compiled_plugin_base::IsMissingValue<double>(initializer_list<double>) const;
template bool compiled_plugin_base::IsMissingValue<float>(initializer_list<float>) const;

shared_ptr<info<double>> compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel,
                                                     const params& theParams, const forecast_type& theType,
                                                     bool returnPacked) const
{
	return Fetch<double>(theTime, theLevel, theParams, theType, returnPacked);
}

template <typename T>
shared_ptr<info<T>> compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel,
                                                const params& theParams, const forecast_type& theType,
                                                bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);

	shared_ptr<info<T>> ret;

	try
	{
		/*
		 * Fetching of packed data is quite convoluted:
		 *
		 * 1) Fetch packed data iff cuda unpacking is enabled (UseCudaForPacking() == true): it makes no sense
		 * to unpack the data in himan with CPU. If we allow fetcher to return packed data, it will implicitly
		 * disable cache integration of fetched data.
		 *
		 * 2a) If caller does not want packed data (returnPacked == false), unpack it here and insert to cache.
		 *
		 * 2b) If caller wants packed data, return data as-is and leave cache integration to caller.
		 */

		ret =
		    f->Fetch<T>(itsConfiguration, theTime, theLevel, theParams, theType, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack<T>({ret}, itsConfiguration->UseCacheForReads());
		}
#endif
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
		}
	}

	return ret;
}

template shared_ptr<info<double>> compiled_plugin_base::Fetch<double>(const forecast_time&, const level&, const params&,
                                                                      const forecast_type&, bool) const;
template shared_ptr<info<float>> compiled_plugin_base::Fetch<float>(const forecast_time&, const level&, const params&,
                                                                    const forecast_type&, bool) const;

shared_ptr<info<double>> compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel,
                                                     const param& theParam, const forecast_type& theType,
                                                     bool returnPacked) const
{
	return Fetch<double>(theTime, theLevel, theParam, theType, returnPacked);
}

template <typename T>
shared_ptr<info<T>> compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel,
                                                const param& theParam, const forecast_type& theType,
                                                bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);

	shared_ptr<info<T>> ret;

	try
	{
		ret =
		    f->Fetch<T>(itsConfiguration, theTime, theLevel, theParam, theType, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack<T>({ret}, itsConfiguration->UseCacheForReads());
		}
#endif
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
		}
	}

	return ret;
}

template shared_ptr<info<double>> compiled_plugin_base::Fetch<double>(const forecast_time&, const level&, const param&,
                                                                      const forecast_type&, bool) const;
template shared_ptr<info<float>> compiled_plugin_base::Fetch<float>(const forecast_time&, const level&, const param&,
                                                                    const forecast_type&, bool) const;

template <typename T>
shared_ptr<info<T>> compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel,
                                                const param& theParam, const forecast_type& theType,
                                                const vector<string>& geomNames, const producer& sourceProd,
                                                bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);

	shared_ptr<info<T>> ret = nullptr;

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);

	cnf->SourceGeomNames(geomNames);
	cnf->SourceProducers({sourceProd});

	try
	{
		ret = f->Fetch<T>(cnf, theTime, theLevel, theParam, theType, cnf->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack<T>({ret}, cnf->UseCacheForReads());
		}
#endif
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
		}
	}

	return ret;
}

template shared_ptr<info<double>> compiled_plugin_base::Fetch<double>(const forecast_time& e, const level&,
                                                                      const param&, const forecast_type&,
                                                                      const vector<string>&, const producer&,
                                                                      bool) const;

template shared_ptr<info<float>> compiled_plugin_base::Fetch<float>(const forecast_time& e, const level&, const param&,
                                                                    const forecast_type&, const vector<string>&,
                                                                    const producer&, bool) const;

shared_ptr<info<double>> compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel,
                                                     const param& theParam, const forecast_type& theType,
                                                     const vector<string>& geomNames, const producer& sourceProd,
                                                     bool returnPacked) const
{
	return Fetch<double>(theTime, theLevel, theParam, theType, geomNames, sourceProd, returnPacked);
}

template <typename T>
void compiled_plugin_base::AllocateMemory(info<T> myTargetInfo)
{
	size_t paramIndex = myTargetInfo.template Index<param>();

	for (myTargetInfo.template Reset<param>(); myTargetInfo.template Next<param>();)
	{
		if (myTargetInfo.IsValidGrid())
		{
			if (myTargetInfo.Grid()->Class() == kRegularGrid)
			{
				myTargetInfo.Data().Resize(dynamic_pointer_cast<regular_grid>(myTargetInfo.Grid())->Ni(),
				                           dynamic_pointer_cast<regular_grid>(myTargetInfo.Grid())->Nj());
			}
			else
			{
				myTargetInfo.Data().Resize(myTargetInfo.Grid()->Size(), 1);
			}
		}
	}

	myTargetInfo.template Index<param>(paramIndex);
}

template void compiled_plugin_base::AllocateMemory<double>(info<double>);
template void compiled_plugin_base::AllocateMemory<float>(info<float>);

template <typename T>
void compiled_plugin_base::DeallocateMemory(info<T> myTargetInfo)
{
	size_t paramIndex = myTargetInfo.template Index<param>();

	for (myTargetInfo.template Reset<param>(); myTargetInfo.template Next<param>();)
	{
		if (myTargetInfo.IsValidGrid())
		{
			myTargetInfo.Data().Clear();
		}
	}

	myTargetInfo.template Index<param>(paramIndex);
}

template void compiled_plugin_base::DeallocateMemory<double>(info<double>);
template void compiled_plugin_base::DeallocateMemory<float>(info<float>);
