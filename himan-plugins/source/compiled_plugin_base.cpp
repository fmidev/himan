/**
 *
 * @file compiled_plugin_base.cpp
 *
 */

#include "compiled_plugin_base.h"
#include "cuda_helper.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <boost/thread.hpp>

#include "cache.h"
#include "fetcher.h"
#include "radon.h"
#include "writer.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

mutex dimensionMutex, singleFileWriteMutex;

compiled_plugin_base::compiled_plugin_base()
    : itsTimer(),
      itsThreadCount(-1),
      itsDimensionsRemaining(true),
      itsBaseLogger(logger("compiled_plugin_base")),
      itsPluginIsInitialized(false),
      itsPrimaryDimension(kUnknownDimension)
{
}

bool compiled_plugin_base::Next(info<double>& myTargetInfo)
{
	lock_guard<mutex> lock(dimensionMutex);

	if (!itsDimensionsRemaining)
	{
		return false;
	}

	if (itsInfo->Next<level>())
	{
		bool ret = myTargetInfo.Find<level>(itsInfo->Level());
		ASSERT(ret);
		ret = myTargetInfo.Find<forecast_time>(itsInfo->Time());
		ASSERT(ret);
		ret = myTargetInfo.Find<forecast_type>(itsInfo->ForecastType());
		ASSERT(ret);

		return ret;
	}

	// No more levels at this forecast type/time combination; rewind level iterator

	itsInfo->First<level>();

	if (itsInfo->Next<forecast_time>())
	{
		bool ret = myTargetInfo.Find<forecast_time>(itsInfo->Time());
		ASSERT(ret);
		ret = myTargetInfo.Find<level>(itsInfo->Level());
		ASSERT(ret);
		ret = myTargetInfo.Find<forecast_type>(itsInfo->ForecastType());
		ASSERT(ret);

		return ret;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	itsInfo->First<forecast_time>();

	if (itsInfo->Next<forecast_type>())
	{
		bool ret = myTargetInfo.Find<forecast_time>(itsInfo->Time());
		ASSERT(ret);
		ret = myTargetInfo.Find<level>(itsInfo->Level());
		ASSERT(ret);
		ret = myTargetInfo.Find<forecast_type>(itsInfo->ForecastType());
		ASSERT(ret);

		return ret;
	}

	// future threads calling for new dimensions aren't getting any

	itsDimensionsRemaining = false;

	return false;
}

bool compiled_plugin_base::NextExcludingLevel(info<double>& myTargetInfo)
{
	lock_guard<mutex> lock(dimensionMutex);

	if (!itsDimensionsRemaining)
	{
		return false;
	}

	if (itsInfo->Next<forecast_time>())
	{
		bool ret = myTargetInfo.Find<forecast_time>(itsInfo->Time());
		ASSERT(ret);
		ret = myTargetInfo.Find<forecast_type>(itsInfo->ForecastType());
		ASSERT(ret);

		return ret;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	itsInfo->First<forecast_time>();

	if (itsInfo->Next<forecast_type>())
	{
		bool ret = myTargetInfo.Find<forecast_time>(itsInfo->Time());
		ASSERT(ret);
		ret = myTargetInfo.Find<forecast_type>(itsInfo->ForecastType());
		ASSERT(ret);

		return ret;
	}

	// future threads calling for new dimensions aren't getting any

	itsDimensionsRemaining = false;

	return false;
}

bool compiled_plugin_base::SetAB(const info_t& myTargetInfo, const info_t& sourceInfo)
{
	if (myTargetInfo->Level().Type() == kHybrid)
	{
		const size_t paramIndex = myTargetInfo->Index<param>();

		for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>();)
		{
        	        myTargetInfo->Set<level>(sourceInfo->Level());
		}

		myTargetInfo->Index<param>(paramIndex);
	}

	return true;
}

void compiled_plugin_base::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);

	// writing might modify iterator positions --> create a copy

	auto tempInfo = make_shared<info<double>>(*targetInfo);

	tempInfo->Reset<param>();

	while (tempInfo->Next<param>())
	{
		if (!tempInfo->IsValidGrid())
		{
			continue;
		}

		if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
		{
			aWriter->ToFile(tempInfo, itsConfiguration);
		}
		else
		{
			lock_guard<mutex> lock(singleFileWriteMutex);

			aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
		}
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(*targetInfo);
	}
}

void compiled_plugin_base::Start()
{
	if (!itsPluginIsInitialized)
	{
		itsBaseLogger.Error("Start() called before Init()");
		return;
	}

	if (itsPrimaryDimension == kTimeDimension)
	{
		itsInfo->First<forecast_type>();
	}

	boost::thread_group g;

	for (short i = 0; i < itsThreadCount; i++)
	{
		printf("Info::compiled_plugin: Thread %d starting\n", (i + 1));  // Printf is thread safe
		boost::thread* t = new boost::thread(&compiled_plugin_base::Run, this, i + 1);

		g.add_thread(t);
	}

	g.join_all();

	Finish();
}

void compiled_plugin_base::Init(const shared_ptr<const plugin_configuration> conf)
{
	const short MAX_THREADS = 12;  //<! Max number of threads we allow

	itsConfiguration = conf;

	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer.Start();
		itsConfiguration->Statistics()->UsedGPUCount(static_cast<short>(conf->CudaDeviceCount()));
	}

	// Determine thread count

	short coreCount = static_cast<short>(boost::thread::hardware_concurrency());  // Number of cores

	itsThreadCount = MAX_THREADS;

	// If user has specified thread count, always use that
	if (conf->ThreadCount() > 0)
	{
		itsThreadCount = conf->ThreadCount();
	}
	// we don't want to use all cores in a server by default
	else if (MAX_THREADS > coreCount)
	{
		itsThreadCount = coreCount;
	}

	itsInfo = make_shared<info<double>>(itsConfiguration->ForecastTypes(), itsConfiguration->Times(),
	                                    itsConfiguration->Levels(), vector<param>());
	itsInfo->Producer(itsConfiguration->TargetProducer());

	itsPluginIsInitialized = true;
}

void compiled_plugin_base::RunAll(info_t myTargetInfo, unsigned short threadIndex)
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

void compiled_plugin_base::RunTimeDimension(info_t myTargetInfo, unsigned short threadIndex)
{
	while (NextExcludingLevel(*myTargetInfo))
	{
		for (myTargetInfo->Reset<level>(); myTargetInfo->Next<level>();)
		{
			myTargetInfo->FirstValidGrid();

			if (itsConfiguration->UseDynamicMemoryAllocation())
			{
				AllocateMemory(*myTargetInfo);
			}

			Calculate(myTargetInfo, threadIndex);

			if (itsConfiguration->StatisticsEnabled())
			{
				itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
				itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
			}

			WriteToFile(myTargetInfo);
		}
	}
}

void compiled_plugin_base::Run(unsigned short threadIndex)
{
	auto myTargetInfo = make_shared<info<double>>(*itsInfo);
	if (itsPrimaryDimension == kUnknownDimension)
	{
		// The general case: all elements are distributed to all threads in an
		// equal fashion with no dependencies.

		// This method is faster than any of the dimension variations or Run()

		RunAll(myTargetInfo, threadIndex);
	}
	else if (itsPrimaryDimension == kTimeDimension)
	{
		// Each thread will get one time and process that.
		// This is used when f.ex. levels need to be processed
		// in sequential order.

		RunTimeDimension(myTargetInfo, threadIndex);
	}
	else
	{
		itsBaseLogger.Fatal("Invalid primary dimension: " + HPDimensionTypeToString.at(itsPrimaryDimension));
		exit(1);
	}
}

void compiled_plugin_base::Finish()
{
	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer.Stop();
		itsConfiguration->Statistics()->AddToProcessingTime(itsTimer.GetTime());
	}
}

void compiled_plugin_base::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	itsBaseLogger.Fatal("Top level calculate called");
	exit(1);
}

void compiled_plugin_base::SetParams(initializer_list<param> params)
{
	vector<param> paramVec;

	for (auto it = params.begin(); it != params.end(); ++it)
	{
		paramVec.push_back(*it);
	}

	SetParams(paramVec);
}

void compiled_plugin_base::SetParams(initializer_list<param> params, initializer_list<level> levels)
{
	vector<param> paramVec(params);
	vector<level> levelVec(levels);

	SetParams(paramVec, levelVec);
}

void compiled_plugin_base::SetParams(std::vector<param>& params, const vector<level>& levels)
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

	if (itsConfiguration->DatabaseType() == kRadon)
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

				auto levelInfo =
				    r->RadonDB().GetLevelFromDatabaseName(boost::to_upper_copy(HPLevelTypeToString.at(lvl.Type())));

				if (levelInfo.empty())
				{
					itsBaseLogger.Warning("Level type '" + HPLevelTypeToString.at(lvl.Type()) +
					                      "' not found from radon");
					continue;
				}

				auto paraminfo = r->RadonDB().GetParameterFromDatabaseName(itsInfo->Producer().Id(), par.Name(),
				                                                           lvl.Type(), lvl.Value());

				if (paraminfo.empty())
				{
					itsBaseLogger.Warning("Parameter '" + par.Name() + "' definition not found from Radon");
					continue;
				}

				param p(paraminfo);
				p.Aggregation(par.Aggregation());

				par = p;
			}
		}
	}

	// Create a vector that contains a union of current levels and new levels
	vector<level> alllevels;

	for (itsInfo->Reset<level>(); itsInfo->Next<level>();)
	{
		alllevels.push_back(itsInfo->Level());
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

	for (itsInfo->Reset<param>(); itsInfo->Next<param>();)
	{
		allparams.push_back(itsInfo->Param());
	}

	for (const auto p : params)
	{
		if (find(allparams.begin(), allparams.end(), p) == allparams.end())
		{
			allparams.push_back(p);
		}
	}

	if (itsInfo->Size<level>() < alllevels.size())
	{
		itsInfo->Set<level>(alllevels);
	}

	if (itsInfo->Size<param>() < allparams.size())
	{
		itsInfo->Set<param>(allparams);
	}

	/*
	 * Create data structures.
	 */

	if (itsInfo->DimensionSize() == 0)
	{
		itsInfo->Dimensions().resize(itsInfo->Size<forecast_type>() * itsInfo->Size<forecast_time>() *
		                             itsInfo->Size<level>() * itsInfo->Size<param>());
	}

	for (const auto& lvl : levels)
	{
		const auto g = itsConfiguration->BaseGrid();

		for (const auto& par : params)
		{
			itsInfo->First<forecast_type>();
			itsInfo->First<forecast_time>();
			itsInfo->First<level>();
			itsInfo->Reset<param>();

			while (itsInfo->Next())
			{
				if (itsInfo->Param() == par && itsInfo->Level() == lvl)
				{
					auto b = make_shared<base<double>>();
					b->grid = shared_ptr<grid>(g->Clone());

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

					itsInfo->Base(b);
				}
			}
		}
	}
	if (!itsConfiguration->UseDynamicMemoryAllocation())
	{
		itsBaseLogger.Trace("Using static memory allocation");
	}
	else
	{
		itsBaseLogger.Trace("Using dynamic memory allocation");
	}

	itsInfo->Reset();
	itsInfo->First<param>();

	if (itsPrimaryDimension == kUnknownDimension)
	{
		itsInfo->First<forecast_time>();
		itsInfo->First<forecast_type>();
		itsInfo->Reset<level>();
	}

	/*
	 * Do not launch more threads than there are things to calculate.
	 */

	size_t dims = itsInfo->Size<forecast_type>() * itsInfo->Size<forecast_time>() * itsInfo->Size<level>();

	if (itsPrimaryDimension == kTimeDimension)
	{
		dims = itsInfo->Size<forecast_time>() * itsInfo->Size<forecast_type>();
	}

	if (dims < static_cast<size_t>(itsThreadCount))
	{
		itsThreadCount = static_cast<short>(dims);
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
}

void compiled_plugin_base::SetParams(std::vector<param>& params)
{
	vector<level> levels;

	for (size_t i = 0; i < itsInfo->Size<level>(); i++)
	{
		levels.push_back(itsInfo->Peek<level>(i));
	}

	SetParams(params, levels);
}

/*
bool compiled_plugin_base::CompareGrids(initializer_list<shared_ptr<grid>> grids) const
{
    if (grids.size() <= 1)
    {
        throw kUnknownException;
    }

    auto it = grids.begin();
    auto first = *it;

    for (++it; it != grids.end(); ++it)
    {
        if (!*it)
        {
            continue;
        }

        if (*first != **it)
        {
            return false;
        }
    }

    return true;
}*/

bool compiled_plugin_base::IsMissingValue(initializer_list<double> values) const
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

info_t compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel, const params& theParams,
                                   const forecast_type& theType, bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);

	info_t ret;

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

		ret = f->Fetch(itsConfiguration, theTime, theLevel, theParams, theType, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack({ret}, itsConfiguration->UseCache());
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

info_t compiled_plugin_base::Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
                                   const forecast_type& theType, bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);

	info_t ret;

	try
	{
		ret = f->Fetch(itsConfiguration, theTime, theLevel, theParam, theType, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack({ret}, itsConfiguration->UseCache());
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

HPDimensionType compiled_plugin_base::PrimaryDimension() const
{
	return itsPrimaryDimension;
}
void compiled_plugin_base::PrimaryDimension(HPDimensionType thePrimaryDimension)
{
	if (itsInfo->Size<param>() > 0)
	{
		itsBaseLogger.Fatal("PrimaryDimension() must be called before plugin initialization is finished");
		exit(1);
	}

	itsPrimaryDimension = thePrimaryDimension;
}

void compiled_plugin_base::AllocateMemory(info<double> myTargetInfo)
{
	size_t paramIndex = myTargetInfo.Index<param>();

	for (myTargetInfo.Reset<param>(); myTargetInfo.Next<param>();)
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

	myTargetInfo.Index<param>(paramIndex);
}

void compiled_plugin_base::DeallocateMemory(info<double> myTargetInfo)
{
	size_t paramIndex = myTargetInfo.Index<param>();

	for (myTargetInfo.Reset<param>(); myTargetInfo.Next<param>();)
	{
		if (myTargetInfo.IsValidGrid())
		{
			myTargetInfo.Data().Clear();
		}
	}

	myTargetInfo.Index<param>(paramIndex);
}
