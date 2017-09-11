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
#include "neons.h"
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

bool compiled_plugin_base::Next(info& myTargetInfo)
{
	lock_guard<mutex> lock(dimensionMutex);

	if (!itsDimensionsRemaining)
	{
		return false;
	}

	if (itsInfo->NextLevel())
	{
		bool ret = myTargetInfo.Level(itsInfo->Level());
		assert(ret);
		ret = myTargetInfo.Time(itsInfo->Time());
		assert(ret);
		ret = myTargetInfo.ForecastType(itsInfo->ForecastType());
		assert(ret);

		return ret;
	}

	// No more levels at this forecast type/time combination; rewind level iterator

	itsInfo->FirstLevel();

	if (itsInfo->NextTime())
	{
		bool ret = myTargetInfo.Time(itsInfo->Time());
		assert(ret);
		ret = myTargetInfo.Level(itsInfo->Level());
		assert(ret);
		ret = myTargetInfo.ForecastType(itsInfo->ForecastType());
		assert(ret);

		return ret;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	itsInfo->FirstTime();

	if (itsInfo->NextForecastType())
	{
		bool ret = myTargetInfo.Time(itsInfo->Time());
		assert(ret);
		ret = myTargetInfo.Level(itsInfo->Level());
		assert(ret);
		ret = myTargetInfo.ForecastType(itsInfo->ForecastType());
		assert(ret);

		return ret;
	}

	// future threads calling for new dimensions aren't getting any

	itsDimensionsRemaining = false;

	return false;
}

bool compiled_plugin_base::NextExcludingLevel(info& myTargetInfo)
{
	lock_guard<mutex> lock(dimensionMutex);

	if (!itsDimensionsRemaining)
	{
		return false;
	}

	if (itsInfo->NextTime())
	{
		bool ret = myTargetInfo.Time(itsInfo->Time());
		assert(ret);
		ret = myTargetInfo.ForecastType(itsInfo->ForecastType());
		assert(ret);

		return ret;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	itsInfo->FirstTime();

	if (itsInfo->NextForecastType())
	{
		bool ret = myTargetInfo.Time(itsInfo->Time());
		assert(ret);
		ret = myTargetInfo.ForecastType(itsInfo->ForecastType());
		assert(ret);

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
		const size_t paramIndex = myTargetInfo->ParamIndex();

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
		{
			myTargetInfo->Grid()->AB(sourceInfo->Grid()->AB());
		}

		myTargetInfo->ParamIndex(paramIndex);
	}

	return true;
}

void compiled_plugin_base::WriteToFile(const info& targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

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
			lock_guard<mutex> lock(singleFileWriteMutex);

			aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
		}
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(targetInfo);
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
		itsInfo->FirstForecastType();
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
		itsConfiguration->Statistics()->UsedGPUCount(conf->CudaDeviceCount());
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

	itsInfo = itsConfiguration->Info();

	itsPluginIsInitialized = true;
}

void compiled_plugin_base::RunAll(info_t myTargetInfo, unsigned short threadIndex)
{
	while (Next(*myTargetInfo))
	{
		if (itsConfiguration->UseDynamicMemoryAllocation())
		{
			AllocateMemory(*myTargetInfo);
		}

		assert(myTargetInfo->Data().Size() > 0);

		Calculate(myTargetInfo, threadIndex);

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
		}

		WriteToFile(*myTargetInfo);
	}
}

void compiled_plugin_base::RunTimeDimension(info_t myTargetInfo, unsigned short threadIndex)
{
	while (NextExcludingLevel(*myTargetInfo))
	{
		for (myTargetInfo->ResetLevel(); myTargetInfo->NextLevel();)
		{
			if (itsConfiguration->UseDynamicMemoryAllocation())
			{
				AllocateMemory(*myTargetInfo);
			}

			assert(myTargetInfo->Data().Size() > 0);

			Calculate(myTargetInfo, threadIndex);

			if (itsConfiguration->StatisticsEnabled())
			{
				itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
				itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
			}

			WriteToFile(*myTargetInfo);
		}
	}
}

void compiled_plugin_base::Run(unsigned short threadIndex)
{
	auto myTargetInfo = make_shared<info>(*itsInfo);
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

	// If no other info is holding access to grids in this info,
	// they are automatically destroyed and memory is released.

	itsInfo->Clear();
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

void compiled_plugin_base::SetParams(std::vector<param>& params)
{
	if (params.empty())
	{
		itsBaseLogger.Fatal("size of target parameter vector is zero");
		exit(1);
	}

	// GRIB 1

	if (itsConfiguration->OutputFileType() == kGRIB1 && itsInfo->Producer().Class() != kPreviClass)
	{
		HPDatabaseType dbtype = itsConfiguration->DatabaseType();

		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = GET_PLUGIN(neons);

			for (unsigned int i = 0; i < params.size(); i++)
			{
				if (params[i].Name() == "DUMMY")
				{
					// special placeholder parameter which is replaced later
					continue;
				}

				long table2Version = itsInfo->Producer().TableVersion();

				if (table2Version == kHPMissingInt)
				{
					auto prodinfo = n->NeonsDB().GetProducerDefinition(itsInfo->Producer().Id());

					if (!prodinfo.empty())
					{
						table2Version = boost::lexical_cast<long>(prodinfo["no_vers"]);
					}
				}

				if (table2Version == kHPMissingInt)
				{
					itsBaseLogger.Warning("table2Version not found from Neons for producer " +
					                      boost::lexical_cast<string>(itsInfo->Producer().Name()));
					continue;
				}

				long parm_id = n->NeonsDB().GetGridParameterId(table2Version, params[i].Name());

				if (parm_id == -1)
				{
					string msg = "Grib1 parameter definition not found from Neons for table version " +
					             boost::lexical_cast<string>(table2Version) + ", parameter name " + params[i].Name();

					itsBaseLogger.Warning(msg);
					continue;
				}

				params[i].GribIndicatorOfParameter(parm_id);
				params[i].GribTableVersion(table2Version);
			}
		}

		if (dbtype == kRadon || dbtype == kNeonsAndRadon)
		{
			auto r = GET_PLUGIN(radon);

			for (unsigned int i = 0; i < params.size(); i++)
			{
				if (params[i].Name() == "DUMMY")
				{
					// special placeholder parameter which is replaced later
					continue;
				}

				if (params[i].GribIndicatorOfParameter() != kHPMissingInt &&
				    params[i].GribTableVersion() != kHPMissingInt)
				{
					continue;
				}

				// We'll fetch the parameter informatio using the type of
				// the first level in the info. This will obviously not work
				// correctly if multiple level types are within one info.

				auto firstLevel = itsInfo->PeekLevel(0);
				auto levelInfo = r->RadonDB().GetLevelFromDatabaseName(
				    boost::to_upper_copy(HPLevelTypeToString.at(firstLevel.Type())));

				if (levelInfo.empty())
				{
					itsBaseLogger.Warning("Level type '" + HPLevelTypeToString.at(firstLevel.Type()) +
					                      "' not found from radon");
					continue;
				}

				map<string, string> paraminfo = r->RadonDB().GetParameterFromDatabaseName(
				    itsInfo->Producer().Id(), params[i].Name(), firstLevel.Type(), firstLevel.Value());

				if (paraminfo.empty() || paraminfo["grib1_number"].empty() || paraminfo["grib1_table_version"].empty())
				{
					string msg = "Grib1 parameter definition not found from Radon for producer " +
					             to_string(itsInfo->Producer().Id()) + ", parameter name " + params[i].Name();

					itsBaseLogger.Warning(msg);
					continue;
				}

				params[i].GribIndicatorOfParameter(stoi(paraminfo["grib1_number"]));
				params[i].GribTableVersion(stoi(paraminfo["grib1_table_version"]));

				if (!paraminfo["precision"].empty())
				{
					params[i].Precision(stoi(paraminfo["precision"]));
				}
			}
		}
	}

	itsInfo->Params(params);

	/*
	 * Create data structures.
	 */

	itsInfo->Create(itsInfo->itsBaseGrid.get(), !itsConfiguration->UseDynamicMemoryAllocation());
	itsInfo->itsBaseGrid.reset();

	if (!itsConfiguration->UseDynamicMemoryAllocation())
	{
		itsBaseLogger.Trace("Using static memory allocation");
	}
	else
	{
		itsBaseLogger.Trace("Using dynamic memory allocation");
	}

	itsInfo->Reset();
	itsInfo->FirstParam();

	if (itsPrimaryDimension == kUnknownDimension)
	{
		itsInfo->FirstTime();
		itsInfo->FirstForecastType();
		itsInfo->ResetLevel();
	}

	/*
	 * Do not launch more threads than there are things to calculate.
	 */

	size_t dims = itsInfo->SizeForecastTypes() * itsInfo->SizeTimes() * itsInfo->SizeLevels();

	if (itsPrimaryDimension == kTimeDimension)
	{
		dims = itsInfo->SizeTimes() * itsInfo->SizeForecastTypes();
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

#ifdef HAVE_CUDA
void compiled_plugin_base::Unpack(initializer_list<info_t> infos)
{
	auto c = GET_PLUGIN(cache);

	for (auto it = infos.begin(); it != infos.end(); ++it)
	{
		info_t tempInfo = *it;

		if (!tempInfo->Grid()->IsPackedData() || tempInfo->Grid()->PackedData().packedLength == 0)
		{
			// Safeguard: This particular info does not have packed data
			continue;
		}

		assert(tempInfo->Grid()->PackedData().ClassName() == "simple_packed");

		util::Unpack({tempInfo->Grid()});

		if (itsConfiguration->UseCache())
		{
			c->Insert(*tempInfo);
		}
	}
}
#endif

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
		 * 1) Fetch packed data iff cuda unpacking is enabled (UseCudaForPacking() == true): it makes no sense to unpack
		 * the data in himan with CPU.
		 *    If we allow fetcher to return packed data, it will implicitly disable cache integration of fetched data.
		 *
		 * 2a) If caller does not want packed data (returnPacked == false), unpack it here and insert to cache.
		 *
		 * 2b) If caller wants packed data, return data as-is and leave cache integration to caller.
		 */

		ret = f->Fetch(itsConfiguration, theTime, theLevel, theParams, theType, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->Grid()->IsPackedData())
		{
			assert(ret->Grid()->PackedData().ClassName() == "simple_packed");

			util::Unpack({ret->Grid()});

			auto c = GET_PLUGIN(cache);

			c->Insert(*ret);
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
		if (!returnPacked && ret->Grid()->IsPackedData())
		{
			assert(ret->Grid()->PackedData().ClassName() == "simple_packed");

			util::Unpack({ret->Grid()});

			auto c = GET_PLUGIN(cache);

			c->Insert(*ret);
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

HPDimensionType compiled_plugin_base::PrimaryDimension() const { return itsPrimaryDimension; }
void compiled_plugin_base::PrimaryDimension(HPDimensionType thePrimaryDimension)
{
	if (itsInfo->SizeParams() > 0)
	{
		itsBaseLogger.Fatal("PrimaryDimension() must be called before plugin initialization is finished");
		exit(1);
	}

	itsPrimaryDimension = thePrimaryDimension;
}

void compiled_plugin_base::AllocateMemory(info myTargetInfo)
{
	if (myTargetInfo.Grid()->Class() == kIrregularGrid)
	{
		return;
	}

	size_t paramIndex = myTargetInfo.ParamIndex();

	for (myTargetInfo.ResetParam(); myTargetInfo.NextParam();)
	{
		myTargetInfo.Data().Resize(myTargetInfo.Grid()->Ni(), myTargetInfo.Grid()->Nj());
	}

	myTargetInfo.ParamIndex(paramIndex);
}

void compiled_plugin_base::DeallocateMemory(info myTargetInfo)
{
	if (myTargetInfo.Grid()->Class() == kIrregularGrid)
	{
		return;
	}

	size_t paramIndex = myTargetInfo.ParamIndex();

	for (myTargetInfo.ResetParam(); myTargetInfo.NextParam();)
	{
		myTargetInfo.Grid()->Data().Clear();
	}

	myTargetInfo.ParamIndex(paramIndex);
}
