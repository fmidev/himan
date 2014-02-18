/**
 *
 * @file compiled_plugin_base.cpp
 *
 * @date Jan 15, 2013
 * @author partio
 */

#include "compiled_plugin_base.h"
#include <boost/thread.hpp>
#include "plugin_factory.h"
#include "logger_factory.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "writer.h"
#include "pcuda.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const double kInterpolatedValueEpsilon = 0.00001; //<! Max difference between two grid points (if smaller, points are considered the same)
mutex itsAdjustDimensionMutex;

compiled_plugin_base::compiled_plugin_base() : itsPluginIsInitialized(false)
{
	itsBaseLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("compiled_plugin_base"));
}

bool compiled_plugin_base::InterpolateToPoint(shared_ptr<const NFmiGrid> targetGrid, shared_ptr<NFmiGrid> sourceGrid, bool gridsAreEqual, double& value)
{

	/*
	 * Logic of interpolating values:
	 *
	 * 1) If source and target grids are equal, meaning that the grid AND the area
	 *	properties are effectively the same, do not interpolate. Instead return
	 *	the value of the source grid point that matches the ordering number of the
	 *	target grid point (ie. target grid point #1 --> source grid point #1 etc).
	 *
	 * 2) If actual interpolation is needed, first get the *grid* coordinates of the
	 *	latlon target point in the *source* grid. Then check if those grid coordinates
	 *  are very close to an actual grid point -- if so, return the value of the grid
	 *  point. This serves two purposes:
	 *	- We don't need to interpolate if the distance between requested grid point
	 *	  and actual grid point is small enough, saving some CPU cycles
	 *	- Sometimes when the requested grid point is close to grid edge, floating
	 *	  point inaccuracies might move it outside the grid. If this happens, the
	 *	  interpolation fails even though the grid point is valid.
	 *
	 * 3) If requested source grid point is not near an actual grid point, interpolate
	 *	the value of the point.
	 */

	// Step 1)

	if (gridsAreEqual)
	{
		value = sourceGrid->FloatValue(targetGrid->GridPoint());
		return true;
	}

	// Step 2)

	const NFmiPoint targetLatLonPoint = targetGrid->LatLon();
	const NFmiPoint sourceGridPoint = sourceGrid->LatLonToGrid(targetLatLonPoint);

	bool noInterpolation = (
						fabs(sourceGridPoint.X() - round(sourceGridPoint.X())) < kInterpolatedValueEpsilon &&
						fabs(sourceGridPoint.Y() - round(sourceGridPoint.Y())) < kInterpolatedValueEpsilon
	);

	if (noInterpolation)
	{
		value = sourceGrid->FloatValue(sourceGridPoint);
		return true;
	}

	// Step 3)

	return sourceGrid->InterpolateToGridPoint(sourceGridPoint, value);

}

bool compiled_plugin_base::AdjustLeadingDimension(shared_ptr<info> myTargetInfo)
{

	lock_guard<mutex> lock(itsAdjustDimensionMutex);

	// Leading dimension can be: time or level

	if (itsLeadingDimension == kTimeDimension)
	{
		if (!itsInfo->NextTime())
		{
			return false;
		}

		myTargetInfo->Time(itsInfo->Time());
	}
	else if (itsLeadingDimension == kLevelDimension)
	{
		if (!itsInfo->NextLevel())
		{
			return false;
		}

		myTargetInfo->Level(itsInfo->Level());
	}
	else
	{
		throw runtime_error(ClassName() + ": Invalid dimension type: " + boost::lexical_cast<string> (itsLeadingDimension));
	}

	return true;
}

bool compiled_plugin_base::AdjustNonLeadingDimension(shared_ptr<info> myTargetInfo)
{
	if (itsLeadingDimension == kTimeDimension)
	{
		return myTargetInfo->NextLevel();
	}
	else if (itsLeadingDimension == kLevelDimension)
	{
		return myTargetInfo->NextTime();
	}
	else
	{
		throw runtime_error(ClassName() + ": unsupported leading dimension: " + boost::lexical_cast<string> (itsLeadingDimension));
	}
}

void compiled_plugin_base::ResetNonLeadingDimension(shared_ptr<info> myTargetInfo)
{
	if (itsLeadingDimension == kTimeDimension)
	{
		myTargetInfo->ResetLevel();
	}
	else if (itsLeadingDimension == kLevelDimension)
	{
		myTargetInfo->ResetTime();
	}
	else
	{
		throw runtime_error(ClassName() + ": unsupported leading dimension: " + boost::lexical_cast<string> (itsLeadingDimension));
	}
}

himan::level compiled_plugin_base::LevelTransform(const himan::producer& sourceProducer,
													const himan::param& targetParam,
													const himan::level& targetLevel) const
{

	level sourceLevel = targetLevel;

	string levelName = HPLevelTypeToString.at(targetLevel.Type());
	string key = boost::lexical_cast<string> (sourceProducer.Id()) + "_" + levelName + "_" + targetParam.Name();

	// Return value from cache if present
	
	try
	{
		sourceLevel.Type(itsLevelTransformMap.at(key));

		sourceLevel.Name(levelName);

		if (sourceLevel.Type() == kGround)
		{
			sourceLevel.Value(0);
		}

		return sourceLevel;

	}
	catch (...)
	{

	}

	if (sourceProducer.TableVersion() != kHPMissingInt)
	{
		shared_ptr<neons> n = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

		string lvlName = n->NeonsDB().GetGridLevelName(targetParam.Name(), targetLevel.Type(), 204, sourceProducer.TableVersion());

		HPLevelType lvlType = kUnknownLevel;

		double lvlValue = targetLevel.Value();

		if (lvlName == "GROUND")
		{
			lvlType = kGround;
			lvlValue = 0;
		}
		else if (lvlName == "PRESSURE")
		{
			lvlType = kPressure;
		}
		else if (lvlName == "HYBRID")
		{
			lvlType = kHybrid;
		}
		else if (lvlName == "HEIGHT")
		{
			lvlType = kHeight;
		}
		else
		{
			throw runtime_error(ClassName() + ": Unknown level type: " + lvlName);
		}

		sourceLevel = level(lvlType, lvlValue, lvlName);
	}
	else
	{
		sourceLevel = targetLevel;
	}

	itsLevelTransformMap[key] = sourceLevel.Type();

	return sourceLevel;
}

bool compiled_plugin_base::SetAB(shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	if (myTargetInfo->Level().Type() == kHybrid)
	{
		size_t index = myTargetInfo->ParamIndex();

		myTargetInfo->Grid()->AB(sourceInfo->Grid()->AB());

		myTargetInfo->ParamIndex(index);
	}

	return true;
}

bool compiled_plugin_base::SwapTo(shared_ptr<info> myTargetInfo, HPScanningMode targetScanningMode)
{

	if (myTargetInfo->Grid()->ScanningMode() != targetScanningMode)
	{
		HPScanningMode originalMode = myTargetInfo->Grid()->ScanningMode();

		myTargetInfo->Grid()->ScanningMode(targetScanningMode);

		myTargetInfo->Grid()->Swap(originalMode);
	}

	return true;
}

void compiled_plugin_base::WriteToFile(shared_ptr<const info> targetInfo)
{
	shared_ptr<writer> aWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

	// writing might modify iterator positions --> create a copy

	auto tempInfo = make_shared<info> (*targetInfo);

	if (itsConfiguration->FileWriteOption() == kNeons || itsConfiguration->FileWriteOption() == kMultipleFiles)
	{
		// If info holds multiple parameters, we must loop over them all
		// Note! We only loop over the parameters, not over the times or levels!

		tempInfo->ResetParam();

		while (tempInfo->NextParam())
		{
			aWriter->ToFile(tempInfo, itsConfiguration);
		}
	}
	else if (itsConfiguration->FileWriteOption() == kSingleFile)
	{
		aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
	}
}

bool compiled_plugin_base::GetAndSetCuda(shared_ptr<const configuration> conf, int threadIndex)
{
#ifdef HAVE_CUDA
	bool ret = conf->UseCuda() && threadIndex <= conf->CudaDeviceCount();

	if (ret)
	{
		shared_ptr<pcuda> p = dynamic_pointer_cast <pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

		ret = p->SetDevice(threadIndex-1);
	}
#else
	bool ret = false;
#endif
	
	return ret;
}

void compiled_plugin_base::ResetCuda() const
{
#ifdef HAVE_CUDA
	shared_ptr<pcuda> p = dynamic_pointer_cast <pcuda> (plugin_factory::Instance()->Plugin("pcuda"));
	p->Reset();
#endif
}

void compiled_plugin_base::Start()
{
	if (!itsPluginIsInitialized)
	{
		itsBaseLogger->Error("Start() called before Init()");
		return;
	}
	
	boost::thread_group g;

	/*
	 * Each thread will have a copy of the target info.
	 */

	for (short i = 0; i < itsThreadCount; i++)
	{

		printf("Info::compiled_plugin: Thread %d starting\n", (i + 1)); // Printf is thread safe

		boost::thread* t = new boost::thread(&compiled_plugin_base::Run,
											 this,
											 make_shared<info> (*itsInfo),
											 i + 1);

		g.add_thread(t);

	}

	g.join_all();

	Finish();
}

void compiled_plugin_base::Init(shared_ptr<const plugin_configuration> conf)
{

	const short MAX_THREADS = 12; //<! Max number of threads we allow

	itsConfiguration = conf;

	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		itsTimer->Start();
		itsConfiguration->Statistics()->UsedGPUCount(conf->CudaDeviceCount());
	}

	// Determine thread count

	short coreCount = static_cast<short> (boost::thread::hardware_concurrency()); // Number of cores

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

	itsLeadingDimension = itsConfiguration->LeadingDimension();
	
	itsPluginIsInitialized = true;
}

void compiled_plugin_base::Run(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, threadIndex);
	}
}

void compiled_plugin_base::Finish()
{

	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer->Stop();
		itsConfiguration->Statistics()->AddToProcessingTime(itsTimer->GetTime());
	}

	if (itsConfiguration->FileWriteOption() == kSingleFile)
	{
		WriteToFile(itsInfo);
	}
}


void compiled_plugin_base::Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	itsBaseLogger->Fatal("Top level calculate called");
	exit(1);
}

void compiled_plugin_base::SetParams(std::vector<param>& params)
{
	if (params.empty())
	{
		itsBaseLogger->Fatal("size of target parameter vector is zero");
		exit(1);
	}
	
	// GRIB 1

	if (itsConfiguration->OutputFileType() == kGRIB1)
	{
		auto n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		for (unsigned int i = 0; i < params.size(); i++)
		{
			long table2Version = itsInfo->Producer().TableVersion();
			long parm_id = n->NeonsDB().GetGridParameterId(table2Version, params[i].Name());

			if (parm_id == -1)
			{
				itsBaseLogger->Warning("Grib1 parameter definitions not found from Neons");
				itsBaseLogger->Warning("table2Version is " + boost::lexical_cast<string> (table2Version) + ", parm_name is " + params[i].Name());
			}

			params[i].GribIndicatorOfParameter(parm_id);
			params[i].GribTableVersion(table2Version);
		}
	}

	itsInfo->Params(params);

	/*
	 * Create data structures.
	 */

	itsInfo->Create();

	/*
	 * Iterators must be reseted since they are at first position after Create()
	 */

	itsInfo->Reset();

	/*
	 * Do not launch more threads than there are things to calculate.
	 */

	if (itsLeadingDimension == kTimeDimension)
	{
		if (itsInfo->SizeTimes() < static_cast<size_t> (itsThreadCount))
		{
			itsThreadCount = static_cast<short> (itsInfo->SizeTimes());
		}
	}
	else if (itsLeadingDimension == kLevelDimension)
	{
		if (itsInfo->SizeLevels() < static_cast<size_t> (itsThreadCount))
		{
			itsThreadCount = static_cast<short> (itsInfo->SizeLevels());
		}
	}
	
	/*
	 * From the timing perspective at this point plugin initialization is
	 * considered to be done
	 */

	if (itsConfiguration->StatisticsEnabled())
	{
		itsConfiguration->Statistics()->UsedThreadCount(itsThreadCount);
		itsTimer->Stop();
		itsConfiguration->Statistics()->AddToInitTime(itsTimer->GetTime());
		itsTimer->Start();
	}
}

#ifdef HAVE_CUDA
void compiled_plugin_base::Unpack(initializer_list<shared_ptr<info>> infos)
{
	for (auto it = infos.begin(); it != infos.end(); ++it)
	{
		shared_ptr<info> tempInfo = *it;

		if (tempInfo->Grid()->PackedData()->packedLength == 0)
		{
			// This particular info does not have packed data
			continue;
		}

		assert(tempInfo->Grid()->PackedData()->ClassName() == "simple_packed");

		double* arr;
		size_t N = tempInfo->Grid()->PackedData()->unpackedLength;

		assert(N);

		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&arr), sizeof(double) * N));

		dynamic_pointer_cast<simple_packed> (tempInfo->Grid()->PackedData())->Unpack(arr, N);

		tempInfo->Data()->Set(arr, N);

		CUDA_CHECK(cudaFreeHost(arr));
	}
}
#endif
