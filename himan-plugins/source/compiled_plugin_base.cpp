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

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "writer.h"
#include "pcuda.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const short MAX_THREADS = 12; //<! Max number of threads we allow
const double kInterpolatedValueEpsilon = 0.00001; //<! Max difference between two grid points (if smaller, points are considered the same)

mutex itsAdjustDimensionMutex;

short compiled_plugin_base::ThreadCount(short userThreadCount) const
{
	short coreCount = static_cast<short> (boost::thread::hardware_concurrency()); // Number of cores

	short threadCount = MAX_THREADS;

	if (userThreadCount > 0)
	{
		threadCount = userThreadCount;
	}
	else if (MAX_THREADS > coreCount)
	{
		threadCount = coreCount;
	}

	return threadCount;
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
	 *	latlon target point. Then check if those grid coordinates are very close
	 *	to a grid point -- if so, return the value of the grid point. This serves two
	 *	purposes:
	 *	- We don't need to interpolate if the distance between requested grid point
	 *	  and actual grid point is small enough, saving some CPU cycles
	 *	- Sometimes when the requested grid point is close to grid edge, floating
	 *	  point inaccuracies might move it outside the grid. If this happens, the
	 *	  interpolation fails even though initially the grid point is valid.
	 *
	 * 3) If requested source grid point is not near and actual grid point, interpolate
	 *	the value of the point.
	 */

	// Step 1)

	if (gridsAreEqual)
	{
		value = sourceGrid->FloatValue(targetGrid->GridPoint());
		return true;
	}

	const NFmiPoint targetLatLonPoint = targetGrid->LatLon();
	const NFmiPoint sourceGridPoint = targetGrid->LatLonToGrid(targetLatLonPoint.X(), targetLatLonPoint.Y());

	// Step 2)

	bool noInterpolation = (fabs(sourceGridPoint.X() - round(sourceGridPoint.X())) < kInterpolatedValueEpsilon &&
		 fabs(sourceGridPoint.Y() - round(sourceGridPoint.Y())) < kInterpolatedValueEpsilon);

	if (noInterpolation)
	{
		value = sourceGrid->FloatValue(sourceGridPoint);
		return true;
	}

	// Step 3)

	return sourceGrid->InterpolateToLatLonPoint(targetLatLonPoint, value);

}

bool compiled_plugin_base::AdjustLeadingDimension(shared_ptr<info> myTargetInfo)
{

	lock_guard<mutex> lock(itsAdjustDimensionMutex);

	// Leading dimension can be: time or level

	if (itsLeadingDimension == kTimeDimension)
	{
		if (!itsFeederInfo->NextTime())
		{
			return false;
		}

		myTargetInfo->Time(itsFeederInfo->Time());
	}
	else if (itsLeadingDimension == kLevelDimension)
	{
		if (!itsFeederInfo->NextLevel())
		{
			return false;
		}

		myTargetInfo->Level(itsFeederInfo->Level());
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

void compiled_plugin_base::StoreGrib1ParameterDefinitions(vector<param> params, long table2Version)
{
	shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

	for (unsigned int i = 0; i < params.size(); i++)
	{
		long parm_id = n->NeonsDB().GetGridParameterId(table2Version, params[i].Name());
		params[i].GribIndicatorOfParameter(parm_id);
		params[i].GribTableVersion(table2Version);
	}
}

void compiled_plugin_base::WriteToFile(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> targetInfo)
{
	shared_ptr<writer> aWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

	// writing might modify iterator positions --> create a copy

	shared_ptr<info> tempInfo(new info(*targetInfo));

	if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
	{
		// if info holds multiple parameters, we must loop over them all

		tempInfo->ResetParam();

		while (tempInfo->NextParam())
		{
			aWriter->ToFile(tempInfo, conf);
		}
	}
	else if (conf->FileWriteOption() == kSingleFile)
	{
		aWriter->ToFile(tempInfo, conf, conf->ConfigurationFile());
	}

	tempInfo.reset();
}

bool compiled_plugin_base::GetAndSetCuda(shared_ptr<const configuration> conf, int threadIndex)
{
	bool ret = conf->UseCuda() && threadIndex <= conf->CudaDeviceCount();

	if (ret)
	{
		shared_ptr<pcuda> p = dynamic_pointer_cast <pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

		ret = p->SetDevice(threadIndex-1);
	}

	return ret;
}

void compiled_plugin_base::ResetCuda() const
{
	shared_ptr<pcuda> p = dynamic_pointer_cast <pcuda> (plugin_factory::Instance()->Plugin("pcuda"));
	p->Reset();
}

int compiled_plugin_base::CudaDeviceId() const
{
	shared_ptr<pcuda> p = dynamic_pointer_cast <pcuda> (plugin_factory::Instance()->Plugin("pcuda"));
	return p->GetDevice();
}
