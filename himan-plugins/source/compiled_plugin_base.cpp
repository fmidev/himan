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

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const unsigned int MAX_THREADS = 12; //<! Max number of threads we allow
const double kInterpolatedValueEpsilon = 0.00001; //<! Max difference between two grid points (if smaller, points are considered the same)

mutex itsAdjustDimensionMutex;

unsigned short compiled_plugin_base::ThreadCount(short userThreadCount) const
{
    unsigned int coreCount = boost::thread::hardware_concurrency(); // Number of cores

    unsigned short threadCount = MAX_THREADS;

    if (userThreadCount > 0)
    {
    	threadCount = userThreadCount;
    }
    else if (MAX_THREADS > coreCount)
    {
    	threadCount = static_cast<unsigned short> (coreCount);
    }

    return threadCount;
}


bool compiled_plugin_base::InterpolateToPoint(shared_ptr<const NFmiGrid> targetGrid, shared_ptr<NFmiGrid> sourceGrid, bool gridsAreEqual, double& value)
{
	const NFmiPoint targetLatLonPoint = targetGrid->LatLon();
	const NFmiPoint targetGridPoint = targetGrid->GridPoint();

	if (gridsAreEqual)
	{
		value = sourceGrid->FloatValue(targetGridPoint);
		return true;
	}

	const NFmiPoint sourceGridPoint = sourceGrid->LatLonToGrid(targetLatLonPoint);

	bool noInterpolation = (fabs(targetGridPoint.X() - round(sourceGridPoint.X())) < kInterpolatedValueEpsilon &&
		 fabs(targetGridPoint.Y() - round(sourceGridPoint.Y())) < kInterpolatedValueEpsilon);

	if (noInterpolation)
	{
		value = sourceGrid->FloatValue();
		return true;
	}

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

	level sourceLevel;

	if (sourceProducer.TableVersion() != kHPMissingInt)
	{
		shared_ptr<neons> n = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

		string lvlName = n->NeonsDB().GetGridLevelName(targetParam.Name(), targetLevel.Type(), 204, sourceProducer.TableVersion());

		HPLevelType lvlType = kUnknownLevel;

		float lvlValue = targetLevel.Value();

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

	return sourceLevel;
}

bool compiled_plugin_base::SetAB(shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	if (myTargetInfo->Level().Type() == kHybrid)
	{
		int index = myTargetInfo->ParamIndex();

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			myTargetInfo->Grid()->AB(sourceInfo->Grid()->AB());
		}

		myTargetInfo->ParamIndex(index);
	}

	return true;
}