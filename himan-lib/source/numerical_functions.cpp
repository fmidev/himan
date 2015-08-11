/**
 * @file numerical_functions.cpp
 * @author tack
 */

#include "numerical_functions.h"
#include "NFmiInterpolation.h"
#include "plugin_factory.h"
#include "fetcher.h"

using namespace himan;

using namespace numerical_functions;

integral::integral()
{
}

void integral::Params(std::vector<param> theParams)
{
	itsParams = theParams;
}

void integral::Function(std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> theFunction)
{
	itsFunction = theFunction;
}

const std::vector<double>& integral::Result() const
{
	assert(itsResult.size());
	
	return itsResult;
}

bool integral::Evaluate()
{
	auto f = GET_PLUGIN(fetcher);
	
	std::vector<info_t> paramInfos;
	//Create a container that contains the parameter data. This container passed as an argument to the function that is being integrated over.
	std::vector<std::valarray<double>> paramsData;
	std::valarray<double> previousLevelValue;
	std::valarray<double> currentLevelValue;
	std::valarray<double> previousLevelHeight;
	std::valarray<double> currentLevelHeight;

	//set manually for testing
	itsLowestLevel = 50; itsHighestLevel = 55;

	for (int lvl=itsLowestLevel; lvl<=itsHighestLevel; ++lvl)
	{
		itsLevel.Value(lvl);
		//fetch parameter and create pointers to their data structures

		//for (param itsParam:itsParams) <-- only for g++ 4.8
		for (unsigned int i = 0; i < itsParams.size(); i++)
		{
			param itsParam = itsParams[i];
        		paramInfos.push_back(f->Fetch(itsConfiguration, itsTime, itsLevel, itsParam, itsType, itsConfiguration->UseCudaForPacking()));
			paramsData.push_back(std::valarray<double> (paramInfos.back()->Data().Values().data(),paramInfos.back()->Data().Size()));
			//allocate result container
			if (!itsResult.size()) itsResult.resize(paramInfos.back()->Data().Size());
		}

		//evaluate integration function TODO if no function is given copy data from info class to currentLevelValue
		if (itsFunction)
		{
			currentLevelValue = itsFunction(paramsData);
		}
		
		//move data from current level to previous level
		if (lvl == itsLowestLevel)
		{
			previousLevelHeight = std::move(currentLevelHeight);
			previousLevelValue = std::move(currentLevelValue);
			continue;
		}

		//perform trapezoideal integration step TODO Implement catching of upper/lower bound 
		for (size_t i=0; i<paramInfos.back()->Data().Size(); ++i)
		{
			itsResult[i] += (currentLevelValue[i] + previousLevelValue[i]) / 2 * (previousLevelHeight[i] - currentLevelHeight[i]);
		}
		
		//move data from current level to previous level at the end of the integration step
                previousLevelHeight = std::move(currentLevelHeight);
                previousLevelValue = std::move(currentLevelValue);
	}
	return true;
}

void integral::LowerBound(const std::vector<double>& theLowerBound)
{
	itsLowerBound = theLowerBound;
	
	// If height limits have missing values we can't process those grid points
	
	itsOutOfBound.resize(itsLowerBound.size(), false);

	for (size_t i = 0; i < itsLowerBound.size(); i++)
	{
		if (itsLowerBound[i] == kFloatMissing)
		{
			itsOutOfBound[i] = true;
		}
	}
}

void integral::UpperBound(const std::vector<double>& theUpperBound)
{
	itsUpperBound = theUpperBound;
	
	// If height limits have missing values we can't process those grid points
	
	itsOutOfBound.resize(itsUpperBound.size(), false);
	
	for (size_t i = 0; i < itsUpperBound.size(); i++)
	{
		if (itsUpperBound[i] == kFloatMissing)
		{
			itsOutOfBound[i] = true;
		}
	}
}
