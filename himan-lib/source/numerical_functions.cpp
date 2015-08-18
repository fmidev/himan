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

const std::valarray<double>& integral::Result() const
{
	assert(itsResult.size());
	
	return itsResult;
}

void integral::Evaluate()
{
	auto f = GET_PLUGIN(fetcher);
	
	std::vector<info_t> paramInfos;
	//Create a container that contains the parameter data. This container passed as an argument to the function that is being integrated over.
	std::vector<std::valarray<double>> paramsData;
	std::valarray<double> previousLevelValue;
	std::valarray<double> currentLevelValue;
	std::valarray<double> previousLevelHeight;
	std::valarray<double> currentLevelHeight;
	std::valarray<bool> missingValueMask;

	for (int lvl=itsLowestLevel; lvl<=itsHighestLevel; ++lvl)
	{
		itsLevel.Value(lvl);
		//fetch parameters

		//for (param itsParam:itsParams) <-- only for g++ 4.8
		for (unsigned int i = 0; i < itsParams.size(); i++)
		{
			param itsParam = itsParams[i];
        		paramInfos.push_back(f->Fetch(itsConfiguration, itsTime, itsLevel, itsParam, itsType, itsConfiguration->UseCudaForPacking()));
			paramsData.push_back(std::valarray<double> (paramInfos.back()->Data().Values().data(),paramInfos.back()->Data().Size()));
			//allocate result container
			if (!itsResult.size()) itsResult.resize(paramInfos.back()->Data().Size());
		}

		//fetch height TODO also implement pressure as hight coordinate
                param heightParam = param("HL-M");
                info_t heights = f->Fetch(itsConfiguration, itsTime, itsLevel, heightParam, itsType, itsConfiguration->UseCudaForPacking());
                currentLevelHeight = std::valarray<double> (heights->Data().Values().data(),heights->Data().Size());

		
		//mask for missing values
		auto missingValueMaskFunction = std::valarray<bool> (false,heights->Data().Size());
		for (unsigned int i = 0; i < paramsData.size(); i++)
		{
			missingValueMaskFunction = (paramsData[i] == kFloatMissing || missingValueMask);
		}

		//evaluate integration function
		if (itsFunction)
		{
			currentLevelValue = itsFunction(paramsData);
		}
		else
		{
			//use first param if no function is given
			currentLevelValue = paramsData[0];
		}
		
		//put missing values back in
		currentLevelValue[missingValueMaskFunction] = kFloatMissing;
		
		//move data from current level to previous level
		if (lvl == itsLowestLevel)
		{
			previousLevelHeight = std::move(currentLevelHeight);
			previousLevelValue = std::move(currentLevelValue);
			continue;
		}

		//perform trapezoideal integration TODO deal with missing values
		//
		//vectorized form of trapezoideal integration

		std::valarray<bool> upperBoundMask (previousLevelHeight > itsLowerBound && currentLevelHeight < itsLowerBound);
		std::valarray<bool> lowerBoundMask (previousLevelHeight > itsUpperBound && currentLevelHeight < itsUpperBound);
		std::valarray<bool> insideBoundsMask (previousLevelHeight <= itsUpperBound && currentLevelHeight >= itsLowerBound);
		// TODO Perhaps it is better to cast valarrays from the mask_array before this step. According to Stroustrup all operators and mathematical function can be applied to mask_array as well. Unfortunately not the case.
		itsResult[upperBoundMask] += (Interpolate(std::valarray<double> (currentLevelValue[upperBoundMask]),std::valarray<double> (previousLevelValue[upperBoundMask]), std::valarray<double> (currentLevelHeight[upperBoundMask]), std::valarray<double> (previousLevelHeight[upperBoundMask]), std::valarray<double> (itsLowerBound[upperBoundMask]))+ std::valarray<double>(previousLevelValue[upperBoundMask]))/ 2 * (std::valarray<double> (previousLevelHeight[upperBoundMask]) - std::valarray<double> (itsLowerBound[upperBoundMask]));
                itsResult[lowerBoundMask] += (Interpolate(std::valarray<double> (currentLevelValue[lowerBoundMask]),std::valarray<double> (previousLevelValue[lowerBoundMask]),std::valarray<double> (currentLevelHeight[lowerBoundMask]),std::valarray<double> (previousLevelHeight[lowerBoundMask]),std::valarray<double> (itsUpperBound[lowerBoundMask]))+std::valarray<double> (previousLevelValue[lowerBoundMask]))/ 2 * (std::valarray<double> (itsUpperBound[lowerBoundMask]) - std::valarray<double> (currentLevelHeight[lowerBoundMask]));
		itsResult[insideBoundsMask] += (std::valarray<double> (previousLevelValue[insideBoundsMask]) + std::valarray<double> (currentLevelValue[insideBoundsMask])) / 2 * (std::valarray<double> (previousLevelHeight[insideBoundsMask]) - std::valarray<double> (currentLevelHeight[insideBoundsMask]));
		
		//serial version of trapezoideal integration
		/*
		for (size_t i=0; i<paramInfos.back()->Data().Size(); ++i)
		{

        		// value is below the lowest limit
        		if (previousLevelHeight[i] > itsLowerBound[i] && currentLevelHeight[i] < itsLowerBound[i])
        		{
                		double lowerValue = previousLevelValue[i]+(currentLevelValue[i]-previousLevelValue[i])*(itsLowerBound[i]-previousLevelHeight[i])/(currentLevelHeight[i]-previousLevelHeight[i]);
                		itsResult[i] += (lowerValue + previousLevelValue[i]) / 2 * (previousLevelHeight[i] - itsLowerBound[i]);
			}
        		// value is above the highest limit
        		else if (previousLevelHeight[i] > itsUpperBound[i] && currentLevelHeight[i] < itsUpperBound[i])
        		{
                		double upperValue = previousLevelValue[i]+(currentLevelValue[i]-previousLevelValue[i])*(itsLowerBound[i]-previousLevelHeight[i])/(currentLevelHeight[i]-previousLevelHeight[i]);
                		itsResult[i] += (upperValue + currentLevelValue[i]) / 2 * (itsUpperBound[i] - currentLevelHeight[i]);

        		}
        		else if (previousLevelHeight[i] <= itsUpperBound[i] && currentLevelHeight[i] >= itsLowerBound[i])        
			{
                		itsResult[i] += (previousLevelValue[i] + currentLevelValue[i]) / 2 * (previousLevelHeight[i] - currentLevelHeight[i]);
        		}

		}*/
		
		//move data from current level to previous level at the end of the integration step
                previousLevelHeight = std::move(currentLevelHeight);
                previousLevelValue = std::move(currentLevelValue);
		paramInfos.clear();
		paramsData.clear();
	}
}

void integral::LowerBound(const std::valarray<double>& theLowerBound)
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

void integral::UpperBound(const std::valarray<double>& theUpperBound)
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

void integral::LowerLevelLimit(int theLowestLevel)
{
	itsLowestLevel = theLowestLevel;
}

void integral::UpperLevelLimit(int theHighestLevel)
{
	itsHighestLevel = theHighestLevel;
}

void integral::ForecastType(forecast_type theType)
{
	itsType = theType;
}

void integral::ForecastTime(forecast_time theTime)
{
	itsTime = theTime;
}

// TODO add check that all information that is needed is given to the class object
bool integral::Complete()
{
	return true;
}
