/**
 * @file transformer.cpp
 *
 * @date May, 2014
 * @author Tack
 */

#include "transformer.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "json_parser.h"
#include "util.h"
#include "NFmiGrid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"


transformer::transformer() : itsBase(0.0), itsScale(1.0), itsTargetUnivID(999)
{
	itsClearTextFormula = "target_param = source_param * itsScale + itsBase";
	itsCudaEnabledCalculation = false;

	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("transformer"));
}

vector<himan::level> transformer::LevelsFromString(const string& levelType, const string& levelValues) const
{
	string levelTypeUpper = levelType;
	boost::to_upper(levelTypeUpper);

	HPLevelType theLevelType;

	if (levelTypeUpper == "HEIGHT")
	{
		theLevelType = kHeight;
	}
	else if (levelTypeUpper == "PRESSURE")
	{
		theLevelType = kPressure;
	}
	else if (levelTypeUpper == "HYBRID")
	{
		theLevelType = kHybrid;
	}
	else if (levelTypeUpper == "GROUND")
	{
		theLevelType = kGround;
	}
	else if (levelTypeUpper == "MEANSEA")
	{
		theLevelType = kMeanSea;
	}
	else
	{
		throw runtime_error("Unknown level type: " + levelType);
	}

	// can cause exception, what will happen then ?

	vector<string> levelsStr = util::Split(levelValues, ",", true);

	vector<level> levels;

	for (size_t i = 0; i < levelsStr.size(); i++)
	{
		levels.push_back(level(theLevelType, boost::lexical_cast<float> (levelsStr[i]), levelType));
	}

	return levels;
}

void transformer::set_additional_parameters()
{
	std::string itsSourceLevelType;
	std::string SourceLevels;

	if(itsConfiguration->Options().count("base"))
	{
		itsBase = boost::lexical_cast<double>(itsConfiguration->GetValue("base"));
	}
	else
	{
		itsLogger->Warning("Base not specified, using default value 0.0");
	}
	
	if(itsConfiguration->Options().count("scale"))
	{
		itsScale = boost::lexical_cast<double>(itsConfiguration->GetValue("scale"));
	}
	else
	{
		itsLogger->Warning("Scale not specified, using default value 1.0");
	}

	if(itsConfiguration->Options().count("target_univ_ID"))
	{
		itsTargetUnivID = boost::lexical_cast<int>(itsConfiguration->GetValue("target_univ_ID"));
	}
	else
	{
		itsLogger->Warning("Target_univ_ID not specified, using  default value 999");
	}
	
	if(itsConfiguration->Options().count("target_param"))
	{
		itsTargetParam = itsConfiguration->GetValue("target_param");
	}
	else
	{
		itsLogger->Warning("Target_param not specified. Exiting program.");
		exit(1);
	}

	if(itsConfiguration->Options().count("source_param"))
	{
		itsSourceParam = itsConfiguration->GetValue("source_param");
	}
	else
	{
		itsSourceParam = itsTargetParam;
		itsLogger->Warning("Source_param not specified, source_param set to target_param");
	}

	if(itsConfiguration->Options().count("source_level_type"))
	{
		itsSourceLevelType = itsConfiguration->GetValue("source_level_type");
	}
	else
	{
		itsSourceLevelType = itsConfiguration->GetValue("leveltype");
		itsLogger->Warning("Source_level_type not specified, value set to leveltype");
	}
	
	if(itsConfiguration->Options().count("source_levels"))
	{
		SourceLevels = itsConfiguration->GetValue("source_levels");
	}
	else
	{
		SourceLevels = itsConfiguration->GetValue("levels");
		itsLogger->Warning("Source_levels not specified, values set to target levels");
	}
	
	// looks useful to use this function to create source_levels
	itsSourceLevels = LevelsFromString(itsSourceLevelType, SourceLevels);
}

void transformer::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	set_additional_parameters();

	/*
	 * Set target parameter to T
	 * - name T-C
	 * - univ_id 4
	 * - grib2 descriptor 0'00'000
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param requestedParam(itsTargetParam, itsTargetUnivID);

	// GRIB 2
	
	//requestedParam.GribDiscipline(pt.get<string>("target_GribDiscipline"));
	//requestedParam.GribCategory(pt.get<string>("target_GribCategory"));
	//requestedParam.GribParameter(pt.get<string>("target_GribParameter"));

	// GRIB 1

	theParams.push_back(requestedParam);

	SetParams(theParams);

	Start();
	
}


/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void transformer::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	shared_ptr<fetcher> aFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameter

	param InputParam(itsSourceParam);

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("transformerThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(threadIndex);

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		// Source info for T

		shared_ptr<info> sourceInfo;

		try
		{
			sourceInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 itsSourceLevels[myTargetInfo->LevelIndex()],
								 InputParam,
								 itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		}
		catch (HPExceptionType& e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing);

					if (itsConfiguration->StatisticsEnabled())
					{
						itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
					}
					
					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
		}

		SetAB(myTargetInfo, sourceInfo);

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (*myTargetInfo->Grid() == *sourceInfo->Grid());

		string deviceType;

/*
#ifdef HAVE_CUDA

		// If we read packed data but grids are not equal we cannot use cuda
		// for calculations (our cuda routines do not know how to interpolate)

		if (!equalGrids && sourceInfo->Grid()->IsPackedData())
		{
			myThreadedLogger->Debug("Unpacking for CPU calculation");

			Unpack({sourceInfo});
		}

		if (useCudaInThisThread && equalGrids)
		{
	
			deviceType = "GPU";

			auto opts = CudaPrepare(myTargetInfo, sourceInfo);

			transformer_cuda::Process(*opts);

			missingCount = opts->missing;
			count = opts->N;

			CudaFinish(move(opts), myTargetInfo, sourceInfo);

		}
		else
#endif
*/
		{

			deviceType = "CPU";

			shared_ptr<NFmiGrid> sourceGrid(sourceInfo->Grid()->ToNewbaseGrid());

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{

				count++;

				double value = kFloatMissing;

				InterpolateToPoint(targetGrid, sourceGrid, equalGrids, value);

				if (value == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				double newValue = value * itsScale + itsBase;

				if (!myTargetInfo->Value(newValue))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}
			}

			/*
			 * Newbase normalizes scanning mode to bottom left -- if that's not what
			 * the target scanning mode is, we have to swap the data back.
			 */

			SwapTo(myTargetInfo, kBottomLeft);

		}

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}

	}
}

/*
#ifdef HAVE_CUDA

unique_ptr<transformer_cuda::options> transformer::CudaPrepare( shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	unique_ptr<transformer_cuda::options> opts(new transformer_cuda::options);

	opts->N = sourceInfo->Data()->Size();

	opts->base = itsBase;
	opts->scale = itsScale;

	opts->source = sourceInfo->ToSimple();
	opts->dest = myTargetInfo->ToSimple();

	return opts;
}

void transformer::CudaFinish(unique_ptr<transformer_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	// Copy data back to infos

	CopyDataFromSimpleInfo(myTargetInfo, opts->dest, false);

	if (sourceInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(sourceInfo, opts->source, true);
	}

	SwapTo(myTargetInfo, sourceInfo->Grid()->ScanningMode());

}

#endif
*/
