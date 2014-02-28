/**
 * @file relative_humidity.cpp
 *
 * @date Jan 21, 2012
 * @author partio
 */

#include "relative_humidity.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const double b = 17.27;
const double c = 237.3;
const double d = 1.8;

relative_humidity::relative_humidity()
{
	itsClearTextFormula = "RH = 100 *  (P * Q / 0.622 / es) * (P - es) / (P - Q * P / 0.622)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("relative_humidity"));

}

void relative_humidity::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to relative humidity
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	param requestedParam ("RH-PRCNT", 13);

	// GRIB 2

	requestedParam.GribDiscipline(0);
	requestedParam.GribCategory(1);
	requestedParam.GribParameter(1);

	params.push_back(requestedParam);

	SetParams(params);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void relative_humidity::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{	
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	params PParams = { param("P-HPA"), param("P-PA") };
	param QParam("Q-KGKG");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("relative_humidityThread #" + boost::lexical_cast<string> (threadIndex)));
	
	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = false;

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double TBase = 0;
		double PScale = 1;
		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);
		
		shared_ptr<info> TInfo;
		shared_ptr<info> PInfo;
		shared_ptr<info> QInfo;

		try
		{
			TInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								TParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

			if (!isPressureLevel)
			{
				PInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								PParams,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
			}

			QInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								QParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

		}
		catch (HPExceptionType& e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

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

		assert(TInfo->Grid()->AB() == PInfo->Grid()->AB());
		
		SetAB(myTargetInfo, TInfo);

		if (TInfo->Param().Unit() == kK)
		{
			TBase = -constants::kKelvin;
		}

		if (!isPressureLevel && (PInfo->Param().Name() == "P-PA" || PInfo->Param().Unit() == kPa))
		{
			PScale = 0.01;
		}

		shared_ptr<NFmiGrid> PGrid;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> QGrid(QInfo->Grid()->ToNewbaseGrid());

		if (!isPressureLevel)
		{
			PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
		}
		
		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && *myTargetInfo->Grid() == *QInfo->Grid() && (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

		string deviceType;

		{
			deviceType = "CPU";

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T = kFloatMissing;
				double P = kFloatMissing;
				double Q = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, QGrid, equalGrids, Q);

				if (isPressureLevel)
				{
					P = myTargetInfo->Level().Value();
				}
				else
				{
					InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				}
				
				if (T == kFloatMissing || P == kFloatMissing || Q == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				// Pressure needs to be hPa and temperature C
				
				double es = util::Es(T) * 0.01;

				T += TBase;
				P *= PScale;

				double RH = (P * Q / himan::constants::kEp / es) * (P - es) / (P - Q * P / himan::constants::kEp);

				if (RH > 1.0)
				{
					RH = 1.0;
				}
				else if (RH < 0.0)
				{
					RH = 0.0;
				}

				RH *= 100;

				if (!myTargetInfo->Value(RH))
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
