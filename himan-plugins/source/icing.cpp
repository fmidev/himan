/**
 * @file icing.cpp
 *
 *  Created on: Jan 03, 2013
 *  @author aaltom
 */

#include "icing.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const double kValueEpsilon = 0.00001;

icing::icing()
{
	itsClearTextFormula = "Icing = round(log(500 * CW * 1000)) + VVcor + Tcor";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icing"));

}

void icing::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to icing
	 * - name ICEIND-N
	 * - univ_id 480
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("ICING-N", 480);

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void icing::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	params VvParam = { param("VV-MS"), param("VV-MMS")};
	param ClParam("CLDWAT-KGKG");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icingThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> TInfo;
		shared_ptr<info> VvInfo;
		shared_ptr<info> ClInfo;

		double VvScale = 1; // Assume we'll have VV-MMS
		
		try
		{
			// Source info for T
			TInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 TParam);
				
			// Source info for Tg
			VvInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 VvParam);

			if (VvInfo)
			{
				VvInfo->First();

				if (VvInfo->Param().Name() == "VV-MS")
				{
					VvScale = 1000;
				}
			}
			
			// Source info for FF
			ClInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 ClParam);
				
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

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Start();
		}
		
		assert(TInfo->Grid()->AB() == VvInfo->Grid()->AB() && TInfo->Grid()->AB() == ClInfo->Grid()->AB());

		SetAB(myTargetInfo, TInfo);
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> VvGrid(VvInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> ClGrid(ClInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() &&
							*myTargetInfo->Grid() == *VvInfo->Grid() &&
							*myTargetInfo->Grid() == *ClInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		string deviceType = "CPU";
		
		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double T = kFloatMissing;
			double Vv = kFloatMissing;
			double Cl = kFloatMissing;

			InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
			InterpolateToPoint(targetGrid, VvGrid, equalGrids, Vv);
			InterpolateToPoint(targetGrid, ClGrid, equalGrids, Cl);

			if (T == kFloatMissing || Vv == kFloatMissing || Cl == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			double Icing;
			double TBase = 273.15;
			int vCor = kHPMissingInt;
			int tCor = kHPMissingInt;

			T = T - TBase;
			Vv *= VvScale;
			
			// Vertical velocity correction factor

			if (Vv < 0)
			{
				vCor = -1;
			}
			else if ((Vv >= 0) && (Vv <= 50))
			{
				vCor = 0;
			}
			else if ((Vv >= 50) && (Vv <= 100))
			{
				vCor = 1;
			}
			else if ((Vv >= 100) && (Vv <= 200))
			{
				vCor = 2;
			}
			else if ((Vv >= 200) && (Vv <= 300))
			{
				vCor = 3;
			}
			else if ((Vv >= 300) && (Vv <= 1000))
			{
				vCor = 4;
			}
			else if (Vv > 1000)
			{
				vCor = 5;
			}

			// Temperature correction factor

			if ((T <= 0) && (T > -1))
			{
				tCor = -2;
			}
			else if ((T <= -1) && (T > -2))
			{
				tCor = -1;
			}
			else if ((T <= -2) && (T > -3))
			{
				tCor = 0;
			}
			else if ((T <= -3) && (T > -12))
			{
				tCor = 1;
			}
			else if ((T <= -12) && (T > -15))
			{
				tCor = 0;
			}
			else if ((T <= -15) && (T > -18))
			{
				tCor = -1;
			}
			else if (T < -18)
			{
				tCor = -2;
			}
			else 
			{
				tCor = 0;
			}

			if ((Cl <= 0) || (T > 0))
			{
				Icing = 0;
			}
			else {
				Icing = round(log(500 * Cl * 1000)) + vCor + tCor;
			}

			// Maximum and minimum values for index

			if (Icing > 15)
			{
				Icing = 15;
			}

			if (Icing < 0)
			{
				Icing = 0;
			}

			if (!myTargetInfo->Value(Icing))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

		/*
		 * Newbase normalizes scanning mode to bottom left -- if that's not what
		 * the target scanning mode is, we have to swap the data back.
		 */

		SwapTo(myTargetInfo, kBottomLeft);

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Stop();
			itsConfiguration->Statistics()->AddToProcessingTime(processTimer->GetTime());

#ifdef DEBUG
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on " + deviceType);
#endif
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}
