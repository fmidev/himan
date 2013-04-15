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
#include "writer.h"
#include "pcuda.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const double kValueEpsilon = 0.00001;

icing::icing() : itsUseCuda(false)
{
	itsClearTextFormula = "Icing = round(log(500 * CW * 1000)) + VVcor + Tcor";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icing"));

}

void icing::Process(std::shared_ptr<const plugin_configuration> conf)
{

	shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	if (c && c->HaveCuda())
	{
		string msg = "I possess the powers of CUDA ";

		if (!conf->UseCuda())
		{
			msg += ", but I won't use them";
		}
		else
		{
			msg += ", and I'm not afraid to use them";
			itsUseCuda = true;
		}

		itsLogger->Info(msg);

	}

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());

	shared_ptr<info> targetInfo = conf->Info();

	boost::thread_group g;

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

	theRequestedParam.GribParameter(103);
	theRequestedParam.GribTableVersion(203);

	theParams.push_back(theRequestedParam);

	targetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	targetInfo->Create();

	/*
	 * Initialize parent class functions for dimension handling
	 */

	Dimension(conf->LeadingDimension());
	FeederInfo(shared_ptr<info> (new info(*targetInfo)));
	FeederInfo()->Param(theRequestedParam);

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > targetInfos;

	targetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

		boost::thread* t = new boost::thread(&icing::Run,
											 this,
											 targetInfos[i],
											 conf,
											 i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->FileWriteOption() == kSingleFile)
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
}

void icing::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, conf, theThreadIndex);
	}
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void icing::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param VvParam("VV-MMS");
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

		try
		{
			// Source info for T
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 TParam);
				
			// Source info for Tg
			VvInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 VvParam);

			// Source info for FF
			ClInfo = theFetcher->Fetch(conf,
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
				continue;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
			}
		}

		assert(TInfo->Grid()->AB() == VvInfo->Grid()->AB() && TInfo->Grid()->AB() == ClInfo->Grid()->AB());

		SetAB(myTargetInfo, TInfo);
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> VvGrid(VvInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> ClGrid(ClInfo->Grid()->ToNewbaseGrid());

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() &&
							*myTargetInfo->Grid() == *VvInfo->Grid() &&
							*myTargetInfo->Grid() == *ClInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

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
			int vCor;
			int tCor;

			T = T - TBase;

			// Vertical velocity correction factor

			if (Vv < 0)
			{
				vCor = -1;
			}
			else if ((Vv >= 0) && (Vv <= 50))
			{
				vCor = 1;
			}
			else if ((Vv >= 50) && (Vv <= 100))
			{
				vCor = 2;
			}
			else if ((Vv >= 100) && (Vv <= 200))
			{
				vCor = 3;
			}
			else if ((Vv >= 200) && (Vv <= 300))
			{
				vCor = 4;
			}
			else if ((Vv >= 300) && (Vv <= 1000))
			{
				vCor = 5;
			}
			else
			{
				vCor = 0;
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
				tCor = 2;
			}
			else if (T <= -18)
			{
				tCor = 3;
			}
			else
			{
				tCor = 0;
			}

			if ((fabs(Cl - 0) < kValueEpsilon) || (T > 0))
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

			else if (Icing < 0)
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


		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf);
		}
	}
}
