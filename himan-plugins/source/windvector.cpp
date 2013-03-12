/**
 * @file windvector.cpp
 *
 *  Created on: Jan 21, 2013
 *  @author aaltom
 */

#include "windvector.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"
#include <math.h>
#include "NFmiRotatedLatLonArea.h"
#include "NFmiStereographicArea.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "pcuda.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const double kRadToDeg = 57.295779513082; // 180 / PI

windvector::windvector()
	: itsUseCuda(false)
	, itsSeaCalculation(false)
	, itsIceCalculation(false)
	, itsAirCalculation(false)
{
	itsClearTextFormula = "speed = sqrt(U*U+V*V) ; direction = round(180/PI * atan2(U,V) + offset) ; vector = round(U/10) + 100 * round(V)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvector"));

}

void windvector::Process(std::shared_ptr<const plugin_configuration> conf)
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

	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

	/*
	 * Get producer information from neons
	 */

	if (conf->FileWriteOption() == kNeons)
	{
		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string,string> prodInfo = n->ProducerInfo(targetInfo->Producer().Id());

		if (!prodInfo.empty())
		{
			producer prod(targetInfo->Producer().Id());

			prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
			prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
			prod.Name(prodInfo["name"]);

			targetInfo->Producer(prod);
		}

	}

	/*
	 * Set target parameter to windvector
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param requestedVectorParam("DF-MS", 22);

	param requestedSpeedParam("FF-MS", 21);
	requestedSpeedParam.GribDiscipline(0);
	requestedSpeedParam.GribCategory(2);
	requestedSpeedParam.GribParameter(1);

	param requestedDirParam("DD-D", 20);
	requestedDirParam.GribDiscipline(0);
	requestedDirParam.GribCategory(2);
	requestedDirParam.GribParameter(0);

	if (conf->Exists("for_ice") && conf->GetValue("for_ice") == "true")
	{
		requestedSpeedParam = param("IFF-MS", 389);
		requestedDirParam = param("IDD-D", 390);
		itsIceCalculation = true;
	}
	else if (conf->Exists("for_sea") && conf->GetValue("for_sea") == "true")
	{
		requestedSpeedParam = param("SFF-MS", 163);
		requestedDirParam = param("SDD-D", 164);
		itsSeaCalculation = true;
	}
	else
	{
		itsAirCalculation = true;
	}

	theParams.push_back(requestedSpeedParam);
	theParams.push_back(requestedDirParam);

	if (itsAirCalculation)
	{
		theParams.push_back(requestedVectorParam);
	}

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
	FeederInfo()->ParamIndex(0); // Set index to first param (it doesn't matter which one, as long as its set

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > targetInfos;

	targetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

		boost::thread* t = new boost::thread(&windvector::Run,
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

void windvector::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
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

void windvector::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param UParam("U-MS");
	param VParam("V-MS");

	double directionOffset = 180; // For wind direction add this

	if (itsSeaCalculation)
	{
		UParam = param("WVELU-MS");
		VParam = param("WVELV-MS");
		directionOffset = 0;
	}
	else if (itsIceCalculation)
	{
		UParam = param("IVELU-MS");
		VParam = param("IVELV-MS");
		directionOffset = 0;
	}

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvectorThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->ParamIndex(0);

	// Fetch source level definition

	level sourceLevel = compiled_plugin_base::LevelTransform(conf->SourceProducer(), UParam, myTargetInfo->PeakLevel(0));

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> UInfo;
		shared_ptr<info> VInfo;

		try
		{
			// Source info for U
			UInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 sourceLevel,
								 UParam);
				
			// Source info for V
			VInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 sourceLevel,
								 VParam);
				
		}
		catch (HPExceptionType e)
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

		// if source producer is Hirlam, we must de-stagger U and V grid

		/*if (conf->SourceProducer().Id() == 1 && sourceLevel.Type() != kHeight)
		{
			UInfo->Grid()->Stagger(-0.5, 0);
			VInfo->Grid()->Stagger(0, -0.5);
		}*/

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> UGrid(UInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> VGrid(VInfo->Grid()->ToNewbaseGrid());

		UGrid->InterpolationMethod(kNearestPoint);
		VGrid->InterpolationMethod(kNearestPoint);

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *UInfo->Grid() && *myTargetInfo->Grid() == *VInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		assert(UInfo->Grid()->Projection() == VInfo->Grid()->Projection());

		bool needRotLatLonGridRotation = (UInfo->Grid()->Projection() == kRotatedLatLonProjection && UInfo->Grid()->UVRelativeToGrid());
		bool needStereographicGridRotation = (UInfo->Grid()->Projection() == kStereographicProjection && UInfo->Grid()->UVRelativeToGrid());

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double U = kFloatMissing;
			double V = kFloatMissing;

			InterpolateToPoint(targetGrid, UGrid, equalGrids, U);
			InterpolateToPoint(targetGrid, VGrid, equalGrids, V);

			if (U == kFloatMissing || V == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->ParamIndex(0);
				myTargetInfo->Value(kFloatMissing);
				myTargetInfo->ParamIndex(1);
				myTargetInfo->Value(kFloatMissing);

				if (itsAirCalculation)
				{
					myTargetInfo->ParamIndex(2);
					myTargetInfo->Value(kFloatMissing);
				}

				continue;
			}

			if (needRotLatLonGridRotation)
			{
				/*
				 * 1. Get coordinates of current grid point in earth-relative form
				 * 2. Get coordinates of current grid point in grid-relative form
				 * 3. Call function UVToEarthRelative() that transforms U and V from grid-relative
				 *    to earth-relative
				 */

				assert(UGrid->Area()->ClassId() == kNFmiRotatedLatLonArea);

				const point regPoint(targetGrid->LatLon());

				const point rotPoint(reinterpret_cast<NFmiRotatedLatLonArea*> (UGrid->Area())->ToRotLatLon(regPoint.ToNFmiPoint()));

				point regUV = util::UVToEarthRelative(regPoint, rotPoint, UInfo->Grid()->SouthPole(), point(U,V));

				// Wind speed should the same with both forms of U and V if no interpolation is done

				assert(!equalGrids || fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);

				U = regUV.X();
				V = regUV.Y();
			}
			else if (needStereographicGridRotation)
			{
				assert(UGrid->Area()->ClassId() == kNFmiStereographicArea);

				double centralLongitude = (reinterpret_cast<NFmiStereographicArea*> (targetGrid->Area())->CentralLongitude());

				point regUV = util::UVToGeographical(centralLongitude, point(U,V));

				// Wind speed should the same with both forms of U and V if no interpolation is done

				assert(!equalGrids || fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);

			}

			double speed = sqrt(U*U + V*V);

			double dir = 0;

			if (speed > 0)
			{
				dir = round(kRadToDeg * atan2(U,V) + directionOffset); // Rounding dir

				if (dir < 0)
				{
					dir += 360;
				}
				else if (dir > 360)
				{
					dir -= 360;
				}
			}

			/*
			 * The order of parameters in infos is and must be always:
			 * index 0 : speed parameter
			 * index 1 : direction parameter
			 * index 2 : vector parameter (optional)
			 */

			myTargetInfo->ParamIndex(0);
			myTargetInfo->Value(speed);

			myTargetInfo->ParamIndex(1);
			myTargetInfo->Value(dir);

			if (itsAirCalculation)
			{

				double windVector = round(U/10) + 100 * round(V);

				myTargetInfo->ParamIndex(2);

				if (!myTargetInfo->Value(windVector))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}
			}
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			shared_ptr<info> tempInfo(new info(*myTargetInfo));

			for (tempInfo->ResetParam(); tempInfo->NextParam(); )
			{
				theWriter->ToFile(tempInfo, conf);
			}
		}
	}
}
