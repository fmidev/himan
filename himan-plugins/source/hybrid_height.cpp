/**
 * @file hybrid_height.cpp
 *
 * @date Apr 5, 2013
 * @author peramaki
 */

#include "hybrid_height.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <math.h>
#include "NFmiGrid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("hybrid_height");

hybrid_height::hybrid_height() : itsFastMode(false)
{
	itsClearTextFormula = "HEIGHT = prevH + (287/9.81) * (T+prevT)/2 * log(prevP / P)";
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to HL-M
	 * - name HL-M
	 * - univ_id 3
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("HL-M", 3);

	// GRIB 2

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(6);

	// GRIB 1

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	/*
	 * For hybrid height we must go through the levels backwards.
	 */

	itsInfo->LevelOrder(kBottomToTop);

	shared_ptr<neons> theNeons = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

	itsBottomLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));

	itsUseGeopotential = (itsConfiguration->SourceProducer().Id() == 1 || itsConfiguration->SourceProducer().Id() == 199);

	if (!itsConfiguration->Exists("fast_mode") && itsConfiguration->GetValue("fast_mode") == "true")
	{
		itsFastMode = true;
	}
	else if (!itsUseGeopotential)
	{
		// When doing exact calculation we must do them sequentially starting from
		// surface closest to ground because every surface's value is depended
		// on the surface below it. Therefore we cannot parallelize the calculation
		// on level basis.
		
		if (Dimension() != kTimeDimension)
		{
			itsLogger->Info("Changing leading_dimension to time");
			Dimension(kTimeDimension);
		}
	}

	Start();
	
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void hybrid_height::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	const param GPParam("P-PA");
	const param PParam("P-HPA");
	const param TParam("T-K");
	const param ZParam("Z-M2S2");
	
	const level H2(himan::kHeight, 2, "HEIGHT");
	const level H0(himan::kHeight, 0, "HEIGHT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	/*
		pitääkö tunnistaa tuottaja?
	*/
	level prevLevel;

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		bool firstLevel(false);
		
		if (itsFastMode || myTargetInfo->Level().Value() == itsBottomLevel)
		{
			firstLevel = true;
		}
		else
		{
			prevLevel = level(myTargetInfo->Level());
			prevLevel.Value(myTargetInfo->Level().Value() + 1);

			prevLevel.Index(prevLevel.Index() + 1);
		}


		shared_ptr<info> PInfo;
		shared_ptr<info> TInfo;			
		shared_ptr<info> prevPInfo;
		shared_ptr<info> prevTInfo;
		shared_ptr<info> prevHInfo;
		shared_ptr<info> GPInfo;
		shared_ptr<info> zeroGPInfo;

		try
		{
			const forecast_time fTime = myTargetInfo->Time();
			
			if (itsUseGeopotential)
			{
				GPInfo = FetchPrevious(fTime, myTargetInfo->Level(), ZParam);
				zeroGPInfo = FetchPrevious(fTime, H0, ZParam);
			
				SetAB(myTargetInfo, GPInfo);

			}
			else
			{

				if (!firstLevel)
				{
					prevTInfo = FetchPrevious(fTime, prevLevel, param("T-K"));
					prevPInfo = FetchPrevious(fTime, prevLevel, param("P-HPA"));
					prevHInfo = FetchPrevious(fTime, prevLevel, param("HL-M"));
				}
				else
				{
					prevPInfo = FetchPrevious(fTime, H0, param("P-PA"));
					prevTInfo = FetchPrevious(fTime, H2, param("T-K"));
				}

				PInfo = FetchPrevious(fTime, myTargetInfo->Level(), PParam);
				TInfo = FetchPrevious(fTime, myTargetInfo->Level(), TParam);

				assert(PInfo->Grid()->AB() == TInfo->Grid()->AB());

				SetAB(myTargetInfo, TInfo);

			}


		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
				myTargetInfo->Data()->Fill(kFloatMissing);

				if (itsConfiguration->StatisticsEnabled())
				{
					itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
					itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
				}

				continue;
			}
			else
			{
				throw;
			}
		}

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Start();
		}

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> PGrid;
		shared_ptr<NFmiGrid> prevPGrid;
		shared_ptr<NFmiGrid> TGrid;
		shared_ptr<NFmiGrid> prevTGrid;
		shared_ptr<NFmiGrid> prevHGrid;
		shared_ptr<NFmiGrid> GPGrid;
		shared_ptr<NFmiGrid> zeroGPGrid;

		bool equalGrids = false;
		
		if (itsUseGeopotential)
		{
			GPGrid = make_shared<NFmiGrid> (*GPInfo->Grid()->ToNewbaseGrid());
			zeroGPGrid = make_shared<NFmiGrid> (*zeroGPInfo->Grid()->ToNewbaseGrid());

			equalGrids = (*myTargetInfo->Grid() == *zeroGPInfo->Grid() && *myTargetInfo->Grid() == *GPInfo->Grid()) ;
		}
		else
		{
			PGrid = make_shared<NFmiGrid> (*PInfo->Grid()->ToNewbaseGrid());
			prevPGrid = make_shared<NFmiGrid> (*prevPInfo->Grid()->ToNewbaseGrid());
			TGrid = make_shared<NFmiGrid> (*TInfo->Grid()->ToNewbaseGrid());
			prevTGrid = make_shared<NFmiGrid> (*prevTInfo->Grid()->ToNewbaseGrid());

			if (!firstLevel)
			{
				prevHGrid = make_shared<NFmiGrid> (*prevHInfo->Grid()->ToNewbaseGrid());

				equalGrids = ( *myTargetInfo->Grid() == *prevTInfo->Grid() && *myTargetInfo->Grid() == *prevPInfo->Grid() && *myTargetInfo->Grid() == *PInfo->Grid() && *myTargetInfo->Grid() == *TInfo->Grid() ); //&& *myTargetInfo->Grid() == *T2mInfo->Grid() && *myTargetInfo->Grid() == *P0mInfo->Grid() );

				if (!firstLevel)
				{
					equalGrids = ( equalGrids && *myTargetInfo->Grid() == *prevHInfo->Grid() );
				}
			}
		}

		string deviceType = "CPU";

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{

			count++;

			double T = kFloatMissing;
			double P = kFloatMissing;
			double prevP = kFloatMissing;
			double prevT = kFloatMissing;
			double prevH = kFloatMissing;

			if (itsUseGeopotential)
			{
				double GP = kFloatMissing, zeroGP = kFloatMissing;

				InterpolateToPoint(targetGrid, GPGrid, equalGrids, GP);
				InterpolateToPoint(targetGrid, zeroGPGrid, equalGrids, zeroGP);

				double height = kFloatMissing;

				if (GP != kFloatMissing && zeroGP != kFloatMissing)
				{
					height = (GP - zeroGP) * himan::constants::kIg;
				}
				else
				{
					missingCount++;
				}

				myTargetInfo->Value(height);

				continue;
			}

			InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
			InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
			InterpolateToPoint(targetGrid, prevPGrid, equalGrids, prevP);		
			InterpolateToPoint(targetGrid, prevTGrid, equalGrids, prevT);
		
			if (!firstLevel)
			{
				prevHGrid->Next();
				InterpolateToPoint(targetGrid, prevHGrid, equalGrids, prevH);
				
				if (prevH == kFloatMissing )
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}
			}

			if (prevT == kFloatMissing || prevP == kFloatMissing || T == kFloatMissing || P == kFloatMissing )
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}


			if (firstLevel)
			{
				prevP /= 100.f;
			}

			double Tave = ( T + prevT ) / 2;
			double deltaZ = (287 / 9.81) * Tave * log(prevP / P);

			double totalHeight(0);

			if (firstLevel)
			{
				totalHeight = deltaZ;		
			}
			else
			{	
				totalHeight = prevH + deltaZ;
			}

			if (!myTargetInfo->Value(totalHeight))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}
		
		}

		if (!itsFastMode)
		{
			firstLevel = false;
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
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on "  + deviceType);
#endif

			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 * */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}

shared_ptr<himan::info> hybrid_height::FetchPrevious(const forecast_time& wantedTime, const level& wantedLevel, const param& wantedParam)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		return f->Fetch(itsConfiguration,
						wantedTime,
						wantedLevel,
						wantedParam);
   	}
	catch (HPExceptionType& e)
	{
		throw;
	}

}
