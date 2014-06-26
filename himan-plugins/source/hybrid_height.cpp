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

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("hybrid_height");

hybrid_height::hybrid_height() : itsFastMode(false)
{
	itsClearTextFormula = "HEIGHT = prevH + (287/9.81) * (T+prevT)/2 * log(prevP / P)";
	itsLogger = logger_factory::Instance()->GetLog(itsName);

}

void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("HL-M", 3, 0, 3, 6)});

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

	const param GPParam("P-PA");
	const param PParam("P-HPA");
	const param TParam("T-K");
	const param ZParam("Z-M2S2");
	
	const level H2(himan::kHeight, 2, "HEIGHT");
	const level H0(himan::kHeight, 0, "HEIGHT");

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	/*
		pitääkö tunnistaa tuottaja?
	*/

	level prevLevel;

	bool firstLevel = false;
		
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


	info_t TInfo, PInfo, prevPInfo, prevTInfo, prevHInfo, GPInfo, zeroGPInfo;

	if (itsUseGeopotential)
	{
		GPInfo = Fetch(forecastTime, forecastLevel, ZParam, false);
		zeroGPInfo = Fetch(forecastTime, H0, ZParam, false);

		SetAB(myTargetInfo, GPInfo);

	}
	else
	{

		if (!firstLevel)
		{
			prevTInfo = Fetch(forecastTime, prevLevel, TParam, false);
			prevPInfo = Fetch(forecastTime, prevLevel, PParam, false);
			prevHInfo = Fetch(forecastTime, prevLevel, param("HL-M"), false);
		}
		else
		{
			prevPInfo = Fetch(forecastTime, H0, GPParam, false);
			prevTInfo = Fetch(forecastTime, H2, TParam, false);
		}

		PInfo = Fetch(forecastTime, forecastLevel, PParam, false);
		TInfo = Fetch(forecastTime, forecastLevel, TParam, false);

		assert(PInfo->Grid()->AB() == TInfo->Grid()->AB());

		SetAB(myTargetInfo, TInfo);

	}

	if ((itsUseGeopotential && (!GPInfo || !zeroGPInfo)) || (!itsUseGeopotential && (!prevTInfo || !prevPInfo || !prevHInfo || !PInfo || !TInfo)))
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}
	
	string deviceType = "CPU";

	if (itsUseGeopotential)
	{
		GPInfo->ResetLocation();
		zeroGPInfo->ResetLocation();
	}
	else
	{
		PInfo->ResetLocation();
		TInfo->ResetLocation();
		prevPInfo->ResetLocation();
		prevTInfo->ResetLocation();
		prevHInfo->ResetLocation();
	}

	LOCKSTEP(myTargetInfo)
	{

		if (itsUseGeopotential)
		{
			GPInfo->NextLocation();
			double GP = GPInfo->Value();

			zeroGPInfo->NextLocation();
			double zeroGP = zeroGPInfo->Value();
		
			if (GP == kFloatMissing || zeroGP == kFloatMissing)
			{
				continue;
			}

			double height = (GP - zeroGP) * himan::constants::kIg;
			
			myTargetInfo->Value(height);
		}
		else
		{
			TInfo->NextLocation();
			double T = TInfo->Value();

			PInfo->NextLocation();
			double P = PInfo->Value();

			prevTInfo->NextLocation();
			double prevT = prevTInfo->Value();

			prevPInfo->NextLocation();
			double prevP = prevPInfo->Value();

			double prevH = kFloatMissing;
			
			if (!firstLevel)
			{

				prevHInfo->NextLocation();
				prevH = prevHInfo->Value();

				if (prevH == kFloatMissing )
				{
					continue;
				}
			}

			if (prevT == kFloatMissing || prevP == kFloatMissing || T == kFloatMissing || P == kFloatMissing )
			{
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

			myTargetInfo->Value(totalHeight);

		}
	}

	if (!itsFastMode)
	{
		firstLevel = false;
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}
