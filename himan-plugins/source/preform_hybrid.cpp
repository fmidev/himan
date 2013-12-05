/**
 * @file preform_hybrid.cpp
 *
 * @date Sep 5, 2013
 * @author partio
 *
 */

#include "preform_hybrid.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "hitool.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

// Korkein sallittu pilven alarajan korkeus, jotta kysessä stratus [m]
const double baseLimit = 300.;

// Vaadittu min. stratuksen paksuus tihkussa ja jäätävässä tihkussa [m]
const double stLimit = 500.;
const double fzStLimit = 800.;

// Kerroksen paksuus pinnasta [m], josta etsitään stratusta (min. BaseLimit+FZstLimit)
const double layer = 2000.;

// Oletus N-kynnysarvo vaaditulle min. stratuksen määrälle [%] (50=yli puoli taivasta)
const double stCover = 50.;

// Kylmin sallittu stratuksen topin T ja kylmin sallittu st:n keskim. T [C] jäätävässä tihkussa
const double stTlimit = -12.;

// Kynnysarvo "riittävän yhtenäisen/ei kerroksittaisen" stratuksen keskim. N-arvolle [%]
const double Nlimit = 70.;

// Vaadittu 2m lämpötilaväli (oltava näiden välissä) [C] jäätävässä tihkussa
const double sfcMax = 0.;
const double sfcMin = -10.;

// Max. sallittu keskim. RH-arvo (suht. kosteus) stratuksen yläpuoliselle kerrokselle [%] (jäätävässä) tihkussa
const double dryLimit = 70.;

// Kynnysarvo vaaditulle stratuksen yläpuolisen kuivan kerroksen paksuudelle [m] (jäätävässä) tihkussa
const double drydz = 1500.;

// Paksuuden raja-arvot vesi- ja lumisateelle [m]
const double waterLim = 1300.;
const double snowLim = 1288.;

// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double FZdzLim = 0.2;

const double kKelvin = 273.15;

const double fzraMA = -100.;
const double fzraPA = 100.;

// Pinnan yläpuolisen plussakerroksen pinta-alan raja-arvot [mC, "metriastetta"]:
const double waterArea = 300;  // alkup. PK:n arvo oli 300
const double snowArea = 50;    // alkup. PK:n arvo oli 50

// Raja-arvot Koistisen olomuotokaavalle (probWater):
// Huom! Käytetään alla sekä IF- että ELSEIF-haaroissa.
const double waterToSleet = 0.5;  // alkup. PK:n arvo oli 0.5
const double sleetToWater = 0.8;  // alkup. PK:n arvo oli 0.8
const double sleetToSnow = 0.2;   // alkup. PK:n arvo oli 0.2

const double wMax = 50;

preform_hybrid::preform_hybrid()
{
	itsClearTextFormula = "<algorithm>";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preform_hybrid"));

}

void preform_hybrid::Process(std::shared_ptr<const plugin_configuration> conf)
{

	unique_ptr<timer> aTimer;

	// Get number of threads to use

	short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->StatisticsEnabled())
	{
		aTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		aTimer->Start();
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedGPUCount(conf->CudaDeviceCount());
	}
	
	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

	/*
	 * Set target parameter to preform_hybrid.
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	param targetParam("PRECFORM-N", 57);

	/*
	 * !!! HUOM !!!
	 *
	 * GRIB2 precipitation type <> FMI precipitation form
	 *
	 * FMI:
	 * 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade
	 *
	 * GRIB2:
	 * 0 = Reserved, 1 = Rain, 2 = Thunderstorm, 3 = Freezing Rain, 4 = Mixed/Ice, 5 = Snow
	 *
	 */

	targetParam.GribDiscipline(0);
	targetParam.GribCategory(1);
	targetParam.GribParameter(19);

	if (conf->OutputFileType() == kGRIB2)
	{
		itsLogger->Error("GRIB2 output requested, conversion between FMI precipitation form and GRIB2 precipitation type is not lossless");
		return;
	}

	params.push_back(targetParam);
	
	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		StoreGrib1ParameterDefinitions(params, targetInfo->Producer().TableVersion());
	}

	targetInfo->Params(params);

	/*
	 * Create data structures.
	 */

	targetInfo->Create();

	/*
	 * Initialize parent class functions for dimension handling
	 */

	if (Dimension() != kTimeDimension)
	{
		itsLogger->Warning("Forcing leading dimension to time");
		Dimension(kTimeDimension);
	}

	FeederInfo(shared_ptr<info> (new info(*targetInfo)));
	FeederInfo()->ParamIndex(0);

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToInitTime(aTimer->GetTime());
		aTimer->Start();
	}

	/*
	 * Each thread will have a copy of the target info.
	 */
	
	for (short i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&preform_hybrid::Run,
											 this,
											 shared_ptr<info> (new info(*targetInfo)),
											 conf,
											 i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToProcessingTime(aTimer->GetTime());
	}

	if (conf->FileWriteOption() == kSingleFile)
	{
		WriteToFile(conf, targetInfo);
	}
}

void preform_hybrid::Run(shared_ptr<info> myTargetInfo,
			   shared_ptr<const plugin_configuration> conf,
			   unsigned short threadIndex)
{

	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, conf, threadIndex);
	}

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void preform_hybrid::Calculate(shared_ptr<info> myTargetInfo,
					 shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
{

	assert(fzStLimit >= stLimit);

	shared_ptr<fetcher> aFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param RRParam("RR-1-MM"); // one hour prec -- should we interpolate in forecast step is 3/6 hours ?
	param TParam("T-K");
	param RHParam("RH-PRCNT");

	level surface0mLevel(kHeight, 0);

	// surface0mLevel = LevelTransform(); TOTEUTA TÄMÄ
	
	level surface2mLevel(kHeight, 2);

	//level groundLevel = LevelTransform(conf->SourceProducer(), TParam, level(kHeight, 2));

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preformHybridThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

//	bool useCudaInThisThread = conf->UseCuda() && threadIndex <= conf->CudaDeviceCount();

	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	
	const param stratusBaseParam("STRATUS-BASE-M");
	const param stratusTopParam("STRATUS-TOP-M");
	const param stratusTopTempParam("STRATUS-TOP-T-K");
	const param stratusMeanTempParam("STRATUS-MEAN-T-K");
	const param stratusMeanCloudinessParam("STRATUS-MEAN-N-PRCNT");
	const param stratusUpperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	const param stratusVerticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MS");

	const param minusAreaParam("MINUS-AREA-T-C");
	const param plusArea1Param("PLUS-AREA-1-T-C");
	const param plusArea2Param("PLUS-AREA-2-T-C");

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		h->Time(myTargetInfo->Time());
		
		// Source infos

		shared_ptr<info> RRInfo, TInfo, RHInfo;
		
		try
		{
			RRInfo = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 RRParam);

			TInfo = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 TParam);

			RHInfo = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 surface2mLevel,
								 RHParam);

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing);

					if (conf->StatisticsEnabled())
					{
						conf->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						conf->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
					}

					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
		}

		//SetAB(myTargetInfo, TInfo);

		size_t missingCount = 0;
		size_t count = 0;

		auto stratus = h->Stratus(conf, myTargetInfo->Time());
		stratus->First();

		myThreadedLogger->Info("Stratus calculated");

		
		auto freezingArea = h->FreezingArea(conf, myTargetInfo->Time());
		freezingArea->First();

		myThreadedLogger->Info("Freezing area calculated");

		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		string deviceType;

		{

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();
			stratus->ResetLocation();
			freezingArea->ResetLocation();

			targetGrid->Reset();

			myTargetInfo->Grid()->Data()->Fill(kFloatMissing);

			while (myTargetInfo->NextLocation()	&& targetGrid->Next()
						&& stratus->NextLocation()
						&& freezingArea->NextLocation()
						&& TInfo->NextLocation()
						&& RRInfo->NextLocation()
						&& RHInfo->NextLocation())
			{

				count++;

				assert(stratus->Param(stratusBaseParam));
				double base = stratus->Value();

				assert(stratus->Param(stratusTopParam));
				double top = stratus->Value();

				assert(stratus->Param(stratusUpperLayerRHParam));
				double upperLayerRH = stratus->Value();

				assert(stratus->Param(stratusVerticalVelocityParam));
				double wAvg = stratus->Value();

				assert(stratus->Param(stratusMeanCloudinessParam));
				double Navg = stratus->Value();

				assert(stratus->Param(stratusMeanTempParam));
				double stTavg = stratus->Value();

				assert(stratus->Param(stratusTopTempParam));
				double Ttop = stratus->Value();
				
				assert(freezingArea->Param(plusArea1Param));
				double plusArea1 = freezingArea->Value();

				assert(freezingArea->Param(plusArea2Param));
				double plusArea2 = freezingArea->Value();

				assert(freezingArea->Param(minusAreaParam));
				double minusArea = freezingArea->Value();

				// Data retrieved directly from database

				double RR = RRInfo->Value();
				double T = TInfo->Value();
				double RH = RHInfo->Value();

				//InterpolateToPoint(targetGrid, RRGrid, equalGrids, RR);
				//InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				
				if (base == kFloatMissing || 
						top == kFloatMissing ||
						upperLayerRH == kFloatMissing ||
						wAvg == kFloatMissing ||
						RR == kFloatMissing ||
						RR == 0 || // No rain --> no rain type
						T == kFloatMissing)
				{
					missingCount++;

					continue;
				}

				cout << "Stratus base " << base << " top " << top << " RH " << upperLayerRH << " ";
				cout << "Freezing area plus1 " << plusArea1 << " plus2 " << plusArea2 << " minus " << minusArea << endl;

				int PreForm = static_cast<int> (kFloatMissing);

				// Unit conversions

				T -= kKelvin;

				const double probWater = util::WaterProbability(T, RH);

				// Start algorithm

				if ((RR <= FZdzLim) && 
					(base < baseLimit) &&
					(top - base >= fzStLimit) &&
					(wAvg < wMax) &&
					(wAvg >= 0) &&
					(Navg > Nlimit) &&
					(Ttop > stTlimit) &&
					(stTavg > stTlimit) &&
					(T > sfcMin) &&
					(T < sfcMax) &&
					(upperLayerRH < dryLimit))
				{
					PreForm = kFreezingDrizzle;
				}
				
				if (plusArea1 != kFloatMissing && minusArea != kFloatMissing)
				{
						
					if (RR > 0.1 && plusArea1 > fzraPA && minusArea < fzraMA && T <= 0)
					{
						PreForm = kFreezingRain;
					}
					else if (plusArea1 > waterArea)
					{
						if (RR < dzLim && base < baseLimit && (top - base) > stLimit && Navg > Nlimit && upperLayerRH < dryLimit)
						{
							PreForm = kDrizzle;
						}
						else
						{
							PreForm = kRain;

							if (probWater < waterToSleet)
							{
								PreForm = kSleet;
							}
						}
					}

					// We probably should have else if here !

					if (plusArea1 > snowArea && plusArea1 <= waterArea)
					{
						PreForm = kSleet;

						if (probWater > sleetToWater)
						{
							if (RR < dzLim && base < baseLimit && (top-base) > stLimit && Navg > Nlimit && upperLayerRH < dryLimit)
							{
								PreForm = kDrizzle;
							}
							else
							{
								PreForm = kRain;
							}
						}
						else if (probWater < sleetToSnow)
						{
							PreForm = kSnow;
						}
					}
				}

				if (!myTargetInfo->Value(PreForm))
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

		if (conf->StatisticsEnabled())
		{
			conf->Statistics()->AddToMissingCount(missingCount);
			conf->Statistics()->AddToValueCount(count);
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (conf->FileWriteOption() != kSingleFile)
		{
			WriteToFile(conf, myTargetInfo);
		}

	}
}
