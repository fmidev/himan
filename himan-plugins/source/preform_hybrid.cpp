/**
 * @file preform_hybrid.cpp
 *
 * @date Sep 5, 2013
 * @author partio
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
using namespace himan;
using namespace himan::plugin;

void Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result);
void FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result);

// Korkein sallittu pilven alarajan korkeus, jotta kysessä stratus [m]
const double baseLimit = 300.;

// Vaadittu min. stratuksen paksuus tihkussa ja jäätävässä tihkussa [m]
const double stLimit = 500.;
const double fzStLimit = 800.;

// Kylmin sallittu stratuksen topin T ja kylmin sallittu st:n keskim. T [C] jäätävässä tihkussa
const double stTlimit = -12.;

// Kynnysarvo "riittävän yhtenäisen/ei kerroksittaisen" stratuksen keskim. N-arvolle [%]
const double Nlimit = 70.;

// Vaadittu 2m lämpötilaväli (oltava näiden välissä) [C] jäätävässä tihkussa
const double sfcMax = 0.;
const double sfcMin = -10.;

// Max. sallittu keskim. RH-arvo (suht. kosteus) stratuksen yläpuoliselle kerrokselle [%] (jäätävässä) tihkussa
const double dryLimit = 70.;

// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double FZdzLim = 0.2;

// Raja-arvot pinnan pakkaskerroksen (MA) ja sen yläpuolisen sulamiskerroksen (PA) pinta-alalle jäätävässä sateessa [mC]
const double fzraMA = -100.;
const double fzraPA = 100.;

// Pinnan yläpuolisen plussakerroksen pinta-alan raja-arvot [mC, "metriastetta"]:
const double waterArea = 300;  // alkup. PK:n arvo oli 300
const double snowArea = 50;	// alkup. PK:n arvo oli 50

// Raja-arvot Koistisen olomuotokaavalle (probWater):
// Huom! Käytetään alla sekä IF- että ELSEIF-haaroissa.
const double waterToSleet = 0.5;  // alkup. PK:n arvo oli 0.5
const double sleetToWater = 0.8;  // alkup. PK:n arvo oli 0.8
const double sleetToSnow = 0.2;   // alkup. PK:n arvo oli 0.2

// Max sallittu nousuliike st:ssa [mm/s]
const double wMax = 50;

preform_hybrid::preform_hybrid()
{
	itsClearTextFormula = "<algorithm>";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preform_hybrid"));

}

void preform_hybrid::Process(std::shared_ptr<const plugin_configuration> conf)
{
	// Initialize plugin

	Init(conf);

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

	if (itsConfiguration->OutputFileType() == kGRIB2)
	{
		itsLogger->Error("GRIB2 output requested, conversion between FMI precipitation form and GRIB2 precipitation type is not lossless");
		return;
	}

	params.push_back(targetParam);

	SetParams(params);

	/*
	 * Initialize parent class functions for dimension handling
	 */

	if (Dimension() != kTimeDimension)
	{
		itsLogger->Warning("Forcing leading dimension to time");
		Dimension(kTimeDimension);
	}

	Start();
	
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void preform_hybrid::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
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

	const param stratusBaseParam("STRATUS-BASE-M");
	const param stratusTopParam("STRATUS-TOP-M");
	const param stratusTopTempParam("STRATUS-TOP-T-K");
	const param stratusMeanTempParam("STRATUS-MEAN-T-K");
	const param stratusMeanCloudinessParam("STRATUS-MEAN-N-PRCNT");
	const param stratusUpperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	const param stratusVerticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MS");

	const param minusAreaParam("MINUS-AREA-T-K");
	const param plusArea1Param("PLUS-AREA-1-T-K");
	// const param plusArea2Param("PLUS-AREA-2-T-K");

	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		h->Time(myTargetInfo->Time());

		// Source infos

		shared_ptr<info> RRInfo, TInfo, RHInfo;
		
		try
		{
			RRInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 RRParam);

			TInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 TParam);

			RHInfo = aFetcher->Fetch(itsConfiguration,
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

		//SetAB(myTargetInfo, TInfo);

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<info> stratus;
		shared_ptr<info> freezingArea;

		/*
		 * Spinoff thread will calculate freezing area while main thread calculates
		 * stratus.
		 *
		 * The constructor of std::thread (and boost::thread) deduces argument types
		 * and stores them *by value*.
		 */
		
		boost::thread t(&FreezingArea, itsConfiguration, myTargetInfo->Time(), boost::ref(freezingArea));

		Stratus(itsConfiguration, myTargetInfo->Time(), stratus);

		t.join();

		freezingArea->First();
		stratus->First();

		myThreadedLogger->Info("Stratus and freezing area calculated");
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		string deviceType;

		{

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();
			stratus->ResetLocation();
			TInfo->ResetLocation();
			RRInfo->ResetLocation();
			RHInfo->ResetLocation();
			freezingArea->ResetLocation();

			targetGrid->Reset();

			myTargetInfo->Grid()->Data()->Fill(kFloatMissing);

			assert(myTargetInfo->SizeLocations() == stratus->SizeLocations());
			assert(myTargetInfo->SizeLocations() == freezingArea->SizeLocations());
			assert(myTargetInfo->SizeLocations() == TInfo->SizeLocations());
			assert(myTargetInfo->SizeLocations() == RRInfo->SizeLocations());
			assert(myTargetInfo->SizeLocations() == RHInfo->SizeLocations());

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

				// assert(freezingArea->Param(plusArea2Param));
				// double plusArea2 = freezingArea->Value();

				assert(freezingArea->Param(minusAreaParam));
				double minusArea = freezingArea->Value();

				// Data retrieved directly from database

				double RR = RRInfo->Value();
				double T = TInfo->Value();
				double RH = RHInfo->Value();

				//InterpolateToPoint(targetGrid, RRGrid, equalGrids, RR);
				//InterpolateToPoint(targetGrid, TGrid, equalGrids, T);

				if (RR == kFloatMissing || RR == 0)
				{
					// No rain --> no rain type
					missingCount++;

					continue;
				}
				else if (T == kFloatMissing || RH == kFloatMissing)
				{
					// These variables come directly from database and should
					// not have missing values in regular conditions
					missingCount++;

					continue;
				}

				double PreForm = kFloatMissing;

				// Unit conversions
				
				T -= himan::constants::kKelvin;
				RH *= 100; // 0..1 --> %

				if (Ttop != kFloatMissing)
				{
					Ttop -= himan::constants::kKelvin;
				}

				if (stTavg != kFloatMissing)
				{
					stTavg -= himan::constants::kKelvin;
				}

				if (Navg != kFloatMissing)
				{
					Navg *= 100; // --> %
				}

				if (wAvg != kFloatMissing)
				{
					wAvg *= 1000; // m/s --> mm/s
				}

				bool thickStratusWithLightPrecipitation = (	base			!= kFloatMissing &&
															top				!= kFloatMissing &&
															Navg			!= kFloatMissing &&
															upperLayerRH	!= kFloatMissing &&
															RR				<= dzLim &&
															base			< baseLimit &&
															(top - base)	> stLimit &&
															Navg			> Nlimit &&
															upperLayerRH	< dryLimit);

				const double probWater = util::WaterProbability(T, RH);

				// Start algorithm
				// Possible values for preform: 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade

				if (	base			!=	kFloatMissing &&
						top				!= kFloatMissing &&
						upperLayerRH	!= kFloatMissing &&
						wAvg			!= kFloatMissing &&
						Navg			!= kFloatMissing &&
						stTavg			!= kFloatMissing &&
						Ttop			!= kFloatMissing)
				{

					if ((RR <= FZdzLim) &&
						(base < baseLimit) &&
						((top - base) >= fzStLimit) &&
						(wAvg < wMax) &&
						(wAvg >= 0) &&
						(Navg > Nlimit) &&
						(Ttop > stTlimit) &&
						(stTavg > stTlimit) &&
						(T > sfcMin) &&
						(T <= sfcMax) &&
						(upperLayerRH < dryLimit))
					{
						PreForm = kFreezingDrizzle;
					}
				}
				
				// 2. jäätävää vesisadetta?
				// Löytyykö riittävän paksut: pakkaskerros pinnasta ja sen yläpuolelta plussakerros?
				// (Huom. hyvin paksu pakkaskerros (tai ohut sulamiskerros) -> oikeasti jääjyväsiä/ice pellets fzra sijaan)
				// (Heikoimmat intensiteetit pois, RR>0.1 tms?)
				
				if (PreForm == kFloatMissing && plusArea1 != kFloatMissing && minusArea != kFloatMissing)
				{
						
					if (RR > 0.1 && plusArea1 > fzraPA && minusArea < fzraMA && T <= 0)
					{
						PreForm = kFreezingRain;
					}
					else if (plusArea1 > waterArea)
					{
						if (thickStratusWithLightPrecipitation)
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
			
					// Räntää jos "ei liian paksu lämmin kerros pinnan yläpuolella"

					if (plusArea1 >= snowArea && plusArea1 <= waterArea)
					{
						PreForm = kSleet;

						if (probWater > sleetToWater)
						{
							// Tihkuksi jos riittävän paksu stratus heikolla sateen intensiteetillä
							
							if (thickStratusWithLightPrecipitation)
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

				if (PreForm == kFloatMissing && (plusArea1 < snowArea || plusArea1 == kFloatMissing))
				{
					PreForm = kSnow;
				}

				// FINISHED
				
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

void FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	h->Time(ftime);

	result = h->FreezingArea();

}

void Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));
	
	h->Configuration(conf);
	h->Time(ftime);

	result = h->Stratus();

}