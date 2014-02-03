/**
 * @file preform_pressure.cpp
 *
 * @date Sep 5, 2013
 * @author partio
 *
 */

#define AND &&
#define OR ||

#include "preform_pressure.h"
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

// Paksuuden raja-arvot vesi- ja lumisateelle [m]

const double waterLim = 1300;
const double snowLim = 1288;

// Vaadittu 2m lämpötilaväli (oltava näiden välissä) [C] jäätävässä tihkussa
const double sfcMax = 0;
const double sfcMin = -10;

// Kylmin sallittu stratuksen topin T ja kylmin sallittu st:n keskim. T [C] jäätävässä tihkussa
const double stTlimit = -12;

// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double fzdzLim = 0.2;

 // Max sallittu nousuliike stratuksessa [mm/s] jäätävässä tihkussa (vähentää fzdz esiintymistä)
const double wMax = 50;

// 925 tai 850hPa:n stratuksen ~vähimmäispaksuus [hPa] jäätävässä tihkussa, ja
// sulamiskerroksen (tai sen alapuolisen pakkaskerroksen) vähimmäispaksuus [hPa] jäätävässä sateessa
// (olettaen, että stratuksen/sulamis-/pakkaskerroksen top on 925/850hPa:ssa)
const double stH = 15;

// Raja-arvot Koistisen olomuotokaavalle (probWater):
const double sleetTOwater = 0.8;  // alkup. PK:n arvo oli 0.8
const double sleetTOsnow = 0.2;   // alkup. PK:n arvo oli 0.2
const double waterTOsleet = 0.5;  // alkup. PK:n arvo oli 0.5

// Paksuuden raja-arvot vesi- ja lumisateelle [m]
const double waterThickness = 1300;
const double snowThickness = 1288;

preform_pressure::preform_pressure()
{
	itsClearTextFormula = "<algorithm>";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preform_pressure"));

}

void preform_pressure::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to precipitation form.
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

	Start();
	
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void preform_pressure::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

shared_ptr<fetcher> aFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param RHParam("RH-PRCNT");
	param ZParam("Z-M2S2");
	param RRParam("RR-1-MM"); // one hour prec -- should we interpolate if forecast step is 3/6 hours ?
	param PParam("P-PA");
	param WParam("VV-MMS");
	
	itsConfiguration->FirstSourceProducer();
	
	level groundLevel = LevelTransform(itsConfiguration->SourceProducer(), TParam, level(kHeight, 2));

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);
	level P700(kPressure, 700);
	level P850(kPressure, 850);
	level P925(kPressure, 925);
	level P1000(kPressure, 1000);

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preformPressureThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

//	bool useCudaInThisThread = conf->UseCuda() AND threadIndex <= conf->CudaDeviceCount();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		// Source infos

		shared_ptr<info> TInfo;
		shared_ptr<info> T700Info;
		shared_ptr<info> T850Info;
		shared_ptr<info> T925Info;

		shared_ptr<info> RHInfo;
		shared_ptr<info> RH700Info;
		shared_ptr<info> RH850Info;
		shared_ptr<info> RH925Info;

		shared_ptr<info> Z850Info;
		shared_ptr<info> Z1000Info;

		shared_ptr<info> W925Info;
		shared_ptr<info> W850Info;

		shared_ptr<info> RRInfo;
		shared_ptr<info> PInfo;

		try
		{
			TInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 groundLevel,
								 TParam);

			assert(TInfo->Param().Unit() == kK);

			T700Info = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 P700,
								 TParam);

			assert(T700Info->Param().Unit() == kK);

			T850Info = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 P850,
								 TParam);

			assert(T850Info->Param().Unit() == kK);

			T925Info = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 P925,
								 TParam);

			assert(T925Info->Param().Unit() == kK);

			RHInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 surface2mLevel,
								 RHParam);

			RH700Info = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 P700,
								 RHParam);

			RH850Info = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 P850,
								 RHParam);

			RH925Info = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 P925,
								 RHParam);

			Z850Info = aFetcher->Fetch(itsConfiguration,
					 myTargetInfo->Time(),
					 P850,
					 ZParam);

			Z1000Info = aFetcher->Fetch(itsConfiguration,
					 myTargetInfo->Time(),
					 P1000,
					 ZParam);

			W925Info = aFetcher->Fetch(itsConfiguration,
						 myTargetInfo->Time(),
						 P925,
						 WParam);

			W850Info = aFetcher->Fetch(itsConfiguration,
					 myTargetInfo->Time(),
					 P850,
					 WParam);

			PInfo = aFetcher->Fetch(itsConfiguration,
					 myTargetInfo->Time(),
					 surface0mLevel,
					 PParam);

			RRInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 RRParam);

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

		SetAB(myTargetInfo, TInfo);

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T700Grid(T700Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T925Grid(T925Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> RHGrid(RHInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH700Grid(RH700Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH850Grid(RH850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH925Grid(RH925Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> Z850Grid(Z850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> Z1000Grid(Z1000Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> W925Grid(RH925Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> W850Grid(Z850Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> RRGrid(RRInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = ((*myTargetInfo->Grid() == *TInfo->Grid()) &&
							(*myTargetInfo->Grid() == *T700Info->Grid()) &&
							(*myTargetInfo->Grid() == *T850Info->Grid()) &&
							(*myTargetInfo->Grid() == *T925Info->Grid()) &&
							(*myTargetInfo->Grid() == *RHInfo->Grid()) &&
							(*myTargetInfo->Grid() == *RH700Info->Grid()) &&
							(*myTargetInfo->Grid() == *RH850Info->Grid()) &&
							(*myTargetInfo->Grid() == *RH925Info->Grid()) &&
							(*myTargetInfo->Grid() == *Z850Info->Grid()) &&
							(*myTargetInfo->Grid() == *Z1000Info->Grid()) &&
							(*myTargetInfo->Grid() == *RRInfo->Grid()) &&
							(*myTargetInfo->Grid() == *PInfo->Grid()) &&
							(*myTargetInfo->Grid() == *W850Info->Grid()) &&
							(*myTargetInfo->Grid() == *W925Info->Grid())
							);

		assert(equalGrids);
		
		string deviceType;

		{

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{

				count++;

				double T = kFloatMissing;
				double T700 = kFloatMissing;
				double T850 = kFloatMissing;
				double T925 = kFloatMissing;
				
				double RH = kFloatMissing;
				double RH700 = kFloatMissing;
				double RH850 = kFloatMissing;
				double RH925 = kFloatMissing;

				double Z850 = kFloatMissing;
				double Z1000 = kFloatMissing;

				double W850 = kFloatMissing;
				double W925 = kFloatMissing;

				double P = kFloatMissing;

				double RR = kFloatMissing;

				int PreForm = static_cast<int> (kFloatMissing);

				InterpolateToPoint(targetGrid, RRGrid, equalGrids, RR);

				// No rain --> no rain type

				if (RR == 0)
				{
					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, T700Grid, equalGrids, T700);
				InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
				InterpolateToPoint(targetGrid, T925Grid, equalGrids, T925);

				InterpolateToPoint(targetGrid, RHGrid, equalGrids, RH);
				InterpolateToPoint(targetGrid, RH700Grid, equalGrids, RH700);
				InterpolateToPoint(targetGrid, RH850Grid, equalGrids, RH850);
				InterpolateToPoint(targetGrid, RH925Grid, equalGrids, RH925);

				InterpolateToPoint(targetGrid, Z850Grid, equalGrids, Z850);
				InterpolateToPoint(targetGrid, Z1000Grid, equalGrids, Z1000);

				InterpolateToPoint(targetGrid, W850Grid, equalGrids, W850);
				InterpolateToPoint(targetGrid, W925Grid, equalGrids, W925);

				InterpolateToPoint(targetGrid, PGrid, equalGrids, P);

				if (T == kFloatMissing || T850 == kFloatMissing || T925 == kFloatMissing ||
					RH == kFloatMissing || RH700 == kFloatMissing || RH925 == kFloatMissing ||
					RH850 == kFloatMissing || T700 == kFloatMissing ||
					Z850 == kFloatMissing || Z1000 == kFloatMissing || RR == kFloatMissing ||
					P == kFloatMissing || W925 == kFloatMissing || W850 == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				// Unit conversions

				//<! TODO: Kertoimet tietokannasta!
				
				T -= himan::constants::kKelvin;
				T850 -= himan::constants::kKelvin;
				T925 -= himan::constants::kKelvin;

				RH *= 100;
				RH700 *= 100;
				RH850 *= 100;
				RH925 *= 100;

				P *= 0.01;

				// 850-1000hPa paksuus [m]
				// source data is m^2/s^2 --> convert result to m by multiplying with 1/g
				const double dz850 = (Z850 - Z1000) * himan::constants::kIg;

				// (0=tihku, 1=vesi, 2=räntä, 3=lumi, 4=jäätävä tihku, 5=jäätävä sade)

				// jäätävää tihkua: "-10<T2m<=0, pakkasstratus (pinnassa/sen yläpuolella pakkasta & kosteaa), päällä ei (satavaa) keskipilveä, sade heikkoa"

				if ((T <= sfcMax) AND (T > sfcMin) AND (RH700 < 80) AND (RH > 90) AND (RR < fzdzLim))
				{
					// ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925),
					// tai kun Psfc ei (enää) löydy (eli ei mp-dataa, 6-10vrk)?
					// (riittävän paksu/jäätävä) stratus 925hPa:ssa, jossa nousuliikettä?

					if (P > (925 + stH) AND RH925 > 90 AND T925 < 0 AND T925 > stTlimit AND W925 > 0 AND W925 < wMax)
					{
						PreForm = kFreezingDrizzle;
					}

					// ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
					// (riittävän paksu/jäätävä) stratus 850hPa:ssa, jossa nousuliikettä?
					
					else if ((P <= 925+stH) AND (P > 850+stH) AND (RH850 > 90) AND (T850 < 0) AND (T850 > stTlimit) AND (W850 > 0) AND (W850 < wMax))
					{
						PreForm = kFreezingDrizzle;
					}
				}
				// jäätävää vesisadetta: "pinnassa pakkasta ja sulamiskerros pinnan lähellä"
				// (Heikoimmat intensiteetit pois, RR>0.1 tms?)

				if ((PreForm == kFloatMissing) AND (RR > 0.1) AND (T <= 0) AND ((T925 > 0) OR (T850 > 0) OR (T700 > 0)))
				{

					// ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925)
					// tai kun Psfc ei (enää) löydy (eli ei mp-dataa, 6-10vrk)?
					// (riittävän paksu) sulamiskerros 925hPa:ssa (tai pakkaskerros sen alla)?

					if ((P > 925+stH) AND ((T925 > 0) OR (T850 > 0)))
					{  
						PreForm = kFreezingRain;
					}

					// ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
					// (riittävän paksu) sulamiskerros 850hPa:ssa (tai pakkaskerros sen alla)?

					else if ((P <= 925+stH) AND (P > 850+stH) AND (T850 > 0))
					{  
						PreForm = kFreezingRain ;
					}

					// ollaanko ~1500-3000m merenpinnasta (850<pintapaine<700)?
					// (riittävän paksu) sulamiskerros 700hPa:ssa (tai pakkaskerros sen alla)?

					if ((PreForm == kFloatMissing) AND (P <= 850+stH) AND (P > 700+stH) AND (T700 > 0))
					{  
						PreForm = kFreezingRain;
					}
				}

				/* Use Koistinen formula to possibly change form to sleet or snow
				 * The same formula is in util::PrecipitationForm(), but here we
				 * use different limits to determine the form.
				 */

				const double probWater = util::WaterProbability(T, RH);

				// lumisadetta: "kylmä pinta/rajakerros"
				// (lisätty ehto "OR T<=0" vähentämään paksuustulkinnan virhettä vuoristossa)
				
				if ((PreForm == kFloatMissing) AND ((dz850 < snowThickness) OR (T <= 0)))
				{
					PreForm = kSnow;

					// Koistisen kaavan perusteella muutetaan lumi mahdollisesti rännäksi...

					if ((probWater >= sleetTOsnow) AND (probWater <= sleetTOwater) AND (T > 0))
					{
						PreForm = kSleet ;
					}
					// ...tai vedeksi
					if ((probWater > sleetTOwater) AND (T > 0))
					{
						PreForm = kRain ;
					}
				}

				// räntää: "dz850-1000 snowLim ja waterLim välissä"
				if ((PreForm == kFloatMissing) AND (dz850>=snowThickness) AND (dz850<=waterThickness) AND (T >= 0))
				{
					PreForm = kSleet;
					// Koistisen kaavan perusteella muutetaan räntä mahdollisesti vedeksi...

					if (probWater>sleetTOwater)
					{
						PreForm = kRain;
					}
					// ...tai lumeksi
					else if (probWater < sleetTOsnow)
					{
						PreForm = kSnow;
					}
				}

				// tihkua tai vesisadetta: "lämmin pinta/rajakerros"
				if ((PreForm == kFloatMissing) AND (dz850 > waterThickness) AND (T >= 0))
				{
					// tihkua: "ei (satavaa) keskipilveä, pinnan lähellä kosteaa (stratus), sade heikkoa"
					if ((RH700 < 80) AND (RH > 90) AND (RR <= dzLim))
					{
					  // ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925),
					  // tai kun Psfc ei (enää) löydy (eli ei mp-dataa, 6-10vrk)?
					  // stratus 925hPa:ssa?
					  if ((P > 925) AND (RH925 > 90))
					  {
						  PreForm = kDrizzle;
					  }
					  // ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
					  // stratus 850hPa:ssa?
					  else if ((P <= 925) AND (P > 850) AND (RH850 > 90))
					  {
						  PreForm=kDrizzle;
					  }
					}

					// muuten vesisadetta: "paksu satava pilvi"
					if (PreForm == kFloatMissing)
					{
						  PreForm = kRain;
						  // Koistisen kaavan perusteella muutetaan vesi mahdollisesti rännäksi...
						  if (probWater < waterTOsleet)
						  {
							  PreForm=kSleet;
						  }
						  // ...tai lumeksi (ei käytössä, jottei lämpimissä kerroksissa tule lunta; toisaalta vuoristossa parantaisi tulosta?)
						  // IF (probWater<sleetTOsnow)
						  // {  PreForm=3 }
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
