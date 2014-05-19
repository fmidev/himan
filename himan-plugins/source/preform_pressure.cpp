/**
 * @file preform_pressure.cpp
 *
 * @date Sep 5, 2013
 * @author partio
 *
 */

#define AND &&
#define OR ||
#define MISS kFloatMissing

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

// Olomuotopäättely pinta/peruspainepintadatalla tiivistettynä (tässä järjestyksessä):
//
// 0. Mallissa sadetta (RR>0; RR = rainfall + snowfall)
// 1. Jäätävää tihkua, jos -10<T2m<=0C, pakkasstratus (~-10...-0) jossa nousuliikettä, sade heikkoa, ei satavaa keskipilveä
// 2. Jäätävää vesisadetta, jos T2m<=0C ja pinnan yläpuolella on T>0C kerros, ja RR>0.1
// 3. Lunta, jos snowfall/RR>0.8, tai T<=0C
// 4. Räntää, jos 0.15<snowfall/RR<0.8
// 5. Vettä tai tihkua, jos snowfall/RR<0.15
// 5a. Tihkua, jos stratusta pienellä sadeintensiteetillä, eikä keskipilveä

// Mallin lumisateen osuuden raja-arvot (kokonaissateesta) lumi/räntä/vesiolomuodoille

const double waterLim = 0.15;
const double snowLim = 0.8;

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

	const param TParam("T-K");
	const param RHParam("RH-PRCNT");
	const param SNRParam("SNR-KGM2");

	// Default to one hour precipitation, will change this later on if necessary
	param RRParam("RR-1-MM");
	
	const params PParams({param("P-PA"), param("PGR-PA")});
	const params WParams({param ("VV-MMS"), param("VV-MS")});
	
	itsConfiguration->FirstSourceProducer();
	
	level groundLevel(kHeight, 2);

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);
	level P700(kPressure, 700);
	level P850(kPressure, 850);
	level P925(kPressure, 925);
	level P1000(kPressure, 1000);

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preformPressureThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

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

		shared_ptr<info> W925Info;
		shared_ptr<info> W850Info;

		shared_ptr<info> RRInfo;
		shared_ptr<info> PInfo;

		shared_ptr<info> SNRInfo;

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

			W925Info = aFetcher->Fetch(itsConfiguration,
						 myTargetInfo->Time(),
						 P925,
						 WParams);

			W850Info = aFetcher->Fetch(itsConfiguration,
					 myTargetInfo->Time(),
					 P850,
					 WParams);

			PInfo = aFetcher->Fetch(itsConfiguration,
					 myTargetInfo->Time(),
					 surface0mLevel,
					 PParams);

			RRInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 RRParam);

			SNRInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 surface0mLevel,
								 SNRParam);

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

		shared_ptr<NFmiGrid> W925Grid(W925Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> W850Grid(W850Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> RRGrid(RRInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> SNRGrid(SNRInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = CompareGrids({myTargetInfo->Grid(), TInfo->Grid(), T700Info->Grid(),
						T850Info->Grid(), T925Info->Grid(), RHInfo->Grid(),
						RH700Info->Grid(), RH850Info->Grid(), RH925Info->Grid(),
						RRInfo->Grid(), PInfo->Grid(), W850Info->Grid(),
						W925Info->Grid(), SNRInfo->Grid()});

		string deviceType;

		{

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			double WScale = 1;

			if (W850Info->Param().Name() == "VV-MS")
			{
				WScale = 1000;
			}

			assert(W850Info->Param().Name() == W925Info->Param().Name());

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

				double W850 = kFloatMissing;
				double W925 = kFloatMissing;

				double P = kFloatMissing;

				double RR = kFloatMissing;
				double SNR = kFloatMissing;

				int PreForm = static_cast<int> (kFloatMissing);

				InterpolateToPoint(targetGrid, RRGrid, equalGrids, RR);

				// No rain --> no rain type

				if (RR == 0)
				{
					missingCount++;

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

				InterpolateToPoint(targetGrid, W850Grid, equalGrids, W850);
				InterpolateToPoint(targetGrid, W925Grid, equalGrids, W925);

				InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				InterpolateToPoint(targetGrid, SNRGrid, equalGrids, SNR);

				if (IsMissingValue({T, T850, T925, RH, RH700, RH925, RH850, T700, RR, P, W925, W850, SNR}))
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				// Unit conversions

				//<! TODO: Kertoimet tietokannasta!
				
				T -= himan::constants::kKelvin;
				T700 -= himan::constants::kKelvin;
				T850 -= himan::constants::kKelvin;
				T925 -= himan::constants::kKelvin;

				// Dangerous unit conversion!
				// Parameter name is RH-PRCNT but data is still 0 .. 1
				
				RH *= 100;
				RH700 *= 100;
				RH850 *= 100;
				RH925 *= 100;

				P *= 0.01;

				W850 *= WScale;
				W925 *= WScale;

				/*
				cout	<< "T\t\t" << T << endl
						<< "T700\t\t" << T700 << endl
						<< "T850\t\t" << T850 << endl
						<< "T925\t\t" << T925 << endl
						<< "RH\t\t" << RH << endl
						<< "RH700\t\t" << RH700 << endl
						<< "RH850\t\t" << RH850 << endl
						<< "RH925\t\t" << RH925 << endl
						<< "P\t\t" << P << endl
						<< "W850\t\t" << W850 << endl
						<< "W925\t\t" << W925 << endl
						<< "RR\t\t" << RR << endl
						<< "stH\t\t" << stH << endl
						<< "sfcMin\t\t" << sfcMin << endl
						<< "sfcMax\t\t" << sfcMax << endl
						<< "fzdzLim\t\t" << fzdzLim << endl
						<< "wMax\t\t" << wMax << endl
						<< "stTlimit\t" << stTlimit << endl
						<< "SNR\t\t" << SNR << endl;
				exit(1);
				 */
				// (0=tihku, 1=vesi, 2=räntä, 3=lumi, 4=jäätävä tihku, 5=jäätävä sade)

				// jäätävää tihkua: "-10<T2m<=0, pakkasstratus (pinnassa/sen yläpuolella pakkasta & kosteaa), päällä ei (satavaa) keskipilveä, sade heikkoa"

				if ((T <= sfcMax) AND (T > sfcMin) AND (RH700 < 80) AND (RH > 90) AND (RR <= fzdzLim))
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

				if ((PreForm == MISS) AND (RR > 0.1) AND (T <= 0) AND ((T925 > 0) OR (T850 > 0) OR (T700 > 0)))
				{

					// ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925)
					// tai kun Psfc ei (enää) löydy (eli ei mp-dataa, 6-10vrk)?
					// (riittävän paksu) sulamiskerros 925hPa:ssa (tai pakkaskerros sen alla)?

					if ((P > (925+stH)) AND ((T925 > 0) OR (T850 > 0)))
					{  
						PreForm = kFreezingRain;
					}

					// ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
					// (riittävän paksu) sulamiskerros 850hPa:ssa (tai pakkaskerros sen alla)?

					else if ((P <= (925+stH)) AND (P > (850+stH)) AND (T850 > 0))
					{  
						PreForm = kFreezingRain ;
					}

					// ollaanko ~1500-3000m merenpinnasta (850<pintapaine<700)?
					// (riittävän paksu) sulamiskerros 700hPa:ssa (tai pakkaskerros sen alla)?

					if ((P <= 850+stH) AND (P > 700+stH) AND (T700 > 0))
					{  
						PreForm = kFreezingRain;
					}
				}

				// lumisadetta: snowfall >=80% kokonaissateesta

				const double SNR_RR = SNR/RR;

				if (PreForm == MISS AND (SNR_RR >= snowLim OR T <= 0))
				{
					PreForm = kSnow;
				}

				// räntää: snowfall 15...80% kokonaissateesta
				if ((PreForm == MISS) AND (SNR_RR > waterLim) AND (SNR_RR<snowLim))
				{
					PreForm = kSleet;
				}

				// tihkua tai vesisadetta: Rain>=85% kokonaissateesta
				if ((PreForm == MISS) AND (SNR_RR) <= waterLim)
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
							PreForm = kDrizzle;
						}
					}

					// muuten vesisadetta:
					if (PreForm == MISS)
					{
						PreForm = kRain;
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
