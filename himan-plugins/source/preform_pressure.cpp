/**
 * @file preform_pressure.cpp
 *
 * @date Sep 5, 2013
 * @author partio
 *
 */

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
const double TopTlimit = -12;

// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double FZdzLim = 0.2;

const double kKelvin = 273.15;

preform_pressure::preform_pressure()
{
	itsClearTextFormula = "<algorithm>";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preform_pressure"));

}

void preform_pressure::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to preform_pressure.
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

	Dimension(conf->LeadingDimension());
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

		boost::thread* t = new boost::thread(&preform_pressure::Run,
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

void preform_pressure::Run(shared_ptr<info> myTargetInfo,
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

void preform_pressure::Calculate(shared_ptr<info> myTargetInfo,
					 shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
{

shared_ptr<fetcher> aFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param RHParam("RH-PRCNT");
	param ZParam("Z-M2S2");
	param RRParam("RR-1-MM"); // one hour prec -- should we interpolate in forecast step is 3/6 hours ?

	conf->FirstSourceProducer();
	
	level groundLevel = LevelTransform(conf->SourceProducer(), TParam, level(kHeight, 2));

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);
	level P700(kPressure, 700);
	level P850(kPressure, 850);
	level P925(kPressure, 925);
	level P1000(kPressure, 1000);

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("preformPressureThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

//	bool useCudaInThisThread = conf->UseCuda() && threadIndex <= conf->CudaDeviceCount();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		// Source infos

		shared_ptr<info> TInfo;
		shared_ptr<info> T850Info;
		shared_ptr<info> T925Info;

		shared_ptr<info> RHInfo;
		shared_ptr<info> RH700Info;
		shared_ptr<info> RH925Info;

		shared_ptr<info> Z850Info;
		shared_ptr<info> Z1000Info;

		shared_ptr<info> RRInfo;

		try
		{
			TInfo = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 groundLevel,
								 TParam);

			assert(TInfo->Param().Unit() == kK);

			T850Info = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 P850,
								 TParam);

			assert(T850Info->Param().Unit() == kK);

			T925Info = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 P925,
								 TParam);

			assert(T925Info->Param().Unit() == kK);

			RHInfo = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 surface2mLevel,
								 RHParam);

			RH700Info = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 P700,
								 RHParam);

			RH925Info = aFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 P925,
								 RHParam);

			Z850Info = aFetcher->Fetch(conf,
					 myTargetInfo->Time(),
					 P850,
					 ZParam);

			Z1000Info = aFetcher->Fetch(conf,
					 myTargetInfo->Time(),
					 P1000,
					 ZParam);

			RRInfo = aFetcher->Fetch(conf,
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

		SetAB(myTargetInfo, TInfo);

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T925Grid(T925Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> RHGrid(RHInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH700Grid(RH700Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH925Grid(RH925Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> Z850Grid(Z850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> Z1000Grid(Z1000Info->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> RRGrid(RRInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = ((*myTargetInfo->Grid() == *TInfo->Grid()) &&
							(*myTargetInfo->Grid() == *T850Info->Grid()) &&
							(*myTargetInfo->Grid() == *T925Info->Grid()) &&
							(*myTargetInfo->Grid() == *RHInfo->Grid()) &&
							(*myTargetInfo->Grid() == *RH700Info->Grid()) &&
							(*myTargetInfo->Grid() == *RH925Info->Grid()) &&
							(*myTargetInfo->Grid() == *Z850Info->Grid()) &&
							(*myTargetInfo->Grid() == *Z1000Info->Grid()) &&
							(*myTargetInfo->Grid() == *RRInfo->Grid()));

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
				double T850 = kFloatMissing;
				double T925 = kFloatMissing;
				
				double RH = kFloatMissing;
				double RH700 = kFloatMissing;
				double RH925 = kFloatMissing;

				double Z850 = kFloatMissing;
				double Z1000 = kFloatMissing;

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
				InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
				InterpolateToPoint(targetGrid, T925Grid, equalGrids, T925);

				InterpolateToPoint(targetGrid, RHGrid, equalGrids, RH);
				InterpolateToPoint(targetGrid, RH700Grid, equalGrids, RH700);
				InterpolateToPoint(targetGrid, RH925Grid, equalGrids, RH925);

				InterpolateToPoint(targetGrid, Z850Grid, equalGrids, Z850);
				InterpolateToPoint(targetGrid, Z1000Grid, equalGrids, Z1000);

				if (T == kFloatMissing || T850 == kFloatMissing || T925 == kFloatMissing || 
					RH == kFloatMissing || RH700 == kFloatMissing || RH925 == kFloatMissing ||
					Z850 == kFloatMissing || Z1000 == kFloatMissing || RR == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				// Unit conversions

				//<! TODO: Kertoimet tietokannasta!
				
				T -= kKelvin;
				T850 -= kKelvin;
				T925 -= kKelvin;

				RH *= 100;
				RH700 *= 100;
				RH925 *= 100;

				// 850-1000hPa paksuus [m]
				// source data is m^2/s^2 --> convert result to m by multiplying with 1/g
				const double dz850 = (Z850 - Z1000) * 0.1019;
								
				// jäätävää tihkua: "-10<T2m<0, pakkasstratus (T>~-10), päällä ei (satavaa) keskipilveä, sade heikkoa"
				// hieman lievempi pinnan kosteusehto ehkä parempi FZDZ:lle, esim. RH>90?

				if ((T < sfcMax) && (T > sfcMin) && 
					(T925 < 0) && (T925 > TopTlimit) &&
					(RH700 < 80) && (RH925 > 90) &&
					(RH > 95) && (RR <= FZdzLim))
				{
					PreForm = kFreezingDrizzle;
				}

				// jäätävää vesisadetta: "pinnassa pakkasta ja sulamiskerros pinnan lähellä"
				
				else if ((T < 0) && ((T925 > 0) || (T850 > 0)))
				{
					PreForm = kFreezingRain;
				}

				// tihkua tai vesisadetta: "lämmin pinta/rajakerros"

				else if (dz850 > waterLim)
				{
					// tihkua: "ei (satavaa) keskipilveä ja pinnan lähellä kosteaa pienellä sadeintensiteetillä"

					if ((RH700 < 80) && (RH925 > 90) && (RH > 95) && (RR <= dzLim))
				    {
						PreForm = kDrizzle;
					}

					// muuten vesisadetta: "paksu satava pilvi"
					else
				    {
						PreForm = kRain;

						/* Use Koistinen formula to possibly change form to sleet or snow
						 * The same formula is in util::PrecipitationForm(), but here we
						 * use different limits to determine the form.
						 *
						 * TODO: Ask why 0.5 is used.
						 */

						const double probWater = 1 / (1 + exp(22 - 2.7 * T - 0.2 * RH));

						if (probWater < 0.5)  // alkup. PK:n arvo 0.5
						{
							PreForm = kSleet;
						}
						
						// ...tai lumeksi (ei käytössä, jottei lämpimillä pintakerroksilla tule lunta)

						//if (probWater<0.2)
						// {  PreForm=3; }
					}
				}

				// räntää: "dz850-1000 snowLim ja waterLim välissä"

				else if (dz850>=snowLim && dz850<=waterLim)
				{
					PreForm=kSleet;

					// Use Koistinen formula to possibly change form to rain or snow
					PreForm = util::PrecipitationForm(T, RH);
				}

				// lumisadetta: "kylmä pinta/rajakerros"
				else if (dz850<snowLim)
				{
					PreForm = kSnow;

					// Use Koistinen formula to possibly change form to sleet or rain
					PreForm = util::PrecipitationForm(T, RH);
	
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
