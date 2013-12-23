/**
 * @file hitool.cpp
 *
 * @date Sep 3, 2013
 * @author partio
 */

#include "logger_factory.h"
#include "plugin_factory.h"
#include "hitool.h"
#include <NFmiInterpolation.h>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

hitool::hitool()
	: itsTime()
{
    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("hitool"));
}

hitool::hitool(shared_ptr<plugin_configuration> conf)
	: itsTime()
{
    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("hitool"));
	itsConfiguration = conf;
}

shared_ptr<modifier> hitool::CreateModifier(HPModifierType modifierType) const
{

	shared_ptr<himan::modifier> mod;

	switch (modifierType)
	{
		case kMaximumModifier:
			mod = make_shared<modifier_max> ();
			break;

		case kMinimumModifier:
			mod = make_shared<modifier_min> ();
			break;

		case kMaximumMinimumModifier:
			mod = make_shared<modifier_maxmin> ();
			break;

		case kFindHeightModifier:
			mod = make_shared<modifier_findheight> ();
			break;

		case kFindValueModifier:
			mod = make_shared<modifier_findvalue> ();
			break;

		case kAverageModifier:
			mod = make_shared<modifier_mean> ();
			break;

		case kCountModifier:
			mod = make_shared<modifier_count> ();
			break;

		case kAccumulationModifier:
			mod = make_shared<modifier_sum> ();
			break;

		default:
			itsLogger->Fatal("Unknown modifier type: " + boost::lexical_cast<string> (modifierType));
			exit(1);
			break;

	}

	return mod;
}


vector<double> hitool::VerticalExtremeValue(shared_ptr<modifier> mod,
							HPLevelType wantedLevelType,
							const param& sourceParam,
							const param& targetParam,
							const vector<double>& firstLevelValueInfo,
							const vector<double>& lastLevelValueInfo,
							const vector<double>& findValueInfo) const
{
	shared_ptr<plugin::neons> n = dynamic_pointer_cast <plugin::neons> (plugin_factory::Instance()->Plugin("neons"));
	assert(wantedLevelType == kHybrid);

	// Move this to convenience functions?
	if (findValueInfo.size())
	{
		mod->FindValue(findValueInfo);
	}

	// Should we loop over all producers ?

	producer prod = itsConfiguration->SourceProducer(0);
	
	// first means first in sorted order, ie smallest number ie the highest level
	
	long firstHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "first hybrid level number"));
	long lastHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));


//	const double base = sourceParam.Base();
//	const double scale = sourceParam.Scale();


	for (int levelValue = lastHybridLevel; levelValue >= firstHybridLevel && !mod->CalculationFinished(); levelValue--)
	{

		level currentLevel(kHybrid, levelValue, "HYBRID");

		//itsLogger->Debug("Level " + boost::lexical_cast<string> (currentLevel.Value()) + ": height range crossed for " + boost::lexical_cast<string> (numFinishedLocations) +
		//	"/" + boost::lexical_cast<string> (finishedLocations.size()) + " grid points");

		valueheight data = GetData(currentLevel, sourceParam, itsTime);

		auto values = data.first;
		auto heights = data.second;

		assert(heights->Grid()->Size() == values->Grid()->Size());

		values->First();
		heights->First();

		mod->Process(values->Grid()->Data()->Values(), heights->Grid()->Data()->Values());

		/*while (mod->NextLocation() && values->NextLocation() && heights->NextLocation())
		{
			if (values->LocationIndex() != heights->LocationIndex())
			{
				cout << values->LocationIndex() << " " << heights->LocationIndex() << endl;
			}
			assert(values->LocationIndex() == heights->LocationIndex());

			if (finishedLocations[values->LocationIndex()])
			{
				continue;
			}
			
			double v = values->Value();
			double h = heights->Value();

			if (h == kFloatMissing)
			{
				continue;
			}

			double lowerHeight = h + 1;
			double upperHeight = h - 1;

			if (firstLevelValueInfo)
			{
				firstLevelValueInfo->NextLocation();
				lowerHeight = firstLevelValueInfo->Value();
			}

			if (lastLevelValueInfo)
			{
				lastLevelValueInfo->NextLocation();
				upperHeight = lastLevelValueInfo->Value();
			}

			if (lowerHeight == kFloatMissing || lowerHeight == kHPMissingValue)
			{
				continue;
			}
			else if (upperHeight == kFloatMissing || upperHeight == kHPMissingValue)
			{
				continue;
			}

			if (h < lowerHeight)
			{
				continue;
			}

			if (h > upperHeight)
			{
				finishedLocations[values->LocationIndex()] = true;
				continue;
			}

			v = v * scale + base;

			mod->Calculate(v, h);
		}
		 */
	}

	//ret->Grid()->Data()->Set(mod->Result());
	return mod->Result();
}

valueheight hitool::GetData(const level& wantedLevel, const param& wantedParam,	const forecast_time& wantedTime) const
{

	shared_ptr<info> values, heights;
	shared_ptr<plugin::fetcher> f = dynamic_pointer_cast <plugin::fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		if (!values)
		{
			values = f->Fetch(itsConfiguration,
								wantedTime,
								wantedLevel,
								wantedParam);
		}

		if (!heights)
		{
			heights = f->Fetch(itsConfiguration,
								wantedTime,
								wantedLevel,
								param("HL-M"));
		}
	}
	catch (HPExceptionType e)
	{
		switch (e)
		{
			case kFileDataNotFound:
			break;

			default:
				throw runtime_error("hitool: Unable to proceed");
			break;
		}
	}


	assert(values && heights);
	assert(values->Grid()->Size() == heights->Grid()->Size());

	// No Merge() here since that will mess up cache
	
	valueheight ret = valueheight(values,heights);
	return ret;
}

/* CONVENIENCE FUNCTIONS */

vector<double> hitool::VerticalHeight(const param& wantedParam,
						const vector<double>& firstLevelValueInfo,
						const vector<double>& lastLevelValueInfo,
						const vector<double>& findValueInfo,
						size_t findNth) const
{

//	wantedParam.Aggregation(kMinimum);

	auto modifier = CreateModifier(kFindHeightModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue(modifier, kHybrid, wantedParam, param("HL-M"), firstLevelValueInfo, lastLevelValueInfo, findValueInfo);
}

vector<double> hitool::VerticalMinimum(const param& wantedParam,
						const vector<double>& firstLevelValueInfo,
						const vector<double>& lastLevelValueInfo)const
{
	//parm.Aggregation(kMinimum);
	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo);
}

vector<double> hitool::VerticalMaximum(const param& wantedParam,
						const vector<double>& firstLevelValueInfo,
						const vector<double>& lastLevelValueInfo) const
{
	//parm.Aggregation(kMinimum);
	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo);
}

vector<double> hitool::VerticalAverage(const param& wantedParam,
						const vector<double>& firstLevelValueInfo,
						const vector<double>& lastLevelValueInfo) const
{
	//parm.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo);
}

vector<double> hitool::VerticalCount(const param& wantedParam,
						const vector<double>& firstLevelValueInfo,
						const vector<double>& lastLevelValueInfo,
						const vector<double>& findValueInfo) const
{
	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo);
}

vector<double> hitool::VerticalValue(const param& wantedParam, const vector<double>& heightInfo) const
{
	//parm.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, param("HL-M"), wantedParam);
}

void hitool::Time(const forecast_time& theTime)
{
	itsTime = theTime;
}

shared_ptr<info> hitool::Stratus()
{

	// Vaadittu min. stratuksen paksuus tihkussa [m]
	const double stLimit = 500.;

	// Kerroksen paksuus pinnasta [m], josta etsitään stratusta (min. BaseLimit+FZstLimit)
	const double layer = 2000.;

	// N-kynnysarvo vaaditulle min. stratuksen määrälle [%] (50=yli puoli taivasta):
	const double stCover = 0.5;

	// Kynnysarvo vaaditulle stratuksen yläpuolisen kuivan kerroksen paksuudelle [m] (jäätävässä) tihkussa:
	const double drydz = 1500.;
	
	const param baseParam("STRATUS-BASE-M");
	const param topParam("STRATUS-TOP-M");
	const param topTempParam("STRATUS-TOP-T-K");
	const param meanTempParam("STRATUS-MEAN-T-K");
	const param meanCloudinessParam("STRATUS-MEAN-N-PRCNT");
	const param upperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	const param verticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MS");

	vector<param> params = { baseParam, topParam, topTempParam, meanTempParam, meanCloudinessParam, upperLayerRHParam, verticalVelocityParam };
	vector<forecast_time> times = { itsTime };
	vector<level> levels = { level(kHeight, 0, "HEIGHT") };

	auto ret = make_shared<info> (*itsConfiguration->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create();

	vector<double> constData1(ret->Data()->Size(), 0);
	
	auto constData2 = constData1;
	fill(constData2.begin(), constData2.end(), stLimit);

	// N-kynnysarvot stratuksen ala- ja ylärajalle [%] (tarkkaa stCover arvoa ei aina löydy)

	param wantedParam("N-0TO1");

	/**
	 * Etsitään parametrin N minimiarvo korkeusvälillä 0 .. stLimit (=500)
     */

	itsLogger->Info("Searching for stratus lower limit");

	auto baseThreshold = VerticalMinimum(wantedParam, constData1, constData2);
	
	for (size_t i = 0; i < baseThreshold.size(); i++)
	{
		if (baseThreshold[i] == kFloatMissing || baseThreshold[i] < stCover)
		{
			baseThreshold[i] = stCover;
		}
	}

	ret->Param(baseParam);
	ret->Data()->Set(baseThreshold);

	fill(constData1.begin(), constData1.end(), stLimit);
	fill(constData2.begin(), constData2.end(), layer);

	/**
	 * Etsitään parametrin N minimiarvo korkeusvälillä stLimit (=500) .. layer (=2000)
     */

	itsLogger->Info("Searching for stratus upper limit");

	auto topThreshold = VerticalMinimum(wantedParam, constData1, constData2);

	for (size_t i = 0; i < topThreshold.size(); i++)
	{
		if (baseThreshold[i] == kFloatMissing || topThreshold[i] < stCover)
		{
			topThreshold[i] = stCover;
		}
	}

	// Stratus Base/top [m]
	// _findh: 0 = viimeinen löytyvä arvo pinnasta ylöspäin, 1 = ensimmäinen löytyvä arvo
	// (Huom. vertz-funktio hakee tarkkaa arvoa, jota ei aina löydy esim. heti pinnasta lähtevälle
	//  stratukselle; joskus siis tuloksena on virheellisesti Base=top)

	fill(constData1.begin(), constData1.end(), 0);

	/**
	 * Etsitään parametrin N ensimmäisen baseThreshold-arvon korkeus väliltä 0 .. layer (=2000)
	 */

	itsLogger->Info("Searching for stratus base accurate value");

	auto stratusBase = VerticalHeight(wantedParam, constData1, constData2, baseThreshold);

	//VAR Base = VERTZ_FINDH(N_EC,0,Layer,BaseThreshold,1)

	if (!ret->Param(baseParam))
	{
		throw runtime_error("Impossible error");
	}

	ret->Data()->Set(stratusBase);

	size_t missing = 0;

	for (size_t i = 0; i < stratusBase.size(); i++)
	{
		if (stratusBase[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus base number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusBase.size()));
	
	fill(constData2.begin(), constData2.end(), layer);

	//VAR Top = VERTZ_FINDH(N_EC,0,Layer,TopThreshold,0)

	/**
	 * Etsitään parametrin N viimeisen topThreshold-arvon korkeus väliltä 0 .. layer (=2000)
	 */

	itsLogger->Info("Searching for stratus top accurate value");
	auto stratusTop = VerticalHeight(wantedParam, constData1, constData2, topThreshold, 0);

	ret->Param(topParam);
	ret->Data()->Set(stratusTop);

	missing = 0;
	
	for (size_t i = 0; i < stratusTop.size(); i++)
	{
		if (stratusTop[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus top number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (stratusTop.size()));

	// Keskimääräinen RH stratuksen yläpuolisessa kerroksessa (jäätävä tihku)

	itsLogger->Info("Searching for humidity in layers above stratus top");

	wantedParam = param("RH-PRCNT");

	assert(constData1.size() == constData2.size() && constData1.size() == stratusTop.size());
	
	for (size_t i = 0; 	i < constData1.size(); i++)
	{

		if (stratusTop[i] == kFloatMissing)
		{
			constData1[i] = kFloatMissing;
			constData2[i] = kFloatMissing;
		}
		else
		{
			constData1[i] = stratusTop[i] + 100;
			constData2[i] = stratusTop[i] + drydz;
		}
	}

	//VERTZ_AVG(RH_EC,Top+100,Top+DRYdz)
	//auto upperLayerRH = VerticalExtremeValue(opts);
	auto upperLayerRH = VerticalAverage(wantedParam, constData1, constData2);

	ret->Param(upperLayerRHParam);
	ret->Data()->Set(upperLayerRH);

	missing = 0;

	for (size_t i = 0; i < upperLayerRH.size(); i++)
	{
		if (upperLayerRH[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Upper layer RH number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (upperLayerRH.size()));

	//VERTZ_AVG(N_EC,Base,Top)

	itsLogger->Info("Searching for stratus mean cloudiness");

	wantedParam = param("N-0TO1");

	auto stratusMeanN = VerticalAverage(wantedParam, stratusBase, stratusTop);
	//auto stratusMeanN = VerticalExtremeValue(opts);

	ret->Param(meanCloudinessParam);
	ret->Data()->Set(stratusMeanN);
	
	itsLogger->Info("Searching for stratus top temperature");

	// Stratuksen Topin lämpötila (jäätävä tihku)
	//VAR TTop = VERTZ_GET(T_EC,Top)

	wantedParam = param("T-K");
	
	//auto stratusTopTemp = VerticalExtremeValue(opts);
	auto stratusTopTemp = VerticalValue(wantedParam, stratusTop);

	ret->Param(topTempParam);
	ret->Data()->Set(stratusTopTemp);

	itsLogger->Info("Searching for stratus mean temperature");

	// St:n keskimääräinen lämpötila (poissulkemaan kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
	//VAR stTavg = VERTZ_AVG(T_EC,Base+50,Top-50)

	for (size_t i = 0; i < constData1.size(); i++)
	{
		if (stratusBase[i] == kFloatMissing)
		{
			constData1[i] = kFloatMissing;
			constData2[i] = kFloatMissing;
		}
		else
		{
			constData1[i] = stratusBase[i] + 50;
			constData2[i] = stratusBase[i] - 50;
		}
	}
	
	auto stratusMeanTemp = VerticalAverage(wantedParam, constData1, constData2);

	ret->Param(meanTempParam);
	ret->Data()->Set(stratusMeanTemp);
	
	// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
	//VAR wAvg = VERTZ_AVG(W_EC,Base,Top)

	itsLogger->Info("Searching for mean vertical velocity in stratus");

	wantedParam = param("VV-MS");

	auto stratusVerticalVelocity = VerticalAverage(wantedParam, stratusBase, stratusTop);

	ret->Param(verticalVelocityParam);
	ret->Data()->Set(stratusVerticalVelocity);

	return ret;
}

shared_ptr<info> hitool::FreezingArea()
{

	const param minusAreaParam("MINUS-AREA-T-C");
	const param plusArea1Param("PLUS-AREA-1-T-C");
	const param plusArea2Param("PLUS-AREA-2-T-C");

	vector<param> params = { minusAreaParam, plusArea1Param, plusArea2Param };
	vector<forecast_time> times = { itsTime };
	vector<level> levels = { level(kHeight, 0, "HEIGHT") };

	auto ret = make_shared<info> (*itsConfiguration->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create();

	vector<double> constData1;

	constData1.resize(ret->Data()->Size(), 0);

	auto constData2 = constData1;
	fill(constData2.begin(), constData2.end(), 5000);

	auto constData3 = constData1;
	fill(constData2.begin(), constData2.end(), 273.15); // 0C

	// 0-kohtien lkm pinnasta (yläraja 5km, jotta ylinkin nollakohta varmasti löytyy)
	param wantedParam ("T-K");
	//wantedParam.Base(-273.15);
	
	auto numZeroLevels = VerticalCount(wantedParam, constData1, constData2, constData3);

	//nZeroLevel = VERTZ_FINDC(T_EC,0,5000,0)

	/* Check which values we have. Will slow down processing a bit but
	 * will make subsequent code much easier to understand.
	 */

	bool haveOne = false;
	bool haveTwo = false;
	bool haveThree = false;

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		size_t val = numZeroLevels[i];

		if (val == 1)
		{
			haveOne = true;
		}
		else if (val == 2)
		{
			haveTwo = true;
		}
		else if (val == 3)
		{
			haveThree = true;
		}

		if (haveOne && haveTwo && haveThree)
		{
			break;
		}
	}

	// Get necessary source data based on loop data above

	vector<double> zeroLevel1(numZeroLevels.size(), kFloatMissing);

	auto zeroLevel2 = zeroLevel1;
	auto zeroLevel3 = zeroLevel1;
	auto Tavg1 = zeroLevel1;
	auto Tavg2_two = zeroLevel1;
	auto Tavg2_three = zeroLevel1;
	auto Tavg3 = zeroLevel1;
	auto plusArea1 = zeroLevel1;
	auto plusArea2 = zeroLevel1;
	auto minusArea = zeroLevel1;

	if (haveOne)
	{
		itsLogger->Info("Searching for first zero level height and value");
		
		// Find height of first zero level
		  // ZeroLev1 = VERTZ_FINDH(T_EC,0,5000,0,1)

		zeroLevel1 = VerticalHeight(wantedParam, constData1, constData2, constData3, 1);

		Tavg1 = VerticalAverage(wantedParam, constData1, zeroLevel1);

	}

	if (haveTwo)
	{
		itsLogger->Info("Searching for second zero level height and value");

		// Find height of second zero level

		zeroLevel2 = VerticalHeight(wantedParam, constData1, constData2, constData3, 2);

		assert(zeroLevel1.size());

		Tavg2_two = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	if (haveThree)
	{
		itsLogger->Info("Searching for third zero level height and value");

		// Find height of third zero level

		zeroLevel3 = VerticalHeight(wantedParam, constData1, constData2, constData3, 3);

		assert(zeroLevel1.size());
		assert(zeroLevel2.size());

		Tavg2_three = VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);
		Tavg3 = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		size_t numZeroLevel = numZeroLevels[i];
		
		if (numZeroLevel == 0)
		{
			continue;
		}

		else if (numZeroLevel == 1)
		{
			double zl = zeroLevel1[i], ta = Tavg1[i];
			double pa = kFloatMissing;

			if (zl != kFloatMissing && ta != kFloatMissing)
			{
				pa = zl * ta;
			}

			plusArea1[i] = pa;

		}
		else if (numZeroLevel == 2)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg1[i];
			double zl2 = zeroLevel2[i], ta2 = Tavg2_two[i];
			double pa = kFloatMissing, ma = kFloatMissing;

			if (zl2 != kFloatMissing && zl1 != kFloatMissing && ta2 != kFloatMissing)
			{
				pa = (zl2 - zl1) * ta2;
			}

			plusArea1[i] = pa;

			if (zl1 != kFloatMissing && ta1 != kFloatMissing)
			{
				ma = zl1 * ta1;
			}

			minusArea[i] = ma;
		}
		else if (numZeroLevel == 3)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg1[i];
			double zl2 = zeroLevel2[i], ta2 = Tavg2_three[i];
			double zl3 = zeroLevel3[i], ta3 = Tavg3[i];

			double pa1 = kFloatMissing, pa2 = kFloatMissing, ma = kFloatMissing;

			if (zl1 != kFloatMissing && zl2 != kFloatMissing && ta2 != kFloatMissing)
			{
				pa2 = (zl3 - zl2) * ta2;
			}

			plusArea2[i] = pa2;

			if (zl1 != kFloatMissing && pa2 != kFloatMissing && ta1 != kFloatMissing)
			{
				pa1 = zl1 * ta1 + pa2;
			}

			plusArea1[i] = pa1;

			if (zl2 != kFloatMissing && zl1 != kFloatMissing && ta3 != kFloatMissing)
			{
				ma = (zl2 - zl1) * ta3;
			}
			
			minusArea[i] = ma;
		}
		
	}

	ret->Param(minusAreaParam);
	ret->Data()->Set(minusArea);

	ret->Param(plusArea1Param);
	ret->Data()->Set(plusArea1);

	ret->Param(plusArea2Param);
	ret->Data()->Set(plusArea2);

	return ret;

}

void hitool::Configuration(shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = conf;
}
