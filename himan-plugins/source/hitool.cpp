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

level hitool::LevelForHeight(const producer& prod, double height) const
{

	switch (prod.Id())
	{
		case 1:
		case 230:

			break;
	}
	
	return level();
}

vector<double> hitool::VerticalExtremeValue(shared_ptr<modifier> mod,
							HPLevelType wantedLevelType,
							const param& wantedParam,
							const vector<double>& lowerHeight,
							const vector<double>& upperHeight,
							const vector<double>& findValue) const
{
	shared_ptr<plugin::neons> n = dynamic_pointer_cast <plugin::neons> (plugin_factory::Instance()->Plugin("neons"));
	assert(wantedLevelType == kHybrid);

	// Move this to convenience functions?

	if (findValue.size())
	{
		mod->FindValue(findValue);
	}

	if (lowerHeight.size())
	{
		mod->LowerHeight(lowerHeight);
	}

	if (upperHeight.size())
	{
		mod->UpperHeight(upperHeight);
	}

	// Should we loop over all producers ?

	producer prod = itsConfiguration->SourceProducer(0);
	
	// first means first in sorted order, ie smallest number ie the highest level
	
	long firstHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "first hybrid level number"));
	long lastHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));

	for (long levelValue = lastHybridLevel; levelValue >= firstHybridLevel && !mod->CalculationFinished(); levelValue--)
	{

		level currentLevel(kHybrid, levelValue, "HYBRID");

		valueheight data = GetData(currentLevel, wantedParam, itsTime);

		auto values = data.first;
		auto heights = data.second;

		assert(heights->Grid()->Size() == values->Grid()->Size());

		values->First();
		heights->First();

		mod->Process(values->Grid()->Data()->Values(), heights->Grid()->Data()->Values());

		size_t heightsCrossed = mod->HeightsCrossed();

		itsLogger->Debug("Level " + boost::lexical_cast<string> (currentLevel.Value()) + ": height range crossed for " + boost::lexical_cast<string> (heightsCrossed) +
			"/" + boost::lexical_cast<string> (values->Data()->Size()) + " grid points");

	}

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
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue,
						const vector<double>& findValue,
						size_t findNth) const
{
	auto modifier = CreateModifier(kFindHeightModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue(modifier, kHybrid, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

vector<double> hitool::VerticalMinimum(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue)const
{
	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid,  wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalAverage(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalSum(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kAccumulationModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalCount(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue,
						const vector<double>& findValue) const
{
	return VerticalExtremeValue(CreateModifier(kCountModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

vector<double> hitool::VerticalValue(const param& wantedParam, const vector<double>& heightInfo) const
{
	//parm.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, wantedParam, vector<double> (), vector<double> (), heightInfo);
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
	
	param baseParam("STRATUS-BASE-M");
	baseParam.Unit(kM);

	param topParam("STRATUS-TOP-M");
	topParam.Unit(kM);
	
	param topTempParam("STRATUS-TOP-T-K");
	topTempParam.Unit(kK);

	param meanTempParam("STRATUS-MEAN-T-K");
	meanTempParam.Unit(kK);

	param meanCloudinessParam("STRATUS-MEAN-N-PRCNT");
	meanCloudinessParam.Unit(kPrcnt);

	param upperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	upperLayerRHParam.Unit(kPrcnt);

	param verticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MMS");
	verticalVelocityParam.Unit(kMs);

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
		assert(baseThreshold[i] != kFloatMissing);
		if (baseThreshold[i] < stCover)
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
		assert(topThreshold[i] != kFloatMissing);
		if (topThreshold[i] < stCover)
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
	
	// fill(constData2.begin(), constData2.end(), layer);

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

	missing = 0;

	for (size_t i = 0; i < stratusMeanN.size(); i++)
	{
		if (stratusMeanN[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus mean cloudiness number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (stratusMeanN.size()));

	ret->Param(meanCloudinessParam);
	ret->Data()->Set(stratusMeanN);
	
	itsLogger->Info("Searching for stratus top temperature");

	// Stratuksen Topin lämpötila (jäätävä tihku)
	//VAR TTop = VERTZ_GET(T_EC,Top)

	wantedParam = param("T-K");
	
	auto stratusTopTemp = VerticalValue(wantedParam, stratusTop);

	missing = 0;

	for (size_t i = 0; i < stratusTopTemp.size(); i++)
	{
		if (stratusTopTemp[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus top temperature number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusTopTemp.size()));

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
			constData2[i] = stratusTop[i] - 50;
		}
	}

	auto stratusMeanTemp = VerticalAverage(wantedParam, constData1, constData2);

	missing = 0;

	for (size_t i = 0; i < stratusMeanTemp.size(); i++)
	{
		if (stratusMeanTemp[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus mean temperature number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusMeanTemp.size()));
	
	ret->Param(meanTempParam);
	ret->Data()->Set(stratusMeanTemp);

	// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
	//VAR wAvg = VERTZ_AVG(W_EC,Base,Top)

	itsLogger->Info("Searching for mean vertical velocity in stratus");

	wantedParam = param("VV-MMS");

	auto stratusVerticalVelocity = VerticalAverage(wantedParam, stratusBase, stratusTop);

	missing = 0;

	for (size_t i = 0; i < stratusVerticalVelocity.size(); i++)
	{
		if (stratusVerticalVelocity[i] == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus vertical velocity number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusVerticalVelocity.size()));

	ret->Param(verticalVelocityParam);
	ret->Data()->Set(stratusVerticalVelocity);

	return ret;
}

shared_ptr<info> hitool::FreezingArea()
{

	param minusAreaParam("MINUS-AREA-T-K");
	minusAreaParam.Unit(kK);

	param plusArea1Param("PLUS-AREA-1-T-K");
	plusArea1Param.Unit(kK);

	param plusArea2Param("PLUS-AREA-2-T-K");
	plusArea2Param.Unit(kK);

	vector<param> params = { minusAreaParam, plusArea1Param, plusArea2Param };
	vector<forecast_time> times = { itsTime };
	vector<level> levels = { level(kHeight, 0, "HEIGHT") };

	auto ret = make_shared<info> (*itsConfiguration->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create();

	vector<double> constData1(ret->Data()->Size(), 0);

	auto constData2 = constData1;
	fill(constData2.begin(), constData2.end(), 5000);

	auto constData3 = constData1;
	fill(constData3.begin(), constData3.end(), himan::constants::kKelvin); // 0C

	// 0-kohtien lkm pinnasta (yläraja 5km, jotta ylinkin nollakohta varmasti löytyy)
	param wantedParam ("T-K");

	itsLogger->Info("Counting number of zero levels");

	auto numZeroLevels = VerticalCount(wantedParam, constData1, constData2, constData3);

#ifdef DEBUG
	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		assert(numZeroLevels[i] != kFloatMissing);
	}
#endif
	
	//nZeroLevel = VERTZ_FINDC(T_EC,0,5000,0)

	/* Check which values we have. Will slow down processing a bit but
	 * will make subsequent code much easier to understand.
	 */

	bool haveOne = false;
	bool haveTwo = false;
	bool haveThree = false;

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		size_t val = static_cast<size_t> (numZeroLevels[i]);

		if (val == 1)
		{
			haveOne = true;
		}
		else if (val == 2)
		{
			haveTwo = true;
		}
		else if (val >= 3)
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

		itsLogger->Info("Searching for average temperature between ground level and first zero level");

		Tavg1 = VerticalAverage(wantedParam, constData1, zeroLevel1);
		
	}

	if (haveTwo)
	{
		itsLogger->Info("Searching for second zero level height and value");

		assert(haveOne);
		
		// Find height of second zero level

		zeroLevel2 = VerticalHeight(wantedParam, constData1, constData2, constData3, 2);

		itsLogger->Info("Searching for average temperature between first and second zero level");

		Tavg2_two = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	if (haveThree)
	{
		itsLogger->Info("Searching for third zero level height and value");

		assert(haveOne && haveTwo);
		
		// Find height of third zero level

		zeroLevel3 = VerticalHeight(wantedParam, constData1, constData2, constData3, 3);

		itsLogger->Info("Searching for average temperature between second and third zero level");

		Tavg2_three = VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);

		itsLogger->Info("Searching for average temperature between first and third zero level");

		Tavg3 = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		short numZeroLevel = static_cast<short> (numZeroLevels[i]);

		// Pintakerros plussalla (1 nollaraja)

		if (numZeroLevel == 1)
		{
			double zl = zeroLevel1[i], ta = Tavg1[i];
			double pa = kFloatMissing;

			if (zl != kFloatMissing && ta != kFloatMissing)
			{
				pa = zl * (ta - himan::constants::kKelvin);
			}

			plusArea1[i] = pa;

		}
		
		// Pintakerros pakkasella, ylempänä T>0 kerros (2 nollarajaa)

		else if (numZeroLevel == 2)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg1[i];
			double zl2 = zeroLevel2[i], ta2 = Tavg2_two[i];
			double pa = kFloatMissing, ma = kFloatMissing;

			if (zl2 != kFloatMissing && zl1 != kFloatMissing && ta2 != kFloatMissing)
			{
				pa = (zl2 - zl1) * (ta2 - himan::constants::kKelvin);
			}

			plusArea1[i] = pa;

			if (zl1 != kFloatMissing && ta1 != kFloatMissing)
			{
				ma = zl1 * (ta1 - himan::constants::kKelvin);
			}

			minusArea[i] = ma;
		}

		// Pintakerroksen lisäksi ylempänä toinen T>0 kerros, joiden välissä pakkaskerros (3 nollarajaa)

		else if (numZeroLevel == 3)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg1[i];
			double zl2 = zeroLevel2[i], ta2 = Tavg2_three[i];
			double zl3 = zeroLevel3[i], ta3 = Tavg3[i];

			double pa1 = kFloatMissing, pa2 = kFloatMissing, ma = kFloatMissing;

			if (zl1 != kFloatMissing && zl2 != kFloatMissing && ta2 != kFloatMissing)
			{
				pa2 = (zl3 - zl2) * (ta2 - himan::constants::kKelvin); // "aloft"
			}

			plusArea2[i] = pa2;

			if (zl1 != kFloatMissing && pa2 != kFloatMissing && ta1 != kFloatMissing)
			{
				pa1 = zl1 * (ta1 - himan::constants::kKelvin) + pa2;
			}

			plusArea1[i] = pa1;

			if (zl2 != kFloatMissing && zl1 != kFloatMissing && ta3 != kFloatMissing)
			{
				ma = (zl2 - zl1) * (ta3 - himan::constants::kKelvin);
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
