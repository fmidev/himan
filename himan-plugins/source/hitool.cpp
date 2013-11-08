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
	: itsScale(1)
	, itsBase(0)
	, itsFirstLevelValueBase(0)
	, itsLastLevelValueBase(0)
{
    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("hitool"));
}

shared_ptr<modifier> hitool::CreateModifier(hitool_search_options& opts, vector<param>& params)
{
	param height("HL-M");

	// CREATE MODIFIER

	shared_ptr<himan::modifier> mod;
	param resultParam(opts.wantedParam);

	switch (opts.wantedModifier)
	{
		case kMaximumModifier:
			mod = make_shared<modifier_max> ();
			resultParam.Aggregation().Type(kMaximum);
			height.Aggregation().Type(kExternalMaximum);
			params.push_back(resultParam);
			params.push_back(height);
			break;

		case kMinimumModifier:
			mod = make_shared<modifier_min> ();
			resultParam.Aggregation().Type(kMinimum);
			height.Aggregation().Type(kExternalMinimum);
			params.push_back(resultParam);
			params.push_back(height);
			break;

		case kMaximumMinimumModifier:
			mod = make_shared<modifier_maxmin> ();
			resultParam.Aggregation().Type(kMaximum);
			params.push_back(resultParam);
			resultParam.Aggregation().Type(kMinimum);
			params.push_back(resultParam);

			height.Aggregation().Type(kExternalMaximum);
			params.push_back(height);
			height.Aggregation().Type(kExternalMinimum);
			params.push_back(height);
			break;

		case kFindHeightModifier:
			mod = make_shared<modifier_findheight> ();
			params.push_back(resultParam);
			params.push_back(height);

			if (opts.findValueInfo)
			{
				mod->FindValue(opts.findValueInfo);
			}
			else
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}
			break;

		case kFindValueModifier:
			mod = make_shared<modifier_findvalue> ();
			params.push_back(resultParam);
			params.push_back(height);

			if (opts.findValueInfo)
			{
				mod->FindValue(opts.findValueInfo);
			}
			else
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}
			break;

		case kAverageModifier:
			mod = make_shared<modifier_mean> ();
			resultParam.Aggregation().Type(kAverage);
			params.push_back(resultParam);
			break;
			
		case kAccumulationModifier:
		default:
			itsLogger->Fatal("Unknown modifier type: " + boost::lexical_cast<string> (opts.wantedModifier));
			exit(1);
			break;

	}
	
	return mod;
}

shared_ptr<info> hitool::VerticalExtremeValue(hitool_search_options& opts)
{
	shared_ptr<plugin::neons> n = dynamic_pointer_cast <plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

	if (!opts.returnHeight)
	{
		//mod->ReturnHeight(false);
		throw runtime_error(ClassName() + ": Must return height for now");
	}

	assert(opts.wantedLevelType == kHybrid);

	// Should we loop over all producers ?

	opts.conf->FirstSourceProducer();
	
	// first means first in sorted order, ie smallest number ie the highest level
	
	long firstHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(opts.conf->SourceProducer().Id(), "first hybrid level number"));
	long lastHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(opts.conf->SourceProducer().Id(), "last hybrid level number"));

	vector<param> params;

	auto mod = CreateModifier(opts, params);
	
	/*
	 * Modifier needs an info instance where it will store its data.
	 * This info will be returned from this function.
	 *
	 * We'll use the plugin_configuration info as a base and just change
	 * the parameters. Calling Create() will re-grid the info using
	 * configuration file arguments.
	 */

	auto ret = make_shared<info>(*opts.conf->Info());

	ret->Params(params);
	
	// Create data backend

	ret->Create();

	mod->Init(ret);

	ret->First();

#ifndef NDEBUG
	size_t retGridSize = ret->Grid()->Size();
#endif
	
	for (int levelValue = lastHybridLevel; levelValue >= firstHybridLevel; levelValue--)
	{
		level currentLevel(kHybrid, levelValue, "HYBRID");

		//itsLogger->Debug("Current level: " + boost::lexical_cast<string> (currentLevel.Value()));

		valueheight data = GetData(opts.conf, currentLevel, opts.wantedParam, opts.wantedTime);

		auto values = data.first;
		auto heights = data.second;

		assert(heights->Grid()->Size() == retGridSize);
		assert(values->Grid()->Size() == retGridSize);

		assert(opts.firstLevelValueInfo);
		assert(opts.lastLevelValueInfo);

		assert(opts.firstLevelValueInfo->Grid()->Size() == retGridSize);
		opts.firstLevelValueInfo->First();

		assert(opts.lastLevelValueInfo->Grid()->Size() == retGridSize);
		opts.lastLevelValueInfo->First();

		mod->ResetLocation();

		values->First(); values->ResetLocation();
		heights->First(); heights->ResetLocation();

		while (mod->NextLocation() && values->NextLocation() && heights->NextLocation() && opts.firstLevelValueInfo->NextLocation() && opts.lastLevelValueInfo->NextLocation())
		{

			double lowerThreshold = opts.firstLevelValueInfo->Value();
			double upperThreshold = opts.lastLevelValueInfo->Value();

			if (lowerThreshold != kHPMissingValue && lowerThreshold != kFloatMissing)
			{
				continue;
			}
			else if (upperThreshold != kHPMissingValue && upperThreshold != kFloatMissing)
			{
				continue;
			}

			double v = values->Value();
			double h = heights->Value();
			
			// Check that we are in given height range

			if (h < lowerThreshold + itsFirstLevelValueBase)
			{
				continue;
			}

			if (h > upperThreshold + itsLastLevelValueBase)
			{
				break;
			}

			v = v * itsScale + itsBase;

			mod->Calculate(v, h);
		}
	}

	return mod->Results();
}

valueheight hitool::GetData(const shared_ptr<const plugin_configuration> conf,
																const level& wantedLevel,
																const param& wantedParam,
																const forecast_time& wantedTime)
{

	conf->ResetSourceProducer();
	shared_ptr<info> values, heights;
	shared_ptr<plugin::fetcher> f = dynamic_pointer_cast <plugin::fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		if (!values)
		{
			values = f->Fetch(conf,
								wantedTime,
								wantedLevel,
								wantedParam);
		}

		if (!heights)
		{
			heights = f->Fetch(conf,
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

double hitool::Base() const
{
	return itsBase;
}

void hitool::Base(double theBase)
{
	itsBase = theBase;
}

double hitool::Scale() const
{
	return itsScale;
}

void hitool::Scale(double theScale)
{
	itsScale = theScale;
}

std::shared_ptr<info> hitool::Stratus(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime)
{

	const param stratusBaseParam("STRATUS-BASE-M");
	const param stratusTopParam("STRATUS-TOP-M");
	const param stratusTopTempParam("STRATUS-TOP-T-K");
	const param stratusMeanTempParam("STRATUS-MEAN-T-K");
	const param stratusUpperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	const param stratusVerticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MS");

	vector<param> params = { param("FAKE-PARAM") };
	vector<forecast_time> times = { wantedTime };
	vector<level> levels = { level(kUnknownLevel, 0, "FAKE-LEVEL") };

	auto constData1 = make_shared<info> (*conf->Info());
	constData1->Params(params);
	constData1->Times(times);
	constData1->Levels(levels);

	constData1->Create();
	constData1->First();
	constData1->Grid()->Data()->Fill(0);

	// Create data backend

	const double stLimit = 500.;
	//const double fzStLimit = 800.;
	const double layer = 2000.;
	const double stCover = 50.;
	const double drydz = 1500.;
	
	auto constData2 = make_shared<info> (*constData1);
	constData2->ReGrid();
	constData2->First();
	constData2->Grid()->Data()->Fill(stLimit);

	// N-kynnysarvot stratuksen ala- ja ylärajalle [%] (tarkkaa stCover arvoa ei aina löydy)

	hitool_search_options opts (param("N-0TO1"),
									wantedTime,
									kHybrid,
									constData1,
									constData2,
									kMinimumModifier,
									conf,
									true,
									1
	);

	itsScale = 100;

	itsLogger->Info("Searching for stratus lower limit");

	// Etsitaan parametrin N minimiarvo välillä 0 .. stLimit

	auto baseThreshold = VerticalExtremeValue(opts);

	baseThreshold->First();

	for (baseThreshold->ResetLocation(); baseThreshold->NextLocation();)
	{
		if (baseThreshold->Value() == kHPMissingValue || baseThreshold->Value() < stCover)
		{
			baseThreshold->Value(stCover);
		}
	}

	constData1->Grid()->Data()->Fill(stLimit);
	constData2->Grid()->Data()->Fill(layer);

	itsLogger->Info("Searching for stratus upper limit");

	auto topThreshold = VerticalExtremeValue(opts);

	topThreshold->First();

	for (topThreshold->ResetLocation(); topThreshold->NextLocation();)
	{
		//assert(topThreshold->Value() != kHPMissingValue);

		if (topThreshold->Value() < stCover)
		{
			topThreshold->Value(stCover);
		}
	}

	// Stratus Base/top [m]
	// _findh: 0 = viimeinen löytyvä arvo pinnasta ylöspäin, 1 = ensimmäinen löytyvä arvo
	// (Huom. vertz-funktio hakee tarkkaa arvoa, jota ei aina löydy esim. heti pinnasta lähtevälle
	//  stratukselle; joskus siis tuloksena on virheellisesti Base=top)

	constData1->Grid()->Data()->Fill(0);
	constData2->Grid()->Data()->Fill(layer);

	//opts.firstLevelValueInfo = constData1;
	//opts.lastLevelValueInfo = constData2;
	opts.wantedModifier = kFindHeightModifier;
	opts.findValueInfo = baseThreshold;

	itsLogger->Info("Searching for stratus base accurate value");

	auto stratusBase = VerticalExtremeValue(opts);
//VAR Base = VERTZ_FINDH(N_EC,0,Layer,BaseThreshold,1)

	//vector<param> p = { stratusBaseParam, param("HL-M") };
	//stratusBase->Params(p);
	
	stratusBase->First();
	stratusBase->ReplaceParam(stratusBaseParam);
	//stratusBase->NextParam();
	//stratusBase->ReplaceParam(stratusBaseParam);
	
	size_t missing = 0;

	for (stratusBase->ResetLocation(); stratusBase->NextLocation();)
	{
		if (stratusBase->Value() == kHPMissingValue)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus base number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusBase->Grid()->Size()));
	
	itsLogger->Info("Searching for stratus top accurate value");

	constData1->Grid()->Data()->Fill(0);
	constData2->Grid()->Data()->Fill(layer);
	opts.findValueInfo = topThreshold;
	opts.findNthValue = 0; // Find LAST value

	//VAR Top = VERTZ_FINDH(N_EC,0,Layer,TopThreshold,0)

	auto stratusTop = VerticalExtremeValue(opts);

	stratusTop->First();
	stratusTop->ReplaceParam(stratusTopParam);

	missing = 0;
	
	for (stratusTop->ResetLocation(); stratusTop->NextLocation();)
	{
		if (stratusTop->Value() == kHPMissingValue)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus top number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (stratusTop->Grid()->Size()));

	// Keskimääräinen RH stratuksen yläpuolisessa kerroksessa (jäätävä tihku)

	itsLogger->Info("Searching for humidity in layers above stratus top");

	opts.wantedParam = param("RH-PRCNT");
	opts.wantedModifier = kAverageModifier;
	opts.firstLevelValueInfo = stratusTop;
	opts.lastLevelValueInfo = stratusTop;

	itsFirstLevelValueBase = 100;
	itsLastLevelValueBase = drydz;

	itsScale = 100;

	auto upperLayerRH = VerticalExtremeValue(opts);

	upperLayerRH->First();
	upperLayerRH->ReplaceParam(stratusUpperLayerRHParam);

	missing = 0;

	for (upperLayerRH->ResetLocation(); upperLayerRH->NextLocation();)
	{
		if (upperLayerRH->Value() == kHPMissingValue)
		{
			missing++;
		}
	}

	itsLogger->Debug("Upper layer RH number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (upperLayerRH->Grid()->Size()));

	//VERTZ_AVG(RH_EC,Top+100,Top+DRYdz)

	itsLogger->Info("Searching for stratus top temperatue");

	// Stratuksen Topin lämpötila (jäätävä tihku)
	//VAR TTop = VERTZ_GET(T_EC,Top)

	opts.wantedModifier = kFindValueModifier;
	opts.wantedParam = param("T-K");
	opts.findValueInfo = stratusTop;

	auto stratusTopTemp = VerticalExtremeValue(opts);

	stratusTopTemp->First();
	stratusTopTemp->ReplaceParam(stratusTopTempParam);

	itsLogger->Info("Searching for stratus mean temperature");

	// St:n keskimääräinen lämpötila (poissulkemaan kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
	//VAR stTavg = VERTZ_AVG(T_EC,Base+50,Top-50)

	itsFirstLevelValueBase = 50;
	itsLastLevelValueBase = -50;
	opts.firstLevelValueInfo = stratusBase;
	opts.lastLevelValueInfo = stratusBase;

	auto stratusMeanTemp = VerticalExtremeValue(opts);

	stratusMeanTemp->First();
	stratusMeanTemp->ReplaceParam(stratusMeanTempParam);

	// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
	//VAR wAvg = VERTZ_AVG(W_EC,Base,Top)

	itsLogger->Info("Searching for mean vertical velocity in stratus");

	opts.wantedParam = param("VV-MS");
	opts.lastLevelValueInfo = stratusTop;

	itsFirstLevelValueBase = itsLastLevelValueBase = 0;

	auto stratusVerticalVelocity = VerticalExtremeValue(opts);

	stratusVerticalVelocity->First();
	stratusVerticalVelocity->ReplaceParam(stratusVerticalVelocityParam);

	vector<shared_ptr<info>> datas = { stratusTop, upperLayerRH, stratusTopTemp, stratusMeanTemp, stratusVerticalVelocity };

	stratusBase->Merge(datas);

	return stratusBase;
}

shared_ptr<info> hitool::FreezingArea(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime)
{

	const param minusAreaParam("MINUS-AREA-T-C");
	const param plusArea1Param("PLUS-AREA-1-T-C");
	const param plusArea2Param("PLUS-AREA-2-T-C");

	vector<param> params = { param("FAKE-PARAM") };
	vector<forecast_time> times = { wantedTime };
	vector<level> levels = { level(kUnknownLevel, 0, "FAKE-LEVEL") };
	
	auto constData1 = make_shared<info> (*conf->Info());
	constData1->Params(params);
	constData1->Times(times);
	constData1->Levels(levels);

	constData1->Create();
	constData1->First();
	constData1->Grid()->Data()->Fill(0);

	auto constData2 = make_shared<info> (*constData1);
	constData2->ReGrid();
	constData2->First();
	constData2->Grid()->Data()->Fill(5000);

	auto constData3 = make_shared<info> (*constData2);
	constData1->ReGrid();

	hitool_search_options opts (param("T-K"),
									wantedTime,
									kHybrid,
									constData1,
									constData2,
									kCountModifier,
									conf,
									true,
									1
	);

	opts.findValueInfo = constData3;
	
	itsBase = -273.15;

	// 0-kohtien lkm pinnasta (yläraja 5km, jotta ylinkin nollakohta varmasti löytyy)

	auto numZeroLevels = VerticalExtremeValue(opts);
	
	//nZeroLevel = VERTZ_FINDC(T_EC,0,5000,0)

	numZeroLevels->First();

	/* Check which values we have. Will slow down processing a bit but
	 * will make subsequent code much easier to understand.
	 */

	bool haveOne = false;
	bool haveTwo = false;
	bool haveThree = false;

	for (numZeroLevels->ResetLocation(); numZeroLevels->NextLocation();)
	{
		size_t numZeroLevel = numZeroLevels->Value();

		if (numZeroLevel == 1)
		{
			haveOne = true;
		}
		else if (numZeroLevel == 2)
		{
			haveTwo = true;
		}
		else if (numZeroLevel == 3)
		{
			haveThree = true;
		}

		if (haveOne && haveTwo && haveThree)
		{
			break;
		}
	}

	// Get necessary source data based on loop data above

	shared_ptr<info> zeroLevel1, zeroLevel2, zeroLevel3;
	shared_ptr<info> Tavg1, Tavg2, Tavg3;

	if (haveOne)
	{
		itsLogger->Info("Searching for first zero level height");
		
		// Find height of first zero level
		opts.wantedModifier = kFindHeightModifier;

		zeroLevel1 = VerticalExtremeValue(opts);

		opts.lastLevelValueInfo = zeroLevel1;
		opts.wantedModifier = kAverageModifier;

		Tavg1 = VerticalExtremeValue(opts);
	}

	if (haveTwo)
	{
		assert(haveOne);

		itsLogger->Info("Searching for second zero level height");

		// Find height of second zero level

		opts.wantedModifier = kFindHeightModifier;
		opts.findNthValue = 2;

		zeroLevel2 = VerticalExtremeValue(opts);

		assert(zeroLevel1);

		opts.firstLevelValueInfo = zeroLevel1;
		opts.lastLevelValueInfo = zeroLevel2;
		opts.wantedModifier = kAverageModifier;

		Tavg2 = VerticalExtremeValue(opts);
	}

	if (haveThree)
	{
		assert(haveOne);
		assert(haveTwo);

		itsLogger->Info("Searching for third zero level height");

		// Find height of third zero level

		opts.wantedModifier = kFindHeightModifier;
		opts.findNthValue = 3;

		zeroLevel3 = VerticalExtremeValue(opts);

		assert(zeroLevel1);
		assert(zeroLevel2);

		opts.firstLevelValueInfo = zeroLevel1;
		opts.lastLevelValueInfo = zeroLevel2;
		opts.wantedModifier = kAverageModifier;

		Tavg3 = VerticalExtremeValue(opts);
	}

	auto plusArea1 = make_shared<info> (*numZeroLevels);
	plusArea1->ReGrid();
	plusArea1->Grid()->Data()->Fill(kHPMissingValue);

	auto minusArea = make_shared<info> (*plusArea1);
	minusArea->ReGrid();

	auto plusArea2 = make_shared<info> (*plusArea1);
	plusArea2->ReGrid();

	for (numZeroLevels->ResetLocation(); numZeroLevels->NextLocation(); )
	{
		size_t numZeroLevel = numZeroLevels->Value();
		size_t locationIndex = numZeroLevels->LocationIndex();
		
		plusArea1->LocationIndex(locationIndex);
		plusArea2->LocationIndex(locationIndex);
		minusArea->LocationIndex(locationIndex);

		if (numZeroLevel == 0)
		{
			continue;
		}
		if (numZeroLevel == 1)
		{
			zeroLevel1->LocationIndex(locationIndex);
			Tavg1->LocationIndex(locationIndex);
			plusArea1->Value(zeroLevel1->Value() * Tavg1->Value());
		}
		else if (numZeroLevel == 2)
		{
			zeroLevel1->LocationIndex(locationIndex);
			zeroLevel2->LocationIndex(locationIndex);

			Tavg1->LocationIndex(locationIndex);
			Tavg2->LocationIndex(locationIndex);

			plusArea1->Value((zeroLevel2->Value() - zeroLevel1->Value()) * Tavg2->Value());
			minusArea->Value(zeroLevel1->Value() * Tavg1->Value());
		}
		else if (numZeroLevel == 3)
		{
			zeroLevel1->LocationIndex(locationIndex);
			zeroLevel2->LocationIndex(locationIndex);
			zeroLevel3->LocationIndex(locationIndex);

			Tavg1->LocationIndex(locationIndex);
			Tavg2->LocationIndex(locationIndex);
			Tavg3->LocationIndex(locationIndex);

			plusArea2->Value((zeroLevel3->Value() - zeroLevel2->Value()) * Tavg2->Value());
			plusArea1->Value(zeroLevel1->Value() * Tavg1->Value() + plusArea2->Value());
			minusArea->Value((zeroLevel2->Value() - zeroLevel1->Value()) * Tavg3->Value());
		}
		
	}
	
	minusArea->ReplaceParam(minusAreaParam);
	plusArea1->ReplaceParam(plusArea1Param);
	plusArea2->ReplaceParam(plusArea2Param);
	
	vector<shared_ptr<info>> snafu = { plusArea1, plusArea2 };

	minusArea->Merge(snafu);

	return minusArea;
}
