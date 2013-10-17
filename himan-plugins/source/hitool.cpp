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

	size_t retGridSize = ret->Grid()->Size();

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

	param stratusBaseParam("STRATUS-BASE-M");
	param stratusTopParam("STRATUS-TOP-M");
	param stratusTopTempParam("STRATUS-TOP-T-K");
	param stratusMeanTempParam("STRATUS-MEAN-T-K");

	std::vector<param> params { stratusBaseParam, stratusTopParam, stratusTopTempParam, stratusMeanTempParam };
	
	auto ret = make_shared<info> (*conf->Info());

	conf->FirstSourceProducer();

	ret->Producer(conf->SourceProducer());
	ret->Params(params);

	// Create data backend

	ret->Create();

	const double stLimit = 500.;
	//const double fzStLimit = 800.;
	const double layer = 2000.;
	const double stCover = 50.;
	const double drydz = 1500.;

	auto constData1 = make_shared<info> (*ret);
	param p("AS-DF"); // Param name does not matter here
	vector<param> pvec = { p };

	constData1->Params(pvec);
	constData1->Create(); constData1->First();
	constData1->Grid()->Data()->Fill(0);

	auto constData2 = make_shared<info> (*constData1);
	constData2->Create(); constData2->First();
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
		//assert(baseThreshold->Value() != kHPMissingValue);

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

	itsLogger->Info("Searching for stratus mean temperatue");

	// St:n keskimääräinen lämpötila (poissulkemaan kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
	//VAR stTavg = VERTZ_AVG(T_EC,Base+50,Top-50)

	itsFirstLevelValueBase = 50;
	itsLastLevelValueBase = -50;
	opts.firstLevelValueInfo = stratusBase;
	opts.lastLevelValueInfo = stratusBase;

	auto stratusMeanTemp = VerticalExtremeValue(opts);

	// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
	//VAR wAvg = VERTZ_AVG(W_EC,Base,Top)

	itsLogger->Info("Searching for mean vertical velocity in stratus");

	opts.wantedParam = param("VV-MS");
	opts.lastLevelValueInfo = stratusTop;

	itsFirstLevelValueBase = itsLastLevelValueBase = 0;

	auto stratusVerticalVelocity = VerticalExtremeValue(opts);

	vector<shared_ptr<info>> datas = { stratusTop, upperLayerRH, stratusTopTemp, stratusMeanTemp, stratusVerticalVelocity };

	stratusBase->Merge(datas);
	
	return stratusBase;
}

shared_ptr<info> hitool::FreezingArea(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime)
{

	const double freezingAreadz = 100.;
	// Kerroksen paksuus pinnasta, josta etsitään min. lämpötilaa [m]
	const double minLayer = 1100.;
	
	// Mallin (korkein) nollaraja [m]

	shared_ptr<info> freezingLevel;
	
	shared_ptr<plugin::fetcher> f = dynamic_pointer_cast <plugin::fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		freezingLevel = f->Fetch(conf,	wantedTime,	level(kHeight, 0), param("H0C-M"));
	}
	catch (const HPExceptionType& e)
	{
		throw e;
	}

	auto minLevel = make_shared<info> (*freezingLevel);
	minLevel->ReGrid();
	
	// Rajoitetaan min. lämpötilan haku (ylimmän) nollarajan alapuolelle

	freezingLevel->First(); freezingLevel->ResetLocation();
	minLevel->First(); minLevel->ResetLocation();

	while (freezingLevel->NextLocation() && minLevel->NextLocation())
	{
		if (freezingLevel->Value() < minLayer)
		{
			minLevel->Value(freezingLevel->Value() - freezingAreadz);
		}
		else
		{
			minLevel->Value(minLayer);
		}
	}

	auto constData = make_shared<info> (*freezingLevel);

	constData->ReGrid();
	constData->First();
	constData->Grid()->Data()->Fill(0);

	// Min lämpötila ja sen korkeus [m]

	hitool_search_options opts (param("T-K"),
									wantedTime,
									kHybrid,
									constData,
									minLevel,
									kMinimumModifier,
									conf,
									true,
									1
	);

	itsLogger->Info("Searching for freezing area min temperature");

	auto Tmin = VerticalExtremeValue(opts);
//VAR Tmin = VERTZ_MIN(T_EC,0,MinLayer)
//VAR TminH = VERTZ_MINH(T_EC,0,MinLayer)
	constData->Grid()->Data()->Fill(100);
	opts.lastLevelValueInfo = freezingLevel;

	itsLogger->Info("Searching for freezing area max temperature");

	auto Tmax = VerticalExtremeValue(opts);
		
// Max lämpötila ja sen korkeus [m]
//VAR Tmax = VERTZ_MAX(T_EC,100,FZlevel)
//VAR TmaxH = VERTZ_MAXH(T_EC,100,FZlevel)

	Tmin->Merge(Tmax);

	return Tmin;
}