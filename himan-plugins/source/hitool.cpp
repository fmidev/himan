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
/*			resultParam.Aggregation().Type(kMaximum);
			params.push_back(resultParam);
			resultParam.Aggregation().Type(kMinimum);
			params.push_back(resultParam);

			height.Aggregation().Type(kExternalMaximum);
			params.push_back(height);
			height.Aggregation().Type(kExternalMinimum);
			params.push_back(height);*/
			break;

		case kFindHeightModifier:
			mod = make_shared<modifier_findheight> ();
/*
			if (!opts.findValueInfo)
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}

			mod->FindValue(opts.findValueInfo);*/
			break;

		case kFindValueModifier:
			mod = make_shared<modifier_findvalue> ();
/*
			if (!opts.findValueInfo)
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}

			mod->FindValue(opts.findValueInfo);*/
			break;

		case kAverageModifier:
			mod = make_shared<modifier_mean> ();
//			resultParam.Aggregation().Type(kAverage);
			break;

		case kCountModifier:
			mod = make_shared<modifier_count> ();
/*			resultParam.Aggregation().Type(kAverage);

			if (!opts.findValueInfo)
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}

			mod->FindValue(opts.findValueInfo);*/
			break;

		case kAccumulationModifier:
		default:
			itsLogger->Fatal("Unknown modifier type: " + boost::lexical_cast<string> (modifierType));
			exit(1);
			break;

	}

	return mod;
}
/*
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

			if (!opts.findValueInfo)
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}
			
			mod->FindValue(opts.findValueInfo);
			break;

		case kFindValueModifier:
			mod = make_shared<modifier_findvalue> ();
			params.push_back(resultParam);
			params.push_back(height);

			if (!opts.findValueInfo)
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}

			mod->FindValue(opts.findValueInfo);
			break;

		case kAverageModifier:
			mod = make_shared<modifier_mean> ();
			resultParam.Aggregation().Type(kAverage);
			params.push_back(resultParam);
			break;

		case kCountModifier:
			mod = make_shared<modifier_count> ();
			resultParam.Aggregation().Type(kAverage);
			params.push_back(resultParam);

			if (!opts.findValueInfo)
			{
				throw std::runtime_error(ClassName() + ": findValueInfo unset");
			}

			mod->FindValue(opts.findValueInfo);
			break;
			
		case kAccumulationModifier:
		default:
			itsLogger->Fatal("Unknown modifier type: " + boost::lexical_cast<string> (opts.wantedModifier));
			exit(1);
			break;

	}
	
	return mod;
}
*/

shared_ptr<info> hitool::VerticalExtremeValue(shared_ptr<modifier> mod,
							HPLevelType wantedLevelType,
							const param& sourceParam,
							const param& targetParam,
							const shared_ptr<info> firstLevelValueInfo,
							const shared_ptr<info> lastLevelValueInfo,
							const shared_ptr<info> findValueInfo,
							size_t findNth) const
{
	shared_ptr<plugin::neons> n = dynamic_pointer_cast <plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

	assert(wantedLevelType == kHybrid);

	if (findValueInfo)
	{
		mod->FindValue(findValueInfo);
	}
	// Should we loop over all producers ?

	producer prod = itsConfiguration->SourceProducer(0);
	
	// first means first in sorted order, ie smallest number ie the highest level
	
	long firstHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "first hybrid level number"));
	long lastHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));
	
	/*
	 * Modifier needs an info instance where it will store its data.
	 * This info will be returned from this function.
	 *
	 * We'll use the plugin_configuration info as a base and just change
	 * the parameters. Calling Create() will re-grid the info using
	 * configuration file arguments.
	 */

	assert(itsConfiguration);
	auto ret = make_shared<info>(*itsConfiguration->Info());

	ret->Params({targetParam});
	ret->Levels({level(kHybrid, kHPMissingValue, "HYBRID")});
	
	// Create data backend

	ret->Create();

	mod->Init(ret);

	ret->First();

#ifndef NDEBUG
	size_t retGridSize = ret->Grid()->Size();
#endif

	const double base = sourceParam.Base();
	const double scale = sourceParam.Scale();

	// When all grid points have been finished, ie. the heights read from data
	// are higher than upper level value, stop processing
	
	vector<bool> finishedLocations;
	finishedLocations.resize(ret->Grid()->Size(), false);

	for (int levelValue = lastHybridLevel; levelValue >= firstHybridLevel; levelValue--)
	{

		size_t numFinishedLocations = 0;

		for (size_t i = 0; i < finishedLocations.size(); i++)
		{
			if (finishedLocations[i])
			{
				numFinishedLocations++;
			}
		}

		itsLogger->Debug("Height above limits for " + boost::lexical_cast<string> (numFinishedLocations) +
			"/" + boost::lexical_cast<string> (finishedLocations.size()) + " grid points");

		if (numFinishedLocations == finishedLocations.size())
		{
			break;
		}

		level currentLevel(kHybrid, levelValue, "HYBRID");

		itsLogger->Debug("Current level: " + boost::lexical_cast<string> (currentLevel.Value()));

		valueheight data = GetData(currentLevel, sourceParam, itsTime);

		auto values = data.first;
		auto heights = data.second;

		assert(heights->Grid()->Size() == retGridSize);
		assert(values->Grid()->Size() == retGridSize);

		if (firstLevelValueInfo)
		{
			assert(firstLevelValueInfo->Grid()->Size() == retGridSize);
			firstLevelValueInfo->ResetLocation();
		}
		if (lastLevelValueInfo)
		{
			assert(lastLevelValueInfo->Grid()->Size() == retGridSize);
			lastLevelValueInfo->ResetLocation();
		}
		
		mod->ResetLocation();
		
		values->First(); values->ResetLocation();
		heights->First(); heights->ResetLocation();

	
		while (mod->NextLocation() && values->NextLocation() && heights->NextLocation())
		{
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
	}

	return mod->Result();
}

/*

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
*/
	/*
	 * Modifier needs an info instance where it will store its data.
	 * This info will be returned from this function.
	 *
	 * We'll use the plugin_configuration info as a base and just change
	 * the parameters. Calling Create() will re-grid the info using
	 * configuration file arguments.
	 */
/*
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
		opts.firstLevelValueInfo->ResetLocation();

		assert(opts.lastLevelValueInfo->Grid()->Size() == retGridSize);
		opts.lastLevelValueInfo->ResetLocation();

		mod->ResetLocation();

		values->First(); values->ResetLocation();
		heights->First(); heights->ResetLocation();

		while (mod->NextLocation() && values->NextLocation() && heights->NextLocation() && opts.firstLevelValueInfo->NextLocation() && opts.lastLevelValueInfo->NextLocation())
		{

			double lowerThreshold = opts.firstLevelValueInfo->Value();
			double upperThreshold = opts.lastLevelValueInfo->Value();

			if (lowerThreshold == kFloatMissing || lowerThreshold == kFloatMissing)
			{
				continue;
			}
			else if (upperThreshold == kFloatMissing || upperThreshold == kFloatMissing)
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

 */
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

shared_ptr<info> hitool::VerticalHeight(const param& wantedParam,
						const shared_ptr<info> firstLevelValueInfo,
						const shared_ptr<info> lastLevelValueInfo,
						const shared_ptr<info> findValueInfo,
						size_t findNth) const
{

//	wantedParam.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kFindHeightModifier), kHybrid, wantedParam, param("HL-M"), firstLevelValueInfo, lastLevelValueInfo, findValueInfo, findNth);
}

shared_ptr<info> hitool::VerticalMinimum(const param& wantedParam,
						const shared_ptr<info> firstLevelValueInfo,
						const shared_ptr<info> lastLevelValueInfo,
						size_t findNth) const
{
	//parm.Aggregation(kMinimum);
	
	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo, shared_ptr<info> (), findNth);
}

shared_ptr<info> hitool::VerticalMaximum(const param& wantedParam,
						const shared_ptr<info> firstLevelValueInfo,
						const shared_ptr<info> lastLevelValueInfo,
						size_t findNth) const
{
	//parm.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo, shared_ptr<info> (), findNth);
}

shared_ptr<info> hitool::VerticalAverage(const param& wantedParam,
						const shared_ptr<info> firstLevelValueInfo,
						const shared_ptr<info> lastLevelValueInfo) const
{
	//parm.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo);
}

shared_ptr<info> hitool::VerticalCount(const param& wantedParam,
						const shared_ptr<info> firstLevelValueInfo,
						const shared_ptr<info> lastLevelValueInfo,
						const shared_ptr<info> findValueInfo) const
{
	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, wantedParam, firstLevelValueInfo, lastLevelValueInfo);
}

shared_ptr<info> hitool::VerticalValue(const param& wantedParam,
						const shared_ptr<info> heightInfo) const
{
	//parm.Aggregation(kMinimum);

	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, param("HL-M"), wantedParam);
}

void hitool::Time(const forecast_time& theTime)
{
	itsTime = theTime;
}

std::shared_ptr<info> hitool::Stratus(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime)
{

	const param stratusBaseParam("STRATUS-BASE-M");
	const param stratusTopParam("STRATUS-TOP-M");
	const param stratusTopTempParam("STRATUS-TOP-T-K");
	const param stratusMeanTempParam("STRATUS-MEAN-T-K");
	const param stratusMeanCloudinessParam("STRATUS-MEAN-N-PRCNT");
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

	const double stLimit = 500.;
	const double layer = 2000.;
	const double stCover = 50.;
	const double drydz = 1500.;
	
	auto constData2 = make_shared<info> (*constData1);
	constData2->ReGrid();
	constData2->First();
	constData2->Grid()->Data()->Fill(stLimit);

	// N-kynnysarvot stratuksen ala- ja ylärajalle [%] (tarkkaa stCover arvoa ei aina löydy)

	param wantedParam("N-0TO1");
	wantedParam.Base(0); wantedParam.Scale(100);

	itsLogger->Info("Searching for stratus lower limit");

	auto baseThreshold = VerticalMinimum(wantedParam, constData1, constData2);
	
	// Etsitaan parametrin N minimiarvo välillä 0 .. stLimit

	//auto baseThreshold = VerticalExtremeValue(opts);

	baseThreshold->First();

	for (baseThreshold->ResetLocation(); baseThreshold->NextLocation();)
	{
		if (baseThreshold->Value() == kFloatMissing || baseThreshold->Value() < stCover)
		{
			baseThreshold->Value(stCover);
		}
	}

	constData1->Grid()->Data()->Fill(stLimit);
	constData2->Grid()->Data()->Fill(layer);

	itsLogger->Info("Searching for stratus upper limit");

	auto topThreshold = VerticalMinimum(wantedParam, constData1, constData2);

	topThreshold->First();

	for (topThreshold->ResetLocation(); topThreshold->NextLocation();)
	{
		//assert(topThreshold->Value() != kFloatMissing);

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
	//constData2->Grid()->Data()->Fill(layer);

	//opts.wantedModifier = kFindHeightModifier;
	//opts.findValueInfo = baseThreshold;

	itsLogger->Info("Searching for stratus base accurate value");

	auto stratusBase = VerticalHeight(wantedParam, constData1, constData2, baseThreshold);

	//auto stratusBase = VerticalExtremeValue(opts);
	//VAR Base = VERTZ_FINDH(N_EC,0,Layer,BaseThreshold,1)

	stratusBase->First();
	stratusBase->ReplaceParam(stratusBaseParam);
	
	size_t missing = 0;

	for (stratusBase->ResetLocation(); stratusBase->NextLocation();)
	{
		if (stratusBase->Value() == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus base number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusBase->Grid()->Size()));

	itsLogger->Info("Searching for stratus top accurate value");

	constData1->Grid()->Data()->Fill(0);
	constData2->Grid()->Data()->Fill(layer);

	//VAR Top = VERTZ_FINDH(N_EC,0,Layer,TopThreshold,0)

	auto stratusTop = VerticalHeight(wantedParam, constData1, constData2, topThreshold, 0);

	//auto stratusTop = VerticalExtremeValue(opts);

	stratusTop->First();
	stratusTop->ReplaceParam(stratusTopParam);

	missing = 0;
	
	for (stratusTop->ResetLocation(); stratusTop->NextLocation();)
	{
		if (stratusTop->Value() == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Stratus top number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (stratusTop->Grid()->Size()));

	// Keskimääräinen RH stratuksen yläpuolisessa kerroksessa (jäätävä tihku)

	itsLogger->Info("Searching for humidity in layers above stratus top");

	// Source data is already in percents ?
	wantedParam = param("RH-PRCNT");

	for (constData1->ResetLocation() , constData2->ResetLocation() , stratusTop->ResetLocation();
			constData1->NextLocation() && constData2->NextLocation() && stratusTop->NextLocation()
			; )
	{
		if (stratusTop->Value() == kFloatMissing)
		{
			constData1->Value(kFloatMissing); constData2->Value(kFloatMissing);
		}
		else
		{
			constData1->Value(stratusTop->Value() + 100);
			constData2->Value(stratusTop->Value() + drydz);
		}
	}

	//VERTZ_AVG(RH_EC,Top+100,Top+DRYdz)
	//auto upperLayerRH = VerticalExtremeValue(opts);
	auto upperLayerRH = VerticalAverage(wantedParam, constData1, constData2);

	upperLayerRH->First();
	upperLayerRH->ReplaceParam(stratusUpperLayerRHParam);

	missing = 0;

	for (upperLayerRH->ResetLocation(); upperLayerRH->NextLocation();)
	{
		if (upperLayerRH->Value() == kFloatMissing)
		{
			missing++;
		}
	}

	itsLogger->Debug("Upper layer RH number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (upperLayerRH->Grid()->Size()));

	//VERTZ_AVG(N_EC,Base,Top)

	itsLogger->Info("Searching for stratus mean cloudiness");

	wantedParam = param("N-0TO1");
	wantedParam.Scale(100);

	auto stratusMeanN = VerticalAverage(wantedParam, stratusBase, stratusTop);
	//auto stratusMeanN = VerticalExtremeValue(opts);

	stratusMeanN->ReplaceParam(stratusMeanCloudinessParam);

	itsLogger->Info("Searching for stratus top temperatue");

	// Stratuksen Topin lämpötila (jäätävä tihku)
	//VAR TTop = VERTZ_GET(T_EC,Top)

	wantedParam = param("T-K");
	wantedParam.Base(-273.15);
	
	//auto stratusTopTemp = VerticalExtremeValue(opts);
	auto stratusTopTemp = VerticalValue(wantedParam, stratusTop);

	stratusTopTemp->First();
	stratusTopTemp->ReplaceParam(stratusTopTempParam);

	itsLogger->Info("Searching for stratus mean temperature");

	// St:n keskimääräinen lämpötila (poissulkemaan kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
	//VAR stTavg = VERTZ_AVG(T_EC,Base+50,Top-50)

	for (constData1->ResetLocation() , constData2->ResetLocation() , stratusBase->ResetLocation();
			constData1->NextLocation() && constData2->NextLocation() && stratusBase->NextLocation()
			; )
	{
		if (stratusBase->Value() == kFloatMissing)
		{
			constData1->Value(kFloatMissing); constData2->Value(kFloatMissing);
		}
		else
		{
			constData1->Value(stratusBase->Value() + 50);
			constData2->Value(stratusBase->Value() - 50);
		}
	}
	/*itsFirstLevelValueBase = 50;
	itsLastLevelValueBase = -50;
	opts.firstLevelValueInfo = stratusBase;
	opts.lastLevelValueInfo = stratusBase;

	auto stratusMeanTemp = VerticalExtremeValue(opts);
*/
	auto stratusMeanTemp = VerticalAverage(wantedParam, constData1, constData2);
	
	stratusMeanTemp->First();
	stratusMeanTemp->ReplaceParam(stratusMeanTempParam);

	// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
	//VAR wAvg = VERTZ_AVG(W_EC,Base,Top)

	itsLogger->Info("Searching for mean vertical velocity in stratus");

	wantedParam = param("VV-MS");

	//auto stratusVerticalVelocity = VerticalExtremeValue(opts);
	auto stratusVerticalVelocity = VerticalAverage(wantedParam, stratusBase, stratusTop);

	stratusVerticalVelocity->First();
	stratusVerticalVelocity->ReplaceParam(stratusVerticalVelocityParam);

	vector<shared_ptr<info>> datas = { stratusTop, upperLayerRH, stratusTopTemp, stratusMeanTemp, stratusMeanN, stratusVerticalVelocity };

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

	auto constData3 = make_shared<info> (*constData1);
	constData1->ReGrid();

	// 0-kohtien lkm pinnasta (yläraja 5km, jotta ylinkin nollakohta varmasti löytyy)
	param wantedParam ("T-K");
	wantedParam.Base(-273.15);
	
	auto numZeroLevels = VerticalCount(wantedParam, constData1, constData2, constData3);

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
		size_t val = numZeroLevels->Value();

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

	shared_ptr<info> zeroLevel1, zeroLevel2, zeroLevel3;
	shared_ptr<info> Tavg1, Tavg2, Tavg3;

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
		assert(haveOne);

		itsLogger->Info("Searching for second zero level height and value");

		// Find height of second zero level

		zeroLevel2 = VerticalHeight(wantedParam, constData1, constData2, constData3, 2);

		assert(zeroLevel1);

		if (!haveThree)
		{
			// if we have three, Tavg2 is calculated differently
			Tavg2 = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
		}
	}

	if (haveThree)
	{
		assert(haveOne);
		assert(haveTwo);

		itsLogger->Info("Searching for third zero level height and value");

		// Find height of third zero level

		zeroLevel3 = VerticalHeight(wantedParam, constData1, constData2, constData3, 3);

		assert(zeroLevel1);
		assert(zeroLevel2);

		Tavg2 = VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);
		Tavg3 = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	auto plusArea1 = make_shared<info> (*numZeroLevels);
	plusArea1->ReGrid();
	plusArea1->Grid()->Data()->Fill(kFloatMissing);

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

void hitool::Configuration(shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = conf;
}