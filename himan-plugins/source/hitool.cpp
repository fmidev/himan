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
#include <algorithm>

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
	long lastHybridLevel = -1;

	// Karkeaa haarukointia

	switch (mod->Type())
	{
#if 0
	case kFindValueModifier:
		{
			double min_value = *min_element(findValue.begin(), findValue.end());
			// double max_value = *max_element(findValue.begin(), findValue.end());

			if (min_value >= 6000.)
			{
				lastHybridLevel = 35;
			}
			else if (min_value >= 1000.)
			{
				lastHybridLevel = 55;
			}
		}
			break;
#endif
		default:
			lastHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));
			break;
	}

	for (long levelValue = lastHybridLevel; levelValue >= firstHybridLevel && !mod->CalculationFinished(); levelValue--)
	{

		level currentLevel(kHybrid, levelValue, "HYBRID");

		valueheight data;

		try
		{
			data = GetData(currentLevel, wantedParam, itsTime);
		}
		catch (const HPExceptionType& e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					throw;
					//itsLogger->Warning("Parameter of height data not found for level " + boost::lexical_cast<string> (currentLevel.Value()));
					//continue;
					break;
				default:
					throw;
					break;
			}
		}
		
		auto values = data.first;
		auto heights = data.second;

		assert(heights->Grid()->Size() == values->Grid()->Size());

		values->First();
		heights->First();

		mod->Process(values->Grid()->Data()->Values(), heights->Grid()->Data()->Values());

		size_t heightsCrossed = mod->HeightsCrossed();

		string msg = "Level " + boost::lexical_cast<string> (currentLevel.Value()) + ": height range crossed for " + boost::lexical_cast<string> (heightsCrossed) +
			"/" + boost::lexical_cast<string> (values->Data()->Size()) + " grid points";

		itsLogger->Debug(msg);

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
				throw;
				break;

			default:
				throw runtime_error("hitool: Unable to proceed");
				break;
		}
	}


	assert(values);
	assert(heights);
	assert(values->Grid()->Size() == heights->Grid()->Size());

	// No Merge() here since that will mess up cache
	
	valueheight ret = valueheight(values,heights);
	return ret;
}

/* CONVENIENCE FUNCTIONS */

vector<double> hitool::VerticalHeight(const vector<param>& wantedParamList,
						double lowerHeight,
						double upperHeight,
						const vector<double>& findValue,
						size_t findNth) const
{
	
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeight(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeight(const vector<param>& wantedParamList,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue,
						const vector<double>& findValue,
						size_t findNth) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];

	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalHeight(foundParam, firstLevelValue, lastLevelValue, findValue, findNth);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}
	}

	throw runtime_error("Data not found");
	
}

vector<double> hitool::VerticalHeight(const param& wantedParam,
						double lowerHeight,
						double upperHeight,
						const vector<double>& findValue,
						size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	
	return VerticalHeight(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

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

vector<double> hitool::VerticalMinimum(const vector<param>& wantedParamList, double lowerHeight, double upperHeight) const
{
	assert(!wantedParamList.empty());

	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalMinimum(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMinimum(const vector<param>& wantedParamList,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];
	
	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalMinimum(foundParam, firstLevelValue, lastLevelValue);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}
	}

	throw runtime_error("Data not found");
}

vector<double> hitool::VerticalMinimum(const param& wantedParam,
						const double& lowerHeight,
						const double& upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid,  wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMinimum(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid,  wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const vector<param>& wantedParamList, double lowerHeight, double upperHeight) const
{
	assert(!wantedParamList.empty());

	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalMaximum(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const vector<param>& wantedParamList,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];

	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalMaximum(foundParam, firstLevelValue, lastLevelValue);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}

	}

	throw runtime_error("Data not found");
}

vector<double> hitool::VerticalMaximum(const param& wantedParam,
						const double& lowerHeight,
						const double& upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalAverage(const params& wantedParamList, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalAverage(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalAverage(const vector<param>& wantedParamList,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];

	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalAverage(foundParam, firstLevelValue, lastLevelValue);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}

	}

	throw runtime_error("Data not found");
}

vector<double> hitool::VerticalAverage(const param& wantedParam,
						const double& lowerHeight,
						const double& upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalAverage(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalSum(const vector<param>& wantedParamList,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];
	
	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalSum(foundParam, firstLevelValue, lastLevelValue);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}
	}

	throw runtime_error("Data not found");
}

vector<double> hitool::VerticalSum(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kAccumulationModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalCount(const vector<param>& wantedParamList,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue,
						const vector<double>& findValue) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];
	
	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalCount(foundParam, firstLevelValue, lastLevelValue, findValue);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}
	}

	throw runtime_error("Data not found");
}

vector<double> hitool::VerticalCount(const param& wantedParam,
						const vector<double>& firstLevelValue,
						const vector<double>& lastLevelValue,
						const vector<double>& findValue) const
{
	return VerticalExtremeValue(CreateModifier(kCountModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

vector<double> hitool::VerticalValue(const vector<param>& wantedParamList, double wantedHeight) const
{
	assert(!wantedParamList.empty());

	vector<double> heightInfo(itsConfiguration->Info()->Grid()->Size(), wantedHeight);

	return VerticalValue(wantedParamList, heightInfo);
}

vector<double> hitool::VerticalValue(const vector<param>& wantedParamList, const vector<double>& heightInfo) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];

	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalValue(foundParam, heightInfo);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger->Debug("Switching parameter from " + foundParam.Name() + " to " + wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger->Warning("No more valid parameters left");
					throw;
				}
			}
			else
			{
				throw;
			}
		}
	}

	throw runtime_error("Data not found");
}

vector<double> hitool::VerticalValue(const param& wantedParam, const double& height) const
{
	vector<double> heightInfo(itsConfiguration->Info()->Grid()->Size(), height);

	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, wantedParam, vector<double> (), vector<double> (), heightInfo);
}

vector<double> hitool::VerticalValue(const param& wantedParam, const vector<double>& heightInfo) const
{
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

	vector<param> wantedParamList({param("N-0TO1"), param("N-PRCNT")});

	/**
	 * Etsitään parametrin N minimiarvo korkeusvälillä 0 .. stLimit (=500)
     */

	itsLogger->Info("Searching for stratus lower limit");

	auto baseThreshold = VerticalMinimum(wantedParamList, constData1, constData2);
	
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

	auto topThreshold = VerticalMinimum(wantedParamList, constData1, constData2);

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

	auto stratusBase = VerticalHeight(wantedParamList, constData1, constData2, baseThreshold);

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
	auto stratusTop = VerticalHeight(wantedParamList, constData1, constData2, topThreshold, 0);

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

	param wantedParam = param("RH-PRCNT");

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

	//wantedParamList = {param("N-0TO1"), param("N-PRCNT")};

	auto stratusMeanN = VerticalAverage(wantedParamList, stratusBase, stratusTop);
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

	wantedParam = { param("T-K") };
	
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

	const param minusAreaParam("MINUS-AREA-MC"); // metriastetta, mC

	const param plusAreaParam("PLUS-AREA-MC"); // metriastetta, mC

	/*
	param plusArea2Param("PLUS-AREA-2-T-K");
	plusArea2Param.Unit(kK);
*/
	vector<param> params = { minusAreaParam, plusAreaParam };
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
	
	/* Check which values we have. Will slow down processing a bit but
	 * will make subsequent code much easier to understand.
	 */

	bool haveOne = false;
	bool haveTwo = false;
	bool haveThree = false;
	bool haveFour = false;

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
		else if (val == 3)
		{
			haveThree = true;
		}
		else if (val >= 4)
		{
			haveFour = true;
		}

		if (haveOne && haveTwo && haveThree && haveFour)
		{
			break;
		}
	}

	// Get necessary source data based on loop data above

	vector<double> zeroLevel1(numZeroLevels.size(), kFloatMissing);

	auto zeroLevel2 = zeroLevel1;
	auto zeroLevel3 = zeroLevel1;
	auto zeroLevel4 = zeroLevel1;
	auto Tavg1 = zeroLevel1;
	auto Tavg2_two = zeroLevel1;
	auto Tavg2_three = zeroLevel1;
	auto Tavg2_four = zeroLevel1;
	auto Tavg3 = zeroLevel1;
	auto plusArea = zeroLevel1;
	auto minusArea = zeroLevel1;

	if (haveOne)
	{
		itsLogger->Info("Searching for first zero level height and value");

		zeroLevel1 = VerticalHeight(wantedParam, constData1, constData2, constData3, 1);

		itsLogger->Info("Searching for average temperature between ground level and first zero level");

		Tavg1 = VerticalAverage(wantedParam, constData1, zeroLevel1);
		
	}

	if (haveTwo)
	{
		itsLogger->Info("Searching for second zero level height and value");

		assert(haveOne);
		
		zeroLevel2 = VerticalHeight(wantedParam, constData1, constData2, constData3, 2);

		itsLogger->Info("Searching for average temperature between first and second zero level");

		Tavg2_two = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	if (haveThree)
	{
		itsLogger->Info("Searching for third zero level height and value");

		assert(haveOne && haveTwo);

		zeroLevel3 = VerticalHeight(wantedParam, constData1, constData2, constData3, 3);

		itsLogger->Info("Searching for average temperature between second and third zero level");

		Tavg2_three = VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);

		itsLogger->Info("Searching for average temperature between first and third zero level");

		Tavg3 = VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
	}

	if (haveFour)
	{
		itsLogger->Info("Searching for fourth zero level height and value");

		zeroLevel4 = VerticalHeight(wantedParam, constData1, constData2, constData3, 4);

		itsLogger->Info("Searching for average temperature between third and fourth zero level");

		Tavg2_four = VerticalAverage(wantedParam, zeroLevel3, zeroLevel4);

	}
	
	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		short numZeroLevel = static_cast<short> (numZeroLevels[i]);

		// nollarajoja parillinen määrä (pintakerros pakkasella)
		// nollarajoja on siis vähintään kaksi

		double pa = kFloatMissing, ma = kFloatMissing;

		if (numZeroLevel%2 == 0)
		{
			double zl1 = zeroLevel1[i], zl2 = zeroLevel2[i];
			double ta1 = Tavg1[i], ta2 = Tavg2_two[i];
			
			double paloft = kFloatMissing;

			if (zl1 != kFloatMissing && zl2 != kFloatMissing 
					&& ta1 != kFloatMissing && ta2 != kFloatMissing)
			{
				ma = zl1 * (ta1 - constants::kKelvin);
				paloft = (zl2 - zl1) * (ta2 - constants::kKelvin);
			}

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 4 nollarajaa)
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)
			
			if (numZeroLevel >= 4)
			{
				double zl3 = zeroLevel3[i], zl4 = zeroLevel4[i];
				ta2 = Tavg2_four[i];

				if (zl3 != kFloatMissing && zl4 != kFloatMissing && ta2 != kFloatMissing)
				{
					paloft = paloft + (zl4 - zl3) * (ta2 - constants::kKelvin);
				}
			}

			pa = paloft;

		}
		
		// nollarajoja pariton määrä (pintakerros plussalla)

		else if (numZeroLevel%2 == 1)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg1[i];
			double pasfc = kFloatMissing, paloft = kFloatMissing;

			if (zl1 != kFloatMissing && ta1 != kFloatMissing)
			{
				pasfc = zl1 * (ta1 - constants::kKelvin);
				pa = pasfc;
			}

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 3 nollarajaa)
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)

			if (numZeroLevel >= 3)
			{
				double zl2 = zeroLevel2[i], zl3 = zeroLevel3[i];
				double ta2 = Tavg2_three[i], ta3 = Tavg3[i];

				if (zl2 != kFloatMissing && zl3 != kFloatMissing &&
						ta2 != kFloatMissing && ta3 != kFloatMissing)
				{
					paloft = (zl3 - zl2) * (ta2 - constants::kKelvin);
					pa = pasfc + paloft;

					ma = (zl2 - zl1) * (ta3 - constants::kKelvin);
				}
			}
		}

		plusArea[i] = pa;
		minusArea[i] = ma;
	}

	ret->Param(minusAreaParam);
	ret->Data()->Set(minusArea);

	ret->Param(plusAreaParam);
	ret->Data()->Set(plusArea);

	return ret;

}

void hitool::Configuration(shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = conf;
}
