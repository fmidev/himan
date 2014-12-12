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
#include <boost/foreach.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

// I tried to use std::min_element and friends but it was just
// too hard when we have to support missing values

double min(const vector<double>& vec)
{ 
	double ret = 1e38;

	BOOST_FOREACH(double val, vec)
	{
		if (val != kFloatMissing && val < ret) ret = val;
	}

	if (ret == 1e38) ret = kFloatMissing;

	return ret;
}

double max(const vector<double>& vec)
{ 
	double ret = -1e38;

	BOOST_FOREACH(double val, vec)
	{
		if (val != kFloatMissing && val > ret) ret = val;
	}

	if (ret == -1e38) ret = kFloatMissing;

	return ret;
}

pair<double,double> minmax(const vector<double>& vec)
{ 
	double min = 1e38, max = -1e38;

	BOOST_FOREACH(double val, vec)
	{
		if (val != kFloatMissing)
		{
			if (val < min) min = val;
			if (val > max) max = val;
		}
	}

	if (min == 1e38)
	{
		min = kFloatMissing;
		max = kFloatMissing;
	}

	return make_pair(min,max);
}

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

		case kPlusMinusAreaModifier:
			mod = make_shared<modifier_plusminusarea> ();
			break;

		default:
			itsLogger->Fatal("Unknown modifier type: " + boost::lexical_cast<string> (modifierType));
			exit(1);
			break;

	}
	itsLogger->Trace("Creating " + string(HPModifierTypeToString.at(mod->Type())));
	return mod;
}

pair<level,level> hitool::LevelForHeight(const producer& prod, double height) const
{
	using boost::lexical_cast;

	auto n = dynamic_pointer_cast <plugin::neons> (plugin_factory::Instance()->Plugin("neons"));
	
	long producerId = 0;
	
	// Hybrid level heights are calculated by himan, so coalesce the related 
	// forecast producer id with the himan producer id.
	
	switch (prod.Id())
	{
		case 1:
		case 230:
			producerId = 230;
			break;
		
		case 131:
		case 240:
			producerId = 240;
			break;
			
		case 199:
		case 210:
			producerId = 210;
			break;
			
		default:
			itsLogger->Warning("Unsupported producer for hitool::LevelForHeight(): " + lexical_cast<string> (prod.Id()));
			break;
	}
	
	stringstream query;
	
	query << "SELECT min(CASE WHEN maximum_height <= " << height << " THEN level_value ELSE NULL END) AS lowest_level, "
			<< "max(CASE WHEN minimum_height >= " << height << " THEN level_value ELSE NULL END) AS highest_level "
			<< "FROM "
			<< "hybrid_level_height "
			<< "WHERE "
			<< "producer_id = " << producerId;

	n->NeonsDB().Query(query.str());
	
	auto row = n->NeonsDB().FetchRow();
	
	long lowest = lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));
	long highest = lexical_cast<long> (n->ProducerMetaData(prod.Id(), "first hybrid level number"));
	
	if (!row.empty())
	{

		// If requested height is below lowest level (f.ex. 0 meters) or above highest (f.ex. 80km)
		// database query will return null

		long newlowest = (row[0] == "") ? lowest : lexical_cast<long> (row[0]);
		long newhighest = (row[1] == "") ? highest : lexical_cast<long> (row[1]);

		// SQL query returns the level value that precedes the requested value.
		// For first hybrid level (the highest ie max), get one level above the max level if possible
		// For last hybrid level (the lowest ie min), get one level below the min level if possible
		// This means that we have a buffer of two levels for both directions!

		lowest = (newlowest == lowest) ? lowest : newlowest + 1;
		
		highest = (newhighest == highest) ? highest : newhighest - 1;

		if (highest > lowest)
		{
			highest = lowest;
		}
	}
			
	return make_pair<level, level> (level(kHybrid, lowest), level(kHybrid, highest));
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
	long lastHybridLevel = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));

	// Karkeaa haarukointia

	switch (mod->Type())
	{

		/*
		 * -- FIRST HYBRID LEVEL --
		 * 
		 * -- pick highest max above MAX HEIGHT --
		 * 
		 * -- MAX HEIGHT in data--
		 *
		 *  
		 * -- MIN HEIGHT in data --
		 * 
		 * -- pick lowest min below MIN HEIGHT --
		 * 
		 * -- LAST HYBRID LEVEL --
		 */

		case kAverageModifier:
		case kMinimumModifier:
		case kMaximumModifier:
		case kCountModifier:
		{
			double max_value = ::max(upperHeight);
			double min_value = ::min(lowerHeight);

			if (max_value == kFloatMissing || min_value == kFloatMissing)
			{
				itsLogger->Error("Min or max values of given heights are missing");
				throw kFileDataNotFound;
			}

			auto levelsForMaxHeight = LevelForHeight(prod, max_value);
			auto levelsForMinHeight = LevelForHeight(prod, min_value);
		
			firstHybridLevel = static_cast<long> (levelsForMaxHeight.second.Value());
			lastHybridLevel = static_cast<long> (levelsForMinHeight.first.Value());

			itsLogger->Debug("Adjusting level range to " + boost::lexical_cast<string> (lastHybridLevel) + " .. " + boost::lexical_cast<string> (firstHybridLevel) + " for height range " 
				+ boost::lexical_cast<string> (util::round(min_value, 1)) + " .. " + boost::lexical_cast<string> (util::round(max_value, 1)) + " meters");
		}
			break;

		case kFindValueModifier:
		{
			auto p = ::minmax(findValue);
			double max_value = p.second;
			double min_value = p.first;
		
			auto levelsForMaxHeight = LevelForHeight(prod, max_value);
			auto levelsForMinHeight = LevelForHeight(prod, min_value);
		
			firstHybridLevel = static_cast<long> (levelsForMaxHeight.second.Value());
			lastHybridLevel = static_cast<long> (levelsForMinHeight.first.Value());

			itsLogger->Debug("Adjusting level range to " + boost::lexical_cast<string> (lastHybridLevel) + " .. " + boost::lexical_cast<string> (firstHybridLevel) + " for height range " 
				+ boost::lexical_cast<string> (util::round(min_value, 1)) + " .. " + boost::lexical_cast<string> (util::round(max_value, 1)) + " meters");

					
		}
			break;

		default:			
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
					itsLogger->Error("data not found for param " + wantedParam.Name() + " level " + static_cast<string> (currentLevel));
					throw;
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

		mod->Process(values->Grid()->Data().Values(), heights->Grid()->Data().Values());

		size_t heightsCrossed = mod->HeightsCrossed();

		string msg = "Level " + boost::lexical_cast<string> (currentLevel.Value()) + ": height range crossed for " + boost::lexical_cast<string> (heightsCrossed) +
			"/" + boost::lexical_cast<string> (values->Data().Size()) + " grid points";

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
						double findValue,
						size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeight(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeight(const params& wantedParamList,
						double lowerHeight,
						double upperHeight,
						double findValue,
						size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeight(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
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
						double lowerHeight,
						double upperHeight) const
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
						double lowerHeight,
						double upperHeight) const
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
						double lowerHeight,
						double upperHeight) const
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

vector<double> hitool::VerticalSum(const param& wantedParam, double firstLevelValue, double lastLevelValue) const
{
	vector<double> firstLevelValueVector(itsConfiguration->Info()->Grid()->Size(), firstLevelValue);
	vector<double> lastLevelValueVector(itsConfiguration->Info()->Grid()->Size(), lastLevelValue);

	return VerticalExtremeValue(CreateModifier(kAccumulationModifier), kHybrid, wantedParam, firstLevelValueVector, lastLevelValueVector);
}

vector<double> hitool::VerticalSum(const params& wantedParamList, double firstLevelValue, double lastLevelValue) const
{
	vector<double> firstLevelValueVector(itsConfiguration->Info()->Grid()->Size(), firstLevelValue);
	vector<double> lastLevelValueVector(itsConfiguration->Info()->Grid()->Size(), lastLevelValue);

	return VerticalSum(wantedParamList, firstLevelValueVector, lastLevelValueVector);
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

vector<double> hitool::VerticalCount(const param& wantedParam, double firstLevelValue, double lastLevelValue, double findValue) const
{
	vector<double> firstLevelValueVector(itsConfiguration->Info()->Grid()->Size(), firstLevelValue);
	vector<double> lastLevelValueVector(itsConfiguration->Info()->Grid()->Size(), lastLevelValue);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalExtremeValue(CreateModifier(kCountModifier), kHybrid, wantedParam, firstLevelValueVector, lastLevelValueVector, findValueVector);
}

vector<double> hitool::VerticalCount(const params& wantedParamList, double firstLevelValue, double lastLevelValue, double findValue) const
{
	vector<double> firstLevelValueVector(itsConfiguration->Info()->Grid()->Size(), firstLevelValue);
	vector<double> lastLevelValueVector(itsConfiguration->Info()->Grid()->Size(), lastLevelValue);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalCount(wantedParamList, firstLevelValueVector, lastLevelValueVector, findValueVector);
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

vector<double> hitool::VerticalValue(const param& wantedParam, double height) const
{
	vector<double> heightInfo(itsConfiguration->Info()->Grid()->Size(), height);

	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, wantedParam, vector<double> (), vector<double> (), heightInfo);
}

vector<double> hitool::VerticalValue(const param& wantedParam, const vector<double>& heightInfo) const
{
	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, wantedParam, vector<double> (), vector<double> (), heightInfo);
}

vector<double> hitool::PlusMinusArea(const params& wantedParamList, double lowerHeight, double upperHeight) const
{
    vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
    vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

    return PlusMinusArea(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::PlusMinusArea(const vector<param>& wantedParamList,
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
            return PlusMinusArea(foundParam, firstLevelValue, lastLevelValue);
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

vector<double> hitool::PlusMinusArea(const param& wantedParam,
                        double lowerHeight,
                        double upperHeight) const
{
    vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
    vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

    return VerticalExtremeValue(CreateModifier(kPlusMinusAreaModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

vector<double> hitool::PlusMinusArea(const param& wantedParam,
                        const vector<double>& firstLevelValue,
                        const vector<double>& lastLevelValue) const
{
    return VerticalExtremeValue(CreateModifier(kPlusMinusAreaModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue);
}

void hitool::Time(const forecast_time& theTime)
{
	itsTime = theTime;
}

void hitool::Configuration(shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = conf;
	itsConfiguration->Info()->First();
}

