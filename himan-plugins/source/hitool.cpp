/**
 * @file hitool.cpp
 *
 */

#include "hitool.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <algorithm>
#include <boost/format.hpp>

#include "fetcher.h"
#include "neons.h"
#include "radon.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

double min(const vector<double>& vec)
{
	double ret = 1e38;

	for (double val : vec)
	{
		if (val < ret) ret = val;
	}

	if (ret == 1e38) ret = MissingDouble();

	return ret;
}

double max(const vector<double>& vec)
{
	double ret = -1e38;

	for (double val : vec)
	{
		if (val > ret) ret = val;
	}

	if (ret == -1e38) ret = MissingDouble();
        
        return ret;
}

pair<double, double> minmax(const vector<double>& vec)
{
	double min = 1e38, max = -1e38;

	for (double val : vec)
	{
		if (IsValid(val))
		{
			if (val < min) min = val;
			if (val > max) max = val;
		}
	}

	if (min == 1e38)
	{
		min = MissingDouble();
		max = MissingDouble();
	}

	return make_pair(min, max);
}

hitool::hitool() : itsTime(), itsForecastType(kDeterministic), itsHeightUnit(kM)
{
	itsLogger = logger("hitool");
}

hitool::hitool(shared_ptr<plugin_configuration> conf) : itsTime(), itsForecastType(kDeterministic), itsHeightUnit(kM)
{
	itsLogger = logger("hitool");
	itsConfiguration = conf;
}

shared_ptr<modifier> hitool::CreateModifier(HPModifierType modifierType) const
{
	shared_ptr<himan::modifier> mod;

	switch (modifierType)
	{
		case kMaximumModifier:
			mod = make_shared<modifier_max>();
			break;

		case kMinimumModifier:
			mod = make_shared<modifier_min>();
			break;

		case kMaximumMinimumModifier:
			mod = make_shared<modifier_maxmin>();
			break;

		case kFindHeightModifier:
			mod = make_shared<modifier_findheight>();
			break;

		case kFindValueModifier:
			mod = make_shared<modifier_findvalue>();
			break;

		case kAverageModifier:
			mod = make_shared<modifier_mean>();
			break;

		case kCountModifier:
			mod = make_shared<modifier_count>();
			break;

		case kAccumulationModifier:
			mod = make_shared<modifier_sum>();
			break;

		case kPlusMinusAreaModifier:
			mod = make_shared<modifier_plusminusarea>();
			break;

		case kFindHeightGreaterThanModifier:
			mod = make_shared<modifier_findheight_gt>();
			break;

		case kFindHeightLessThanModifier:
			mod = make_shared<modifier_findheight_lt>();
			break;

		default:
			itsLogger.Fatal("Unknown modifier type: " + boost::lexical_cast<string>(modifierType));
			abort();
			break;
	}
	itsLogger.Trace("Creating " + string(HPModifierTypeToString.at(mod->Type())));
	return mod;
}

pair<level, level> hitool::LevelForHeight(const producer& prod, double height) const
{
	assert(itsConfiguration);

	using boost::lexical_cast;

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

		case 134:
		case 243:
			producerId = 243;
			break;

		case 199:
		case 210:
			producerId = 210;
			break;

		case 4:
		case 260:
			producerId = 260;
			break;

		default:
			itsLogger.Error("Unsupported producer for hitool::LevelForHeight(): " + lexical_cast<string>(prod.Id()));
			break;
	}

	stringstream query;

	if (itsHeightUnit == kM)
	{
		query << "SELECT min(CASE WHEN maximum_height <= " << height
		      << " THEN level_value ELSE NULL END) AS lowest_level, "
		      << "max(CASE WHEN minimum_height >= " << height << " THEN level_value ELSE NULL END) AS highest_level "
		      << "FROM "
		      << "hybrid_level_height "
		      << "WHERE "
		      << "producer_id = " << producerId;
	}
	else if (itsHeightUnit == kHPa)
	{
		// Add/subtract 1 already in the query, since due to the composition of the query it will return
		// the first level that is higher than lower height and vice versa

		query << "SELECT max(CASE WHEN minimum_pressure <= " << height
		      << " THEN level_value+1 ELSE NULL END) AS lowest_level, "
		      << "min(CASE WHEN maximum_pressure >= " << height
		      << " THEN level_value-1 ELSE NULL END) AS highest_level "
		      << "FROM "
		      << "hybrid_level_height "
		      << "WHERE "
		      << "producer_id = " << producerId;
		;
	}

	HPDatabaseType dbtype = itsConfiguration->DatabaseType();

	vector<string> row;

	long absolutelowest = kHPMissingInt, absolutehighest = kHPMissingInt;

	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);
		n->NeonsDB().Query(query.str());

		row = n->NeonsDB().FetchRow();

		absolutelowest = lexical_cast<long>(n->ProducerMetaData(prod.Id(), "last hybrid level number"));
		absolutehighest = lexical_cast<long>(n->ProducerMetaData(prod.Id(), "first hybrid level number"));
	}

	if (row.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
	{
		auto r = GET_PLUGIN(radon);
		r->RadonDB().Query(query.str());

		row = r->RadonDB().FetchRow();

		absolutelowest = lexical_cast<long>(r->RadonDB().GetProducerMetaData(prod.Id(), "last hybrid level number"));
		absolutehighest = lexical_cast<long>(r->RadonDB().GetProducerMetaData(prod.Id(), "first hybrid level number"));
	}

	long newlowest = absolutelowest, newhighest = absolutehighest;

	if (!row.empty())
	{
		// If requested height is below lowest level (f.ex. 0 meters) or above highest (f.ex. 80km)
		// database query will return null

		if (row[0] != "")
		{
			// SQL query returns the level value that precedes the requested value.
			// For first hybrid level (the highest ie max), get one level above the max level if possible
			// For last hybrid level (the lowest ie min), get one level below the min level if possible
			// This means that we have a buffer of three levels for both directions!

			newlowest = lexical_cast<long>(row[0]) + 1;

			if (newlowest > absolutelowest)
			{
				newlowest = absolutelowest;
			}
		}

		if (row[1] != "")
		{
			newhighest = lexical_cast<long>(row[1]) - 1;

			if (newhighest < absolutehighest)
			{
				newhighest = absolutehighest;
			}
		}

		if (newhighest > newlowest)
		{
			newhighest = newlowest;
		}
	}

	assert(newlowest >= newhighest);

	return make_pair<level, level>(level(kHybrid, newlowest), level(kHybrid, newhighest));
}

vector<double> hitool::VerticalExtremeValue(shared_ptr<modifier> mod, HPLevelType wantedLevelType,
                                            const param& wantedParam, const vector<double>& lowerHeight,
                                            const vector<double>& upperHeight, const vector<double>& findValue) const
{
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

	if (itsHeightUnit == kHPa)
	{
		mod->HeightInMeters(false);
	}

	// Should we loop over all producers ?

	producer prod = itsConfiguration->SourceProducer(0);

	// first means first in sorted order, ie smallest number ie the highest level

	HPDatabaseType dbtype = itsConfiguration->DatabaseType();

	long highestHybridLevel = kHPMissingInt, lowestHybridLevel = kHPMissingInt;

	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);

		highestHybridLevel = boost::lexical_cast<long>(n->ProducerMetaData(prod.Id(), "first hybrid level number"));
		lowestHybridLevel = boost::lexical_cast<long>(n->ProducerMetaData(prod.Id(), "last hybrid level number"));
	}

	if (highestHybridLevel == kHPMissingInt && (dbtype == kRadon || dbtype == kNeonsAndRadon))
	{
		auto r = GET_PLUGIN(radon);

		try
		{
			highestHybridLevel =
			    boost::lexical_cast<long>(r->RadonDB().GetProducerMetaData(prod.Id(), "first hybrid level number"));
			lowestHybridLevel =
			    boost::lexical_cast<long>(r->RadonDB().GetProducerMetaData(prod.Id(), "last hybrid level number"));
		}
		catch (const boost::bad_lexical_cast& e)
		{
			itsLogger.Error("Unable to get hybrid level information from database");
			throw;
		}
	}

	// Karkeaa haarukointia

	string heightUnit = (itsHeightUnit == kM) ? "meters" : "hectopascal";

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

			if (IsMissing(max_value) || IsMissing(min_value))
			{
				itsLogger.Error("Min or max values of given heights are missing");
				throw kFileDataNotFound;
			}

			auto levelsForMaxHeight = LevelForHeight(prod, max_value);
			auto levelsForMinHeight = LevelForHeight(prod, min_value);

			highestHybridLevel = static_cast<long>(levelsForMaxHeight.second.Value());
			lowestHybridLevel = static_cast<long>(levelsForMinHeight.first.Value());

			assert(lowestHybridLevel >= highestHybridLevel);

			itsLogger.Debug("Adjusting level range to " + boost::lexical_cast<string>(lowestHybridLevel) + " .. " +
							boost::lexical_cast<string>(highestHybridLevel) + " for height range " +
							boost::str(boost::format("%.2f") % min_value) + " .. " +
							boost::str(boost::format("%.2f") % max_value) + " " + heightUnit);
		}
		break;

		case kFindValueModifier:
		{
			auto p = ::minmax(findValue);

			double max_value = p.second;  // highest
			double min_value = p.first;   // lowest

                        if (IsMissing(max_value) || IsMissing(min_value))
                        {
                                itsLogger.Error("Min or max values of given heights are missing");
                                throw kFileDataNotFound;
                        }

			if (itsHeightUnit == kHPa)
			{
				// larger value is closer to ground

				double temp = max_value;
				max_value = min_value;
				min_value = temp;

				assert(min_value >= 10);
				assert(max_value < 1200);
			}

			auto levelsForMaxHeight = LevelForHeight(prod, max_value);
			auto levelsForMinHeight = LevelForHeight(prod, min_value);

			highestHybridLevel = static_cast<long>(levelsForMaxHeight.second.Value());
			lowestHybridLevel = static_cast<long>(levelsForMinHeight.first.Value());

			assert(lowestHybridLevel >= highestHybridLevel);

			itsLogger.Debug("Adjusting level range to " + boost::lexical_cast<string>(lowestHybridLevel) + " .. " +
							boost::lexical_cast<string>(highestHybridLevel) + " for height range " +
							boost::str(boost::format("%.2f") % min_value) + " .. " +
							boost::str(boost::format("%.2f") % max_value) + " " + heightUnit);
		}
		break;

		default:
			break;
	}

	for (long levelValue = lowestHybridLevel; levelValue >= highestHybridLevel && !mod->CalculationFinished();
	     levelValue--)
	{
		level currentLevel(kHybrid, levelValue, "HYBRID");

		valueheight data = GetData(currentLevel, wantedParam, itsTime, itsForecastType);

		auto values = data.first;
		auto heights = data.second;

		assert(heights->Grid()->Size() == values->Grid()->Size());

		values->First();
		heights->First();

		mod->Process(values->Grid()->Data().Values(), heights->Grid()->Data().Values());

#ifdef DEBUG
		size_t heightsCrossed = mod->HeightsCrossed();

		string msg = "Level " + boost::lexical_cast<string>(currentLevel.Value()) + ": height range crossed for " +
		             boost::lexical_cast<string>(heightsCrossed) + "/" +
		             boost::lexical_cast<string>(values->Data().Size()) + " grid points";

		itsLogger.Debug(msg);
#endif
	}

	auto ret = mod->Result();

	assert(mod->HeightsCrossed() == ret.size());
	return ret;
}

valueheight hitool::GetData(const level& wantedLevel, const param& wantedParam, const forecast_time& wantedTime,
                            const forecast_type& wantedType) const
{
	shared_ptr<info> values, heights;
	auto f = GET_PLUGIN(fetcher);

	param heightParam;

	if (itsHeightUnit == kM)
	{
		heightParam = param("HL-M");
	}
	else if (itsHeightUnit == kHPa)
	{
		heightParam = param("P-HPA");
	}
	else
	{
		itsLogger.Fatal("Invalid height unit: " + boost::lexical_cast<string>(itsHeightUnit));
	}

	try
	{
		if (!values)
		{
			values = f->Fetch(itsConfiguration, wantedTime, wantedLevel, wantedParam, wantedType);
		}

		if (!heights)
		{
			heights = f->Fetch(itsConfiguration, wantedTime, wantedLevel, heightParam, wantedType);
		}
	}
	catch (HPExceptionType& e)
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
	assert(heights->Data().MissingCount() != heights->Data().Size());

	// No Merge() here since that will mess up cache

	valueheight ret = valueheight(values, heights);
	return ret;
}

/* CONVENIENCE FUNCTIONS */

vector<double> hitool::VerticalHeight(const vector<param>& wantedParamList, double lowerHeight, double upperHeight,
                                      const vector<double>& findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeight(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeight(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
                                      const vector<double>& lastLevelValue, const vector<double>& findValue,
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalHeight(const param& wantedParam, double lowerHeight, double upperHeight,
                                      double findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeight(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeight(const params& wantedParamList, double lowerHeight, double upperHeight,
                                      double findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeight(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeight(const param& wantedParam, double lowerHeight, double upperHeight,
                                      const vector<double>& findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeight(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeight(const param& wantedParam, const vector<double>& firstLevelValue,
                                      const vector<double>& lastLevelValue, const vector<double>& findValue,
                                      size_t findNth) const
{
	auto modifier = CreateModifier(kFindHeightModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue(modifier, kHybrid, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

vector<double> hitool::VerticalHeightGreaterThan(const param& wantedParam, double lowerHeight, double upperHeight,
                                                 double findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeightGreaterThan(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeightGreaterThan(const params& wantedParamList, double lowerHeight, double upperHeight,
                                                 double findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeightGreaterThan(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeightGreaterThan(const param& wantedParam, double lowerHeight, double upperHeight,
                                                 const vector<double>& findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeightGreaterThan(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeightGreaterThan(const vector<param>& wantedParamList, double lowerHeight,
                                                 double upperHeight, const vector<double>& findValue,
                                                 size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeightGreaterThan(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeightGreaterThan(const vector<param>& wantedParamList,
                                                 const vector<double>& firstLevelValue,
                                                 const vector<double>& lastLevelValue, const vector<double>& findValue,
                                                 size_t findNth) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];

	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalHeightGreaterThan(foundParam, firstLevelValue, lastLevelValue, findValue, findNth);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalHeightGreaterThan(const param& wantedParam, const vector<double>& firstLevelValue,
                                                 const vector<double>& lastLevelValue, const vector<double>& findValue,
                                                 size_t findNth) const
{
	auto modifier = CreateModifier(kFindHeightGreaterThanModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue(modifier, kHybrid, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

vector<double> hitool::VerticalHeightLessThan(const param& wantedParam, double lowerHeight, double upperHeight,
                                              double findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeightLessThan(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeightLessThan(const params& wantedParamList, double lowerHeight, double upperHeight,
                                              double findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalHeightLessThan(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

vector<double> hitool::VerticalHeightLessThan(const param& wantedParam, double lowerHeight, double upperHeight,
                                              const vector<double>& findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeightLessThan(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeightLessThan(const vector<param>& wantedParamList, double lowerHeight,
                                              double upperHeight, const vector<double>& findValue, size_t findNth) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalHeightLessThan(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

vector<double> hitool::VerticalHeightLessThan(const vector<param>& wantedParamList,
                                              const vector<double>& firstLevelValue,
                                              const vector<double>& lastLevelValue, const vector<double>& findValue,
                                              size_t findNth) const
{
	assert(!wantedParamList.empty());

	size_t p_i = 0;

	param foundParam = wantedParamList[p_i];

	for (size_t i = 0; i < wantedParamList.size(); i++)
	{
		try
		{
			return VerticalHeightLessThan(foundParam, firstLevelValue, lastLevelValue, findValue, findNth);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				if (++p_i < wantedParamList.size())
				{
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalHeightLessThan(const param& wantedParam, const vector<double>& firstLevelValue,
                                              const vector<double>& lastLevelValue, const vector<double>& findValue,
                                              size_t findNth) const
{
	auto modifier = CreateModifier(kFindHeightLessThanModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue(modifier, kHybrid, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

vector<double> hitool::VerticalMinimum(const vector<param>& wantedParamList, double lowerHeight,
                                       double upperHeight) const
{
	assert(!wantedParamList.empty());

	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalMinimum(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMinimum(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalMinimum(const param& wantedParam, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalMinimum(const param& wantedParam, const vector<double>& firstLevelValue,
                                       const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kMinimumModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const vector<param>& wantedParamList, double lowerHeight,
                                       double upperHeight) const
{
	assert(!wantedParamList.empty());

	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalMaximum(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalMaximum(const param& wantedParam, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalMaximum(const param& wantedParam, const vector<double>& firstLevelValue,
                                       const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kMaximumModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalAverage(const params& wantedParamList, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalAverage(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::VerticalAverage(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalAverage(const param& wantedParam, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalAverage(const param& wantedParam, const vector<double>& firstLevelValue,
                                       const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kAverageModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalSum(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

	return VerticalExtremeValue(CreateModifier(kAccumulationModifier), kHybrid, wantedParam, firstLevelValueVector,
	                            lastLevelValueVector);
}

vector<double> hitool::VerticalSum(const params& wantedParamList, double firstLevelValue, double lastLevelValue) const
{
	vector<double> firstLevelValueVector(itsConfiguration->Info()->Grid()->Size(), firstLevelValue);
	vector<double> lastLevelValueVector(itsConfiguration->Info()->Grid()->Size(), lastLevelValue);

	return VerticalSum(wantedParamList, firstLevelValueVector, lastLevelValueVector);
}

vector<double> hitool::VerticalSum(const param& wantedParam, const vector<double>& firstLevelValue,
                                   const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kAccumulationModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::VerticalCount(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
                                     const vector<double>& lastLevelValue, const vector<double>& findValue) const
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::VerticalCount(const param& wantedParam, const vector<double>& firstLevelValue,
                                     const vector<double>& lastLevelValue, const vector<double>& findValue) const
{
	return VerticalExtremeValue(CreateModifier(kCountModifier), kHybrid, wantedParam, firstLevelValue, lastLevelValue,
	                            findValue);
}

vector<double> hitool::VerticalCount(const param& wantedParam, double firstLevelValue, double lastLevelValue,
                                     double findValue) const
{
	vector<double> firstLevelValueVector(itsConfiguration->Info()->Grid()->Size(), firstLevelValue);
	vector<double> lastLevelValueVector(itsConfiguration->Info()->Grid()->Size(), lastLevelValue);
	vector<double> findValueVector(itsConfiguration->Info()->Grid()->Size(), findValue);

	return VerticalExtremeValue(CreateModifier(kCountModifier), kHybrid, wantedParam, firstLevelValueVector,
	                            lastLevelValueVector, findValueVector);
}

vector<double> hitool::VerticalCount(const params& wantedParamList, double firstLevelValue, double lastLevelValue,
                                     double findValue) const
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, wantedParam, vector<double>(),
	                            vector<double>(), heightInfo);
}

vector<double> hitool::VerticalValue(const param& wantedParam, const vector<double>& heightInfo) const
{
	return VerticalExtremeValue(CreateModifier(kFindValueModifier), kHybrid, wantedParam, vector<double>(),
	                            vector<double>(), heightInfo);
}

vector<double> hitool::PlusMinusArea(const params& wantedParamList, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return PlusMinusArea(wantedParamList, firstLevelValue, lastLevelValue);
}

vector<double> hitool::PlusMinusArea(const vector<param>& wantedParamList, const vector<double>& firstLevelValue,
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
					itsLogger.Debug("Switching parameter from " + foundParam.Name() + " to " +
									wantedParamList[p_i].Name());
					foundParam = wantedParamList[p_i];
				}
				else
				{
					itsLogger.Warning("No more valid parameters left");
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

vector<double> hitool::PlusMinusArea(const param& wantedParam, double lowerHeight, double upperHeight) const
{
	vector<double> firstLevelValue(itsConfiguration->Info()->Grid()->Size(), lowerHeight);
	vector<double> lastLevelValue(itsConfiguration->Info()->Grid()->Size(), upperHeight);

	return VerticalExtremeValue(CreateModifier(kPlusMinusAreaModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

vector<double> hitool::PlusMinusArea(const param& wantedParam, const vector<double>& firstLevelValue,
                                     const vector<double>& lastLevelValue) const
{
	return VerticalExtremeValue(CreateModifier(kPlusMinusAreaModifier), kHybrid, wantedParam, firstLevelValue,
	                            lastLevelValue);
}

void hitool::Time(const forecast_time& theTime) { itsTime = theTime; }
void hitool::ForecastType(const forecast_type& theForecastType) { itsForecastType = theForecastType; }
void hitool::Configuration(shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = make_shared<plugin_configuration>(
	    *conf);  // not an ideal solution but the next line changes iterator positions so we need a local copy
	itsConfiguration->Info()->First();
}

HPParameterUnit hitool::HeightUnit() const { return itsHeightUnit; }
void hitool::HeightUnit(HPParameterUnit theHeightUnit)
{
	if (theHeightUnit != kM && theHeightUnit != kHPa)
	{
		itsLogger.Error("Invalid height unit: " + boost::lexical_cast<string>(theHeightUnit));
		return;
	}

	itsHeightUnit = theHeightUnit;
}
