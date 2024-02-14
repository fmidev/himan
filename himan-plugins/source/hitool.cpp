/**
 * @file hitool.cpp
 *
 */

#include "hitool.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <algorithm>
#include <shared_mutex>

#include "fetcher.h"
#include "radon.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

static map<string, param> cachedParamMap;
shared_mutex m;

param GetCachedParamName(const param& p)
{
	shared_lock<shared_mutex> lock(m);

	if (cachedParamMap.find(p.Name()) != cachedParamMap.end())
	{
		return cachedParamMap[p.Name()];
	}

	return p;
}
void UpdateCachedParamName(const vector<param>& params, const param& par)
{
	unique_lock<shared_mutex> lock(m);

	for (const auto& p : params)
	{
		cachedParamMap[p.Name()] = par;
	}
}

hitool::hitool() : itsTime(), itsForecastType(kDeterministic), itsHeightUnit(kM), itsLevelType(kHybrid)
{
	itsLogger = logger("hitool");
}
hitool::hitool(shared_ptr<plugin_configuration> conf)
    : itsTime(), itsForecastType(kDeterministic), itsHeightUnit(kM), itsLevelType(kHybrid)
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
			itsLogger.Fatal("Unknown modifier type: " + to_string(modifierType));
			himan::Abort();
			break;
	}
	itsLogger.Trace("Creating " + string(HPModifierTypeToString.at(mod->Type())));
	return mod;
}

pair<level, level> hitool::LevelForHeight(const producer& prod, double height, const std::string& geomName) const
{
	ASSERT(itsConfiguration);

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

		case 7:
		case 270:
			producerId = 270;
			break;

		case 271:
			producerId = 271;
			break;

		case 8:
		case 272:
			producerId = 272;
			break;

		case 10:
		case 261:
			producerId = 10;
			break;

		case 11:
		case 265:
			producerId = 265;
			break;

		case 170:
		case 220:
			producerId = 220;
			break;

		default:
			itsLogger.Error("Unsupported producer for hitool::LevelForHeight(): " + to_string(prod.Id()));
			break;
	}

	stringstream query;

	if (itsHeightUnit == kM)
	{
		query << "WITH levels AS (SELECT max(CASE WHEN attribute = 'last hybrid level number' THEN value::int ELSE "
		         "NULL END) AS lowest, max(CASE WHEN attribute = 'first hybrid level number' THEN value::int ELSE NULL "
		         "END) AS highest FROM producer_meta WHERE producer_id = "
		      << producerId << ") SELECT min(CASE WHEN maximum_height <= " << height
		      << " THEN level_value ELSE l.lowest END) AS lowest_level, max(CASE WHEN minimum_height >= " << height
		      << " THEN level_value ELSE l.highest END) AS highest_level FROM hybrid_level_height, levels l WHERE "
		         "producer_id = "
		      << producerId;
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
	}

	if (geomName.empty() == false)
	{
		query << " AND geometry_id = (SELECT id FROM geom WHERE name = '" << geomName << "')";
	}

	HPDatabaseType dbtype = itsConfiguration->DatabaseType();

	vector<string> row;

	long absolutelowest = kHPMissingInt, absolutehighest = kHPMissingInt;

	if (dbtype == kRadon)
	{
		auto r = GET_PLUGIN(radon);
		r->RadonDB().Query(query.str());

		row = r->RadonDB().FetchRow();

		absolutelowest = stol(r->RadonDB().GetProducerMetaData(prod.Id(), "last hybrid level number"));
		absolutehighest = stol(r->RadonDB().GetProducerMetaData(prod.Id(), "first hybrid level number"));

		if (row.empty() == false && row[0].empty() && geomName.empty() == false)
		{
			// backwards compatibility: if no data is found with this geometry name, try without one
			// note: this should be removed later when we have height data for all geometries

			itsLogger.Warning(
			    fmt::format("Geometry {} is missing height information at radon table hybrid_level_height", geomName));

			auto q = query.str();
			q = q.substr(0, q.find("AND geometry_id"));

			r->RadonDB().Query(q);

			row = r->RadonDB().FetchRow();
		}

		try
		{
			itsLevelType = HPStringToLevelType.at(r->RadonDB().GetProducerMetaData(prod.Id(), "hybrid level type"));
		}
		catch (const exception& e)
		{
		}
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

			newlowest = stol(row[0]) + 1;

			if (newlowest > absolutelowest)
			{
				newlowest = absolutelowest;
			}
		}

		if (row[1] != "")
		{
			newhighest = stol(row[1]) - 1;

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

	ASSERT(newlowest >= newhighest);

	double l = static_cast<double>(newlowest);
	double h = static_cast<double>(newhighest);

	if (itsLevelType == kHybrid)
	{
		return make_pair(level(itsLevelType, l), level(itsLevelType, h));
	}
	else
	{
		return make_pair(level(itsLevelType, l, l + 1), level(itsLevelType, h, h + 1));
	}
}

template <typename T>
vector<T> hitool::VerticalExtremeValue(shared_ptr<modifier> mod, const params& wantedParams,
                                       const vector<T>& lowerHeight, const vector<T>& upperHeight,
                                       const vector<T>& findValue) const
{
	vector<T> ret;
	for (const auto& par : wantedParams)
	{
		try
		{
			auto _par = GetCachedParamName(par);
			ret = VerticalExtremeValue<T>(mod, _par, lowerHeight, upperHeight, findValue);
			UpdateCachedParamName(wantedParams, _par);
			return ret;
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				continue;
			}
			else
			{
				throw;
			}
		}
	}

	return ret;
}

template <typename T>
vector<T> hitool::VerticalExtremeValue(shared_ptr<modifier> mod, const param& wantedParam, const vector<T>& lowerHeight,
                                       const vector<T>& upperHeight, const vector<T>& findValue) const
{
	mod->FindValue(util::Convert<T, double>(findValue));
	mod->LowerHeight(util::Convert<T, double>(lowerHeight));
	mod->UpperHeight(util::Convert<T, double>(upperHeight));

	if (itsHeightUnit == kHPa)
	{
		mod->HeightInMeters(false);
	}

	// Should we loop over all producers ?

	producer prod = itsConfiguration->SourceProducer(0);

	// first means first in sorted order, ie smallest number ie the highest level

	HPDatabaseType dbtype = itsConfiguration->DatabaseType();

	long highestHybridLevel = kHPMissingInt, lowestHybridLevel = kHPMissingInt;

	if (dbtype == kRadon)
	{
		auto r = GET_PLUGIN(radon);

		try
		{
			highestHybridLevel = stol(r->RadonDB().GetProducerMetaData(prod.Id(), "first hybrid level number"));
			lowestHybridLevel = stol(r->RadonDB().GetProducerMetaData(prod.Id(), "last hybrid level number"));
		}
		catch (const invalid_argument& e)
		{
			itsLogger.Error("Unable to get hybrid level information from database");
			throw;
		}

		try
		{
			itsLevelType = HPStringToLevelType.at(r->RadonDB().GetProducerMetaData(prod.Id(), "hybrid level type"));
		}
		catch (const exception& e)
		{
		}
	}

	// Karkeaa haarukointia

	string heightUnit = (itsHeightUnit == kM) ? "meters" : "hectopascal";

	switch (mod->Type())
	{
		case kAverageModifier:
		case kMinimumModifier:
		case kMaximumModifier:
		case kCountModifier:
		case kFindHeightModifier:
		case kFindHeightGreaterThanModifier:
		case kFindHeightLessThanModifier:
		{
			auto iter =
			    std::max_element(upperHeight.begin(), upperHeight.end(),
			                     [](const T& val1, const T& val2) { return (val1 < val2) ? true : IsMissing(val1); });

			const T max_value = *iter;

			iter =
			    std::min_element(lowerHeight.begin(), lowerHeight.end(),
			                     [](const T& val1, const T& val2) { return (val1 < val2) ? true : IsMissing(val2); });

			const T min_value = *iter;

			if (IsMissing(max_value) || IsMissing(min_value))
			{
				itsLogger.Error("Min or max values of given heights are missing");
				throw kFileDataNotFound;
			}

			auto levelsForMaxHeight = LevelForHeight(prod, max_value, itsConfiguration->TargetGeomName());
			auto levelsForMinHeight = LevelForHeight(prod, min_value, itsConfiguration->TargetGeomName());

			highestHybridLevel = static_cast<long>(levelsForMaxHeight.second.Value());
			lowestHybridLevel = static_cast<long>(levelsForMinHeight.first.Value());

			ASSERT(lowestHybridLevel >= highestHybridLevel);

			itsLogger.Debug(fmt::format("Adjusting level range to {} .. {} for height range {:.1f} .. {:.1f} {}",
			                            lowestHybridLevel, highestHybridLevel, min_value, max_value, heightUnit));
		}
		break;

		case kFindValueModifier:
		{
			// Seems like minmax_elements is impossible to get to work with nan-values?
			T min_value = numeric_limits<T>::max(), max_value = numeric_limits<T>::lowest();

			for (const auto& v : findValue)
			{
				if (IsMissing(v))
					continue;

				min_value = min(min_value, v);
				max_value = max(max_value, v);
			}

			if (max_value == numeric_limits<T>::lowest() || min_value == numeric_limits<T>::max())
			{
				itsLogger.Error("Min or max values of given heights are missing");
				throw kFileDataNotFound;
			}

			if (itsHeightUnit == kHPa)
			{
				// larger value is closer to ground

				std::swap(max_value, min_value);

				ASSERT(min_value >= 10);
				ASSERT(max_value < 1200);
			}

			auto levelsForMaxHeight = LevelForHeight(prod, max_value, itsConfiguration->TargetGeomName());
			auto levelsForMinHeight = LevelForHeight(prod, min_value, itsConfiguration->TargetGeomName());

			highestHybridLevel = static_cast<long>(levelsForMaxHeight.second.Value());
			lowestHybridLevel = static_cast<long>(levelsForMinHeight.first.Value());

			ASSERT(lowestHybridLevel >= highestHybridLevel);

			itsLogger.Debug(fmt::format("Adjusting level range to {} .. {} for height range {:.1f} .. {:.1f} {}",
			                            lowestHybridLevel, highestHybridLevel, min_value, max_value, heightUnit));
		}
		break;

		default:
			break;
	}

	for (long levelValue = lowestHybridLevel; levelValue >= highestHybridLevel && !mod->CalculationFinished();
	     levelValue--)
	{
		const double lv = static_cast<double>(levelValue);
		level currentLevel =
		    itsLevelType == kHybrid ? level(itsLevelType, lv, "HYBRID") : level(kGeneralizedVerticalLayer, lv, lv + 1);

		auto data = GetData<double>(currentLevel, wantedParam, itsTime, itsForecastType);

		auto values = data.first;
		auto heights = data.second;

		ASSERT(heights->Grid()->Size() == values->Grid()->Size());

		values->First();
		heights->First();

		mod->Process(values->Data().Values(), heights->Data().Values());

#ifdef DEBUG
		size_t heightsCrossed = mod->HeightsCrossed();

		string msg = "Level " + to_string(currentLevel.Value()) + ": height range crossed for " +
		             to_string(heightsCrossed) + "/" + to_string(values->Data().Size()) + " grid points";

		itsLogger.Debug(msg);
#endif
	}

	auto ret = mod->Result();
	const auto hc = static_cast<double>(mod->HeightsCrossed());
	const auto sz = static_cast<double>(itsConfiguration->BaseGrid()->Size());

	if (hc < sz && hc / sz < 0.9)
	{
		itsLogger.Warning(fmt::format("{} grid points are missing. Are there enough vertical levels?", sz - hc));
	}

	return util::Convert<double, T>(ret);
}

template vector<double> hitool::VerticalExtremeValue<double>(shared_ptr<modifier>, const param&, const vector<double>&,
                                                             const vector<double>&, const vector<double>&) const;
template vector<float> hitool::VerticalExtremeValue<float>(shared_ptr<modifier>, const param&, const vector<float>&,
                                                           const vector<float>&, const vector<float>&) const;

template <typename T>
pair<shared_ptr<info<T>>, shared_ptr<info<T>>> hitool::GetData(const level& wantedLevel, const param& wantedParam,
                                                               const forecast_time& wantedTime,
                                                               const forecast_type& wantedType) const
{
	shared_ptr<info<T>> values, heights;
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
		itsLogger.Fatal("Invalid height unit: " + to_string(itsHeightUnit));
	}

	try
	{
		values = f->Fetch<T>(itsConfiguration, wantedTime, wantedLevel, wantedParam, wantedType);
		heights = f->Fetch<T>(itsConfiguration, wantedTime, wantedLevel, heightParam, wantedType);
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

	ASSERT(values);
	ASSERT(heights);
	ASSERT(values->Grid()->Size() == heights->Grid()->Size());
	ASSERT(heights->Data().MissingCount() != heights->Data().Size());

	// No Merge() here since that will mess up cache

	return make_pair(values, heights);
}

template pair<shared_ptr<info<double>>, shared_ptr<info<double>>> hitool::GetData<double>(const level&, const param&,
                                                                                          const forecast_time&,
                                                                                          const forecast_type&) const;

/* CONVENIENCE FUNCTIONS */

template <typename T>
vector<T> hitool::VerticalHeight(const vector<param>& wantedParamList, T lowerHeight, T upperHeight,
                                 const vector<T>& findValue, int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalHeight<T>(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

template vector<double> hitool::VerticalHeight<double>(const params&, double, double, const vector<double>&, int) const;
template vector<float> hitool::VerticalHeight<float>(const params&, float, float, const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeight(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                 const vector<T>& lastLevelValue, const vector<T>& findValue, int findNth) const
{
	auto modifier = CreateModifier(kFindHeightModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue<T>(modifier, wantedParamList, firstLevelValue, lastLevelValue, findValue);
}

template vector<double> hitool::VerticalHeight<double>(const vector<param>&, const vector<double>&,
                                                       const vector<double>&, const vector<double>&, int) const;
template vector<float> hitool::VerticalHeight<float>(const vector<param>&, const vector<float>&, const vector<float>&,
                                                     const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeight(const param& wantedParam, T lowerHeight, T upperHeight, T findValue, int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalHeight<T>(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

template vector<double> hitool::VerticalHeight<double>(const param&, double, double, double, int) const;
template vector<float> hitool::VerticalHeight<float>(const param&, float, float, float, int) const;

template <typename T>
vector<T> hitool::VerticalHeight(const params& wantedParamList, T lowerHeight, T upperHeight, T findValue,
                                 int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalHeight<T>(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

template vector<double> hitool::VerticalHeight<double>(const params&, double, double, double, int) const;
template vector<float> hitool::VerticalHeight<float>(const params&, float, float, float, int) const;

template <typename T>
vector<T> hitool::VerticalHeight(const param& wantedParam, T lowerHeight, T upperHeight, const vector<T>& findValue,
                                 int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalHeight<T>(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

template vector<double> hitool::VerticalHeight<double>(const param&, double, double, const vector<double>&, int) const;
template vector<float> hitool::VerticalHeight<float>(const param&, float, float, const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeight(const param& wantedParam, const vector<T>& firstLevelValue,
                                 const vector<T>& lastLevelValue, const vector<T>& findValue, int findNth) const
{
	auto modifier = CreateModifier(kFindHeightModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue<T>(modifier, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

template vector<double> hitool::VerticalHeight<double>(const param&, const vector<double>&, const vector<double>&,
                                                       const vector<double>&, int) const;
template vector<float> hitool::VerticalHeight<float>(const param&, const vector<float>&, const vector<float>&,
                                                     const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeightGreaterThan(const param& wantedParam, T lowerHeight, T upperHeight, T findValue,
                                            int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalHeightGreaterThan<T>(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

template vector<double> hitool::VerticalHeightGreaterThan<double>(const param&, double, double, double, int) const;
template vector<float> hitool::VerticalHeightGreaterThan<float>(const param&, float, float, float, int) const;

template <typename T>
vector<T> hitool::VerticalHeightGreaterThan(const params& wantedParamList, T lowerHeight, T upperHeight, T findValue,
                                            int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalHeightGreaterThan<T>(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

template vector<double> hitool::VerticalHeightGreaterThan<double>(const params&, double, double, double, int) const;
template vector<float> hitool::VerticalHeightGreaterThan<float>(const params&, float, float, float, int) const;

template <typename T>
vector<T> hitool::VerticalHeightGreaterThan(const param& wantedParam, T lowerHeight, T upperHeight,
                                            const vector<T>& findValue, int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalHeightGreaterThan<T>(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

template vector<double> hitool::VerticalHeightGreaterThan<double>(const param&, double, double, const vector<double>&,
                                                                  int) const;
template vector<float> hitool::VerticalHeightGreaterThan<float>(const param&, float, float, const vector<float>&,
                                                                int) const;

template <typename T>
vector<T> hitool::VerticalHeightGreaterThan(const vector<param>& wantedParamList, T lowerHeight, T upperHeight,
                                            const vector<T>& findValue, int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalHeightGreaterThan<T>(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

template vector<double> hitool::VerticalHeightGreaterThan<double>(const params&, double, double, const vector<double>&,
                                                                  int) const;
template vector<float> hitool::VerticalHeightGreaterThan<float>(const params&, float, float, const vector<float>&,
                                                                int) const;

template <typename T>
vector<T> hitool::VerticalHeightGreaterThan(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                            const vector<T>& lastLevelValue, const vector<T>& findValue,
                                            int findNth) const
{
	auto modifier = CreateModifier(kFindHeightGreaterThanModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue<T>(modifier, wantedParamList, firstLevelValue, lastLevelValue, findValue);
}

template vector<double> hitool::VerticalHeightGreaterThan<double>(const params&, const vector<double>&,
                                                                  const vector<double>&, const vector<double>&,
                                                                  int) const;
template vector<float> hitool::VerticalHeightGreaterThan<float>(const params&, const vector<float>&,
                                                                const vector<float>&, const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeightGreaterThan(const param& wantedParam, const vector<T>& firstLevelValue,
                                            const vector<T>& lastLevelValue, const vector<T>& findValue,
                                            int findNth) const
{
	auto modifier = CreateModifier(kFindHeightGreaterThanModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue<T>(modifier, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

template vector<double> hitool::VerticalHeightGreaterThan<double>(const param&, const vector<double>&,
                                                                  const vector<double>&, const vector<double>&,
                                                                  int) const;
template vector<float> hitool::VerticalHeightGreaterThan<float>(const param&, const vector<float>&,
                                                                const vector<float>&, const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeightLessThan(const param& wantedParam, T lowerHeight, T upperHeight, T findValue,
                                         int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalHeightLessThan<T>(wantedParam, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

template vector<double> hitool::VerticalHeightLessThan<double>(const param&, double, double, double, int) const;
template vector<float> hitool::VerticalHeightLessThan<float>(const param&, float, float, float, int) const;

template <typename T>
vector<T> hitool::VerticalHeightLessThan(const params& wantedParamList, T lowerHeight, T upperHeight, T findValue,
                                         int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalHeightLessThan<T>(wantedParamList, firstLevelValue, lastLevelValue, findValueVector, findNth);
}

template vector<double> hitool::VerticalHeightLessThan<double>(const params&, double, double, double, int) const;
template vector<float> hitool::VerticalHeightLessThan<float>(const params&, float, float, float, int) const;

template <typename T>
vector<T> hitool::VerticalHeightLessThan(const param& wantedParam, T lowerHeight, T upperHeight,
                                         const vector<T>& findValue, int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalHeightLessThan<T>(wantedParam, firstLevelValue, lastLevelValue, findValue, findNth);
}

template vector<double> hitool::VerticalHeightLessThan<double>(const param&, double, double, const vector<double>&,
                                                               int) const;
template vector<float> hitool::VerticalHeightLessThan<float>(const param&, float, float, const vector<float>&,
                                                             int) const;

template <typename T>
vector<T> hitool::VerticalHeightLessThan(const vector<param>& wantedParamList, T lowerHeight, T upperHeight,
                                         const vector<T>& findValue, int findNth) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalHeightLessThan<T>(wantedParamList, firstLevelValue, lastLevelValue, findValue, findNth);
}

template vector<double> hitool::VerticalHeightLessThan<double>(const params&, double, double, const vector<double>&,
                                                               int) const;
template vector<float> hitool::VerticalHeightLessThan<float>(const params&, float, float, const vector<float>&,
                                                             int) const;

template <typename T>
vector<T> hitool::VerticalHeightLessThan(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                         const vector<T>& lastLevelValue, const vector<T>& findValue, int findNth) const
{
	auto modifier = CreateModifier(kFindHeightLessThanModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue<T>(modifier, wantedParamList, firstLevelValue, lastLevelValue, findValue);
}

template vector<double> hitool::VerticalHeightLessThan<double>(const params&, const vector<double>&,
                                                               const vector<double>&, const vector<double>&, int) const;
template vector<float> hitool::VerticalHeightLessThan<float>(const params&, const vector<float>&, const vector<float>&,
                                                             const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalHeightLessThan(const param& wantedParam, const vector<T>& firstLevelValue,
                                         const vector<T>& lastLevelValue, const vector<T>& findValue, int findNth) const
{
	auto modifier = CreateModifier(kFindHeightLessThanModifier);
	modifier->FindNth(findNth);

	return VerticalExtremeValue<T>(modifier, wantedParam, firstLevelValue, lastLevelValue, findValue);
}

template vector<double> hitool::VerticalHeightLessThan<double>(const param&, const vector<double>&,
                                                               const vector<double>&, const vector<double>&, int) const;
template vector<float> hitool::VerticalHeightLessThan<float>(const param&, const vector<float>&, const vector<float>&,
                                                             const vector<float>&, int) const;

template <typename T>
vector<T> hitool::VerticalMinimum(const vector<param>& wantedParamList, T lowerHeight, T upperHeight) const
{
	ASSERT(!wantedParamList.empty());

	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalMinimum<T>(wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMinimum<double>(const vector<param>&, double, double) const;
template vector<float> hitool::VerticalMinimum<float>(const vector<param>&, float, float) const;

template <typename T>
vector<T> hitool::VerticalMinimum(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                  const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kMinimumModifier), wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMinimum<double>(const vector<param>&, const vector<double>&,
                                                        const vector<double>&) const;
template vector<float> hitool::VerticalMinimum<float>(const vector<param>&, const vector<float>&,
                                                      const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalMinimum(const param& wantedParam, T lowerHeight, T upperHeight) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalExtremeValue<T>(CreateModifier(kMinimumModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMinimum<double>(const param&, double, double) const;
template vector<float> hitool::VerticalMinimum<float>(const param&, float, float) const;

template <typename T>
vector<T> hitool::VerticalMinimum(const param& wantedParam, const vector<T>& firstLevelValue,
                                  const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kMinimumModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMinimum<double>(const param&, const vector<double>&,
                                                        const vector<double>&) const;
template vector<float> hitool::VerticalMinimum<float>(const param&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalMaximum(const vector<param>& wantedParamList, T lowerHeight, T upperHeight) const
{
	ASSERT(!wantedParamList.empty());

	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalMaximum<T>(wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMaximum<double>(const params&, double, double) const;
template vector<float> hitool::VerticalMaximum<float>(const params&, float, float) const;

template <typename T>
vector<T> hitool::VerticalMaximum(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                  const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kMaximumModifier), wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMaximum<double>(const params&, const vector<double>&,
                                                        const vector<double>&) const;
template vector<float> hitool::VerticalMaximum<float>(const params&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalMaximum(const param& wantedParam, T lowerHeight, T upperHeight) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalExtremeValue<T>(CreateModifier(kMaximumModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMaximum<double>(const param&, double, double) const;
template vector<float> hitool::VerticalMaximum<float>(const param&, float, float) const;

template <typename T>
vector<T> hitool::VerticalMaximum(const param& wantedParam, const vector<T>& firstLevelValue,
                                  const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kMaximumModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalMaximum<double>(const param&, const vector<double>&,
                                                        const vector<double>&) const;
template vector<float> hitool::VerticalMaximum<float>(const param&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalAverage(const params& wantedParamList, T lowerHeight, T upperHeight) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalAverage<T>(wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalAverage<double>(const params&, double, double) const;
template vector<float> hitool::VerticalAverage<float>(const params&, float, float) const;

template <typename T>
vector<T> hitool::VerticalAverage(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                  const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kAverageModifier), wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalAverage<double>(const params&, const vector<double>&,
                                                        const vector<double>&) const;
template vector<float> hitool::VerticalAverage<float>(const params&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalAverage(const param& wantedParam, T lowerHeight, T upperHeight) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalExtremeValue<T>(CreateModifier(kAverageModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalAverage<double>(const param&, double, double) const;
template vector<float> hitool::VerticalAverage<float>(const param&, float, float) const;

template <typename T>
vector<T> hitool::VerticalAverage(const param& wantedParam, const vector<T>& firstLevelValue,
                                  const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kAverageModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalAverage<double>(const param&, const vector<double>&,
                                                        const vector<double>&) const;
template vector<float> hitool::VerticalAverage<float>(const param&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalSum(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                              const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kAccumulationModifier), wantedParamList, firstLevelValue,
	                               lastLevelValue);
}

template vector<double> hitool::VerticalSum<double>(const vector<param>&, const vector<double>&,
                                                    const vector<double>&) const;
template vector<float> hitool::VerticalSum<float>(const vector<param>&, const vector<float>&,
                                                  const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalSum(const param& wantedParam, T firstLevelValue, T lastLevelValue) const
{
	vector<T> firstLevelValueVector(itsConfiguration->BaseGrid()->Size(), firstLevelValue);
	vector<T> lastLevelValueVector(itsConfiguration->BaseGrid()->Size(), lastLevelValue);

	return VerticalExtremeValue<T>(CreateModifier(kAccumulationModifier), wantedParam, firstLevelValueVector,
	                               lastLevelValueVector);
}

template vector<double> hitool::VerticalSum<double>(const param&, double, double) const;
template vector<float> hitool::VerticalSum<float>(const param&, float, float) const;

template <typename T>
vector<T> hitool::VerticalSum(const params& wantedParamList, T firstLevelValue, T lastLevelValue) const
{
	vector<T> firstLevelValueVector(itsConfiguration->BaseGrid()->Size(), firstLevelValue);
	vector<T> lastLevelValueVector(itsConfiguration->BaseGrid()->Size(), lastLevelValue);

	return VerticalSum<T>(wantedParamList, firstLevelValueVector, lastLevelValueVector);
}

template vector<double> hitool::VerticalSum<double>(const params&, double, double) const;
template vector<float> hitool::VerticalSum<float>(const params&, float, float) const;

template <typename T>
vector<T> hitool::VerticalSum(const param& wantedParam, const vector<T>& firstLevelValue,
                              const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kAccumulationModifier), wantedParam, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::VerticalSum<double>(const param&, const vector<double>&, const vector<double>&) const;
template vector<float> hitool::VerticalSum<float>(const param&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalCount(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                const vector<T>& lastLevelValue, const vector<T>& findValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kCountModifier), wantedParamList, firstLevelValue, lastLevelValue,
	                               findValue);
}

template vector<double> hitool::VerticalCount<double>(const params&, const vector<double>&, const vector<double>&,
                                                      const vector<double>&) const;
template vector<float> hitool::VerticalCount<float>(const params&, const vector<float>&, const vector<float>&,
                                                    const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalCount(const param& wantedParam, const vector<T>& firstLevelValue,
                                const vector<T>& lastLevelValue, const vector<T>& findValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kCountModifier), wantedParam, firstLevelValue, lastLevelValue,
	                               findValue);
}

template vector<double> hitool::VerticalCount<double>(const param&, const vector<double>&, const vector<double>&,
                                                      const vector<double>&) const;
template vector<float> hitool::VerticalCount<float>(const param&, const vector<float>&, const vector<float>&,
                                                    const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalCount(const param& wantedParam, T firstLevelValue, T lastLevelValue, T findValue) const
{
	vector<T> firstLevelValueVector(itsConfiguration->BaseGrid()->Size(), firstLevelValue);
	vector<T> lastLevelValueVector(itsConfiguration->BaseGrid()->Size(), lastLevelValue);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalExtremeValue<T>(CreateModifier(kCountModifier), wantedParam, firstLevelValueVector,
	                               lastLevelValueVector, findValueVector);
}

template vector<double> hitool::VerticalCount<double>(const param&, double, double, double) const;
template vector<float> hitool::VerticalCount<float>(const param&, float, float, float) const;

template <typename T>
vector<T> hitool::VerticalCount(const params& wantedParamList, T firstLevelValue, T lastLevelValue, T findValue) const
{
	vector<T> firstLevelValueVector(itsConfiguration->BaseGrid()->Size(), firstLevelValue);
	vector<T> lastLevelValueVector(itsConfiguration->BaseGrid()->Size(), lastLevelValue);
	vector<T> findValueVector(itsConfiguration->BaseGrid()->Size(), findValue);

	return VerticalCount<T>(wantedParamList, firstLevelValueVector, lastLevelValueVector, findValueVector);
}

template vector<double> hitool::VerticalCount<double>(const params&, double, double, double) const;
template vector<float> hitool::VerticalCount<float>(const params&, float, float, float) const;

template <typename T>
vector<T> hitool::VerticalValue(const vector<param>& wantedParamList, T wantedHeight) const
{
	ASSERT(!wantedParamList.empty());

	vector<T> heightInfo(itsConfiguration->BaseGrid()->Size(), wantedHeight);

	return VerticalValue<T>(wantedParamList, heightInfo);
}

template vector<double> hitool::VerticalValue<double>(const params&, double) const;
template vector<float> hitool::VerticalValue<float>(const params&, float) const;

template <typename T>
vector<T> hitool::VerticalValue(const vector<param>& wantedParamList, const vector<T>& heightInfo) const
{
	return VerticalExtremeValue<T>(CreateModifier(kFindValueModifier), wantedParamList, vector<T>(), vector<T>(),
	                               heightInfo);
}

template vector<double> hitool::VerticalValue<double>(const params&, const vector<double>&) const;
template vector<float> hitool::VerticalValue<float>(const params&, const vector<float>&) const;

template <typename T>
vector<T> hitool::VerticalValue(const param& wantedParam, T height) const
{
	vector<T> heightInfo(itsConfiguration->BaseGrid()->Size(), height);

	return VerticalExtremeValue<T>(CreateModifier(kFindValueModifier), wantedParam, vector<T>(), vector<T>(),
	                               heightInfo);
}

template vector<double> hitool::VerticalValue<double>(const param&, double) const;
template vector<float> hitool::VerticalValue<float>(const param&, float) const;

template <typename T>
vector<T> hitool::VerticalValue(const param& wantedParam, const vector<T>& heightInfo) const
{
	return VerticalExtremeValue<T>(CreateModifier(kFindValueModifier), wantedParam, vector<T>(), vector<T>(),
	                               heightInfo);
}

template vector<double> hitool::VerticalValue<double>(const param&, const vector<double>&) const;
template vector<float> hitool::VerticalValue<float>(const param&, const vector<float>&) const;

template <typename T>
vector<T> hitool::PlusMinusArea(const params& wantedParamList, T lowerHeight, T upperHeight) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return PlusMinusArea<T>(wantedParamList, firstLevelValue, lastLevelValue);
}

template vector<double> hitool::PlusMinusArea<double>(const params&, double, double) const;
template vector<float> hitool::PlusMinusArea<float>(const params&, float, float) const;

template <typename T>
vector<T> hitool::PlusMinusArea(const vector<param>& wantedParamList, const vector<T>& firstLevelValue,
                                const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kPlusMinusAreaModifier), wantedParamList, firstLevelValue,
	                               lastLevelValue);
}

template vector<double> hitool::PlusMinusArea<double>(const params&, const vector<double>&,
                                                      const vector<double>&) const;
template vector<float> hitool::PlusMinusArea<float>(const params&, const vector<float>&, const vector<float>&) const;

template <typename T>
vector<T> hitool::PlusMinusArea(const param& wantedParam, T lowerHeight, T upperHeight) const
{
	vector<T> firstLevelValue(itsConfiguration->BaseGrid()->Size(), lowerHeight);
	vector<T> lastLevelValue(itsConfiguration->BaseGrid()->Size(), upperHeight);

	return VerticalExtremeValue<T>(CreateModifier(kPlusMinusAreaModifier), wantedParam, firstLevelValue,
	                               lastLevelValue);
}

template vector<double> hitool::PlusMinusArea<double>(const param&, double, double) const;
template vector<float> hitool::PlusMinusArea<float>(const param&, float, float) const;

template <typename T>
vector<T> hitool::PlusMinusArea(const param& wantedParam, const vector<T>& firstLevelValue,
                                const vector<T>& lastLevelValue) const
{
	return VerticalExtremeValue<T>(CreateModifier(kPlusMinusAreaModifier), wantedParam, firstLevelValue,
	                               lastLevelValue);
}

template vector<double> hitool::PlusMinusArea<double>(const param&, const vector<double>&, const vector<double>&) const;
template vector<float> hitool::PlusMinusArea<float>(const param&, const vector<float>&, const vector<float>&) const;

void hitool::Time(const forecast_time& theTime)
{
	itsTime = theTime;
}
void hitool::ForecastType(const forecast_type& theForecastType)
{
	itsForecastType = theForecastType;
}
void hitool::Configuration(shared_ptr<const plugin_configuration> conf)
{
	itsConfiguration = conf;
}

HPParameterUnit hitool::HeightUnit() const
{
	return itsHeightUnit;
}
void hitool::HeightUnit(HPParameterUnit theHeightUnit)
{
	if (theHeightUnit != kM && theHeightUnit != kHPa)
	{
		itsLogger.Error("Invalid height unit: " + to_string(theHeightUnit));
		return;
	}

	itsHeightUnit = theHeightUnit;
}

HPLevelType hitool::LevelType() const
{
	return itsLevelType;
}
void hitool::LevelType(HPLevelType theType)
{
	if (theType != kHybrid && theType != kGeneralizedVerticalLayer)
	{
		itsLogger.Error("Invalid level type: " + HPLevelTypeToString.at(theType));
		return;
	}

	itsLevelType = theType;
}
