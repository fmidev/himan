#pragma once

#include "ensemble.h"
#include "himan_common.h"
#include "info.h"
#include "probability_core.h"
#include "util.h"

namespace PROB
{
bool AllowMissingValuesInEnsemble(const std::string& name)
{
	return (name == "CEIL-2-M" || name == "PRECFORM-N" || name == "PRECFORM2-N" || name == "CSI-N" || name == "CBTCU-FL");
}

/*
 * Bind a comparison operator to the value(s) it compares to. The value is a single number or,
 * for operators =[] and [], a comma-separated list of numbers.
 */

std::function<bool(float)> MakePredicate(himan::HPProcessingType type, const std::string& value);

std::vector<condition> ToConditions(const std::vector<partial_condition>& partial)
{
	std::vector<condition> ret;
	ret.reserve(partial.size());

	for (const auto& pc : partial)
	{
		ret.push_back({pc.parameter, MakePredicate(pc.comparison, pc.value)});
	}

	return ret;
}

template <typename T>
param_configuration<T> ToParamConfiguration(const partial_param_configuration& partial);

template <>
param_configuration<float> ToParamConfiguration(const partial_param_configuration& partial)
{
	param_configuration<float> pc;

	pc.output = partial.output;
	pc.parameter = partial.parameter;

	pc.thresholds.reserve(partial.thresholds.size());

	for (const auto& v : partial.thresholds)
	{
		pc.thresholds.push_back(stof(v));
	}

	pc.useGaussianSpread = partial.useGaussianSpread;
	pc.conditions = ToConditions(partial.conditions);

	return pc;
}

template <>
param_configuration<std::vector<float>> ToParamConfiguration(const partial_param_configuration& partial)
{
	param_configuration<std::vector<float>> pc;

	pc.output = partial.output;
	pc.parameter = partial.parameter;

	pc.thresholds.reserve(partial.thresholds.size());
	for (const auto& v : partial.thresholds)
	{
		const auto elems = himan::util::Split(v, ",");
		std::vector<float> x;
		for (const auto& vv : elems)
		{
			x.push_back(stof(vv));
		}

		pc.thresholds.push_back(x);
	}

	pc.useGaussianSpread = partial.useGaussianSpread;
	pc.conditions = ToConditions(partial.conditions);

	return pc;
}

template <typename T>
T GetThreshold(std::shared_ptr<himan::info<float>>& targetInfo, const param_configuration<T>& paramConf, bool isGrid)
{
	if (isGrid)
	{
		return paramConf.thresholds[0];
	}
	else
	{
		return paramConf.thresholds[targetInfo->LocationIndex()];
	}
}

/*
 * struct EQINCompare
 *
 * This struct implements the comparison operator for EQIN, i.e. check if a data value
 * is in a set of values.
 */

struct EQINCompare : public std::binary_function<float, std::vector<float>, bool>
{
	bool operator()(float a, const std::vector<float>& b) const
	{
		return std::find(b.begin(), b.end(), a) != b.end();
	}
};

/*
 * struct BTWNCompare
 *
 * This struct implements the "in between" operator for BETW, i.e. check if a data value
 * is within a range bounded by [lower value,upper value]
 */

struct BTWNCompare : public std::binary_function<float, std::vector<float>, bool>
{
	bool operator()(float a, const std::vector<float>& b) const
	{
		ASSERT(b.size() == 2);
		return (a >= b[0] && a < b[1]);
	}
};

std::function<bool(float)> MakePredicate(himan::HPProcessingType type, const std::string& value)
{
	std::vector<float> values;

	for (const auto& v : himan::util::Split(value, ","))
	{
		values.push_back(std::stof(v));
	}

	if (values.empty())
	{
		throw std::runtime_error("Condition has no value");
	}

	const float a = values[0];

	switch (type)
	{
		case himan::kProbabilityLessThan:
			return [a](float x) { return x < a; };
		case himan::kProbabilityLessThanOrEqual:
			return [a](float x) { return x <= a; };
		case himan::kProbabilityGreaterThan:
			return [a](float x) { return x > a; };
		case himan::kProbabilityGreaterThanOrEqual:
			return [a](float x) { return x >= a; };
		case himan::kProbabilityEquals:
			return [a](float x) { return x == a; };
		case himan::kProbabilityNotEquals:
			return [a](float x) { return x != a; };
		case himan::kProbabilityEqualsIn:
			return [values](float x) { return EQINCompare()(x, values); };
		case himan::kProbabilityBetween:
			if (values.size() != 2)
			{
				throw std::runtime_error("Condition operator '[]' requires two values");
			}
			return [values](float x) { return BTWNCompare()(x, values); };
		default:
			throw std::runtime_error("Unsupported comparison operator in condition");
	}
}

/*
 * JointProbability()
 *
 * Probability that an ensemble member fulfills all of the rules. The rules are evaluated for the
 * same member of each ensemble, so the rules must have an equal number of values.
 *
 * Missing values are handled separately for each rule, similarly to Probability(): if the rule allows
 * missing values (see AllowMissingValuesInEnsemble()), a missing value means that the rule is not
 * fulfilled, otherwise the member is left out of the calculation. If no members are left, the
 * probability is missing.
 */

float JointProbability(const std::vector<rule_input>& rules)
{
	if (rules.empty())
	{
		throw std::invalid_argument("JointProbability: no rules");
	}

	const size_t memberCount = rules[0].values.size();

	for (const auto& rule : rules)
	{
		if (rule.values.size() != memberCount)
		{
			throw std::invalid_argument("JointProbability: rules have different number of ensemble members");
		}
	}

	long int fulfilled = 0;
	long int total = 0;

	for (size_t i = 0; i < memberCount; i++)
	{
		bool include = true;
		bool match = true;

		for (const auto& rule : rules)
		{
			const float value = rule.values[i];

			if (himan::IsMissing(value))
			{
				include = include && rule.allowMissing;
				match = false;
			}
			else
			{
				match = match && rule.compare(value);
			}
		}

		if (include)
		{
			total++;
			fulfilled += match ? 1 : 0;
		}
	}

	if (total == 0)
	{
		return himan::MissingFloat();
	}

	return static_cast<float>(fulfilled) / static_cast<float>(total);
}

/*
 * ProbabilityWithConditions()
 *
 * Probability that an ensemble member fulfills both the primary comparison (paramConf.parameter
 * compared to the threshold with comp_op) and all of the additional conditions. The conditions
 * are evaluated for the same member of their own ensembles, so the ensembles must be of equal size.
 */

template <typename T>
void ProbabilityWithConditions(std::shared_ptr<himan::info<float>> targetInfo, const param_configuration<T>& paramConf,
                               std::unique_ptr<himan::ensemble>& ens,
                               std::vector<std::unique_ptr<himan::ensemble>>& conditionEns,
                               std::function<bool(float, T)> comp_op)
{
	ASSERT(paramConf.conditions.size() == conditionEns.size());

	targetInfo->Find<himan::param>(paramConf.output);
	targetInfo->ResetLocation();
	ens->ResetLocation();

	for (auto& e : conditionEns)
	{
		e->ResetLocation();
	}

	const bool isGrid = (targetInfo->Grid()->Type() != himan::kPointList);

	const bool allowMissing = AllowMissingValuesInEnsemble(ens->Param().Name());

	std::vector<bool> allowMissingCondition;

	for (const auto& c : paramConf.conditions)
	{
		allowMissingCondition.push_back(AllowMissingValuesInEnsemble(c.parameter.Name()));
	}

	auto nextLocations = [&]()
	{
		bool ok = targetInfo->NextLocation() && ens->NextLocation();

		// all must be advanced, therefore no short circuit evaluation
		for (auto& e : conditionEns)
		{
			ok = e->NextLocation() && ok;
		}

		return ok;
	};

	while (nextLocations())
	{
		const T threshold = GetThreshold<T>(targetInfo, paramConf, isGrid);

		std::vector<rule_input> rules;
		rules.reserve(1 + conditionEns.size());

		rules.push_back({ens->Values(), [&](float x) { return comp_op(x, threshold); }, allowMissing});

		for (size_t i = 0; i < conditionEns.size(); i++)
		{
			rules.push_back({conditionEns[i]->Values(), paramConf.conditions[i].compare, allowMissingCondition[i]});
		}

		targetInfo->Value(JointProbability(rules));
	}
}

template <typename T>
void Probability(std::shared_ptr<himan::info<float>> targetInfo, const param_configuration<T>& paramConf,
                 std::unique_ptr<himan::ensemble>& ens, std::function<bool(float, T)> comp_op)
{
	targetInfo->Find<himan::param>(paramConf.output);
	targetInfo->ResetLocation();
	ens->ResetLocation();

	const bool isGrid = (targetInfo->Grid()->Type() != himan::kPointList);

	// HIMAN-216: allow missing values in ensemble for some parameters
	const bool allowMissing = AllowMissingValuesInEnsemble(ens->Param().Name());

	while (targetInfo->NextLocation() && ens->NextLocation())
	{
		auto values = ens->Values();

		// HIMAN-184: if ensemble has no values, or all values are missing, the resulting probability should
		// be missing

		if (allowMissing == false)
		{
			values.erase(
			    std::remove_if(values.begin(), values.end(), [](const float& v) { return himan::IsMissing(v); }),
			    values.end());
		}

		if (values.empty())
		{
			targetInfo->Value(himan::MissingFloat());
			continue;
		}

		const T threshold = GetThreshold<T>(targetInfo, paramConf, isGrid);

		const long int cnt = std::count_if(values.begin(), values.end(), std::bind2nd(comp_op, threshold));
		const float probability = static_cast<float>(cnt) / static_cast<float>(values.size());

		targetInfo->Value(probability);
	}
}

template <typename T>
void ProbabilityWithGaussianSpread(std::shared_ptr<himan::info<T>> targetInfo, const param_configuration<T>& paramConf,
                                   std::unique_ptr<himan::ensemble>& ens)
{
	targetInfo->template Find<himan::param>(paramConf.output);
	targetInfo->ResetLocation();
	ens->ResetLocation();

	const bool isGrid = (targetInfo->Grid()->Type() != himan::kPointList);

	while (targetInfo->NextLocation() && ens->NextLocation())
	{
		const float mean = ens->Mean();

		if (himan::IsMissing(mean))
		{
			continue;
		}

		const float stde = sqrtf(ens->Variance());
		const T threshold = GetThreshold<T>(targetInfo, paramConf, isGrid);

		const float norm = (threshold - mean) / stde;  // normalize to normal distribution mean=0, stde=1
		float probability =
		    0.5f *
		    (1 + erff(norm * static_cast<float>(M_SQRT1_2)));  // cdf and error function:
		                                                       // https://www.johndcook.com/erf_and_normal_cdf.pdf
		                                                       // probability is now -∞ -> threshold

		if (paramConf.output.ProcessingType().Type() == himan::kProbabilityGreaterThan ||
		    paramConf.output.ProcessingType().Type() == himan::kProbabilityGreaterThanOrEqual)
		{
			probability = 1 - probability;
		}

		targetInfo->Value(probability);
	}
}
}  // namespace PROB
