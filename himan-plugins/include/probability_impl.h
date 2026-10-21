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
 * ProbabilityWithConditions()
 *
 * Probability that an ensemble member fulfills both the primary comparison (paramConf.parameter
 * compared to the threshold with comp_op) and all of the additional conditions. The conditions
 * are evaluated for the same member of their own ensembles, so the ensembles must be of equal size.
 *
 * Missing values are handled separately for each input parameter, similarly to Probability():
 * if the parameter allows missing values (see AllowMissingValuesInEnsemble()), a missing value
 * means that the comparison is not fulfilled, otherwise the member is left out of the calculation.
 * If no members are left, the probability is missing.
 */

template <typename T>
void ProbabilityWithConditions(std::shared_ptr<himan::info<float>> targetInfo,
                               const param_configuration<T>& paramConf, std::unique_ptr<himan::ensemble>& ens,
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

	// Missing value handling is done separately for each input parameter, see AllowMissingValuesInEnsemble()
	const bool allowMissing = AllowMissingValuesInEnsemble(ens->Param().Name());

	std::vector<bool> allowMissingCondition;

	for (const auto& c : paramConf.conditions)
	{
		allowMissingCondition.push_back(AllowMissingValuesInEnsemble(c.parameter.Name()));
	}

	auto nextLocations = [&]() {
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
		const auto values = ens->Values();

		std::vector<std::vector<float>> conditionValues;

		for (const auto& e : conditionEns)
		{
			conditionValues.push_back(e->Values());
		}

		const T threshold = GetThreshold<T>(targetInfo, paramConf, isGrid);

		long int cnt = 0;
		long int total = 0;

		for (size_t i = 0; i < values.size(); i++)
		{
			bool include = true;
			bool fulfilled = true;

			if (himan::IsMissing(values[i]))
			{
				include = include && allowMissing;
				fulfilled = false;
			}
			else
			{
				fulfilled = comp_op(values[i], threshold);
			}

			for (size_t j = 0; j < conditionValues.size(); j++)
			{
				const float v = conditionValues[j][i];

				if (himan::IsMissing(v))
				{
					include = include && allowMissingCondition[j];
					fulfilled = false;
				}
				else
				{
					fulfilled = fulfilled && paramConf.conditions[j].compare(v);
				}
			}

			if (include)
			{
				total++;
				cnt += fulfilled ? 1 : 0;
			}
		}

		if (total == 0)
		{
			targetInfo->Value(himan::MissingFloat());
			continue;
		}

		targetInfo->Value(static_cast<float>(cnt) / static_cast<float>(total));
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
