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
	return (name == "CL-2-FT" || name == "PRECFORM-N" || name == "PRECFORM2-N" || name == "CSI-N");
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
		const auto elems = himan::util::Split(v, ",", false);
		std::vector<float> x;
		for (const auto& vv : elems)
		{
			x.push_back(stof(vv));
		}

		pc.thresholds.push_back(x);
	}

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

		if (paramConf.output.ProcessingType().Type() == himan::kProbabilityGreaterThan)
		{
			probability = 1 - probability;
		}

		targetInfo->Value(probability);
	}
}
}  // namespace PROB
