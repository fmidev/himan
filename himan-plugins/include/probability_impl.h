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
	return (name == "CL-2-FT" || name == "PRECFORM-N" || name == "PRECFORM2-N");
}

template <typename T>
param_configuration<T> ToParamConfiguration(const partial_param_configuration& partial);

template <>
param_configuration<double> ToParamConfiguration(const partial_param_configuration& partial)
{
	param_configuration<double> pc;

	pc.comparison = partial.comparison;
	pc.output = partial.output;
	pc.parameter = partial.parameter;

	pc.thresholds.reserve(partial.thresholds.size());

	for (const auto& v : partial.thresholds)
	{
		pc.thresholds.push_back(stod(v));
	}

	pc.useGaussianSpread = partial.useGaussianSpread;

	return pc;
}

template <>
param_configuration<std::vector<double>> ToParamConfiguration(const partial_param_configuration& partial)
{
	param_configuration<std::vector<double>> pc;

	pc.comparison = partial.comparison;
	pc.output = partial.output;
	pc.parameter = partial.parameter;

	pc.thresholds.reserve(partial.thresholds.size());
	for (const auto& v : partial.thresholds)
	{
		const auto elems = himan::util::Split(v, ",", false);
		std::vector<double> x;
		for (const auto& vv : elems)
		{
			x.push_back(stod(vv));
		}

		pc.thresholds.push_back(x);
	}

	return pc;
}

template <typename T>
T GetThreshold(std::shared_ptr<himan::info>& targetInfo, const param_configuration<T>& paramConf, bool isGrid)
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

struct EQINCompare : public std::binary_function<double, std::vector<double>, bool>
{
	bool operator()(double a, const std::vector<double>& b) const
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

struct BTWNCompare : public std::binary_function<double, std::vector<double>, bool>
{
	bool operator()(double a, const std::vector<double>& b) const
	{
		ASSERT(b.size() == 2);
		return (a >= b[0] && a < b[1]);
	}
};

template <typename T>
void Probability(std::shared_ptr<himan::info> targetInfo, const param_configuration<T>& paramConf,
                 std::unique_ptr<himan::ensemble>& ens, std::function<bool(double, T)> comp_op)
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
			    std::remove_if(values.begin(), values.end(), [](const double& v) { return himan::IsMissingDouble(v); }),
			    values.end());
		}

		if (values.empty())
		{
			targetInfo->Value(himan::MissingDouble());
			continue;
		}

		const T threshold = GetThreshold<T>(targetInfo, paramConf, isGrid);

		const long int cnt = std::count_if(values.begin(), values.end(), std::bind2nd(comp_op, threshold));
		const double probability = static_cast<double>(cnt) / static_cast<double>(values.size());

		targetInfo->Value(probability);
	}
}

template <typename T>
void ProbabilityWithGaussianSpread(std::shared_ptr<himan::info> targetInfo, const param_configuration<T>& paramConf,
                                   std::unique_ptr<himan::ensemble>& ens)
{
	targetInfo->Find<himan::param>(paramConf.output);
	targetInfo->ResetLocation();
	ens->ResetLocation();

	const bool isGrid = (targetInfo->Grid()->Type() != himan::kPointList);

	while (targetInfo->NextLocation() && ens->NextLocation())
	{
		const double mean = ens->Mean();

		if (himan::IsMissing(mean))
		{
			continue;
		}

		const double stde = sqrt(ens->Variance());
		const T threshold = GetThreshold<T>(targetInfo, paramConf, isGrid);

		const double norm = (threshold - mean) / stde;           // normalize to normal distribution mean=0, stde=1
		double probability = 0.5 * (1 + erf(norm * M_SQRT1_2));  // cdf and error function:
		                                                         // https://www.johndcook.com/erf_and_normal_cdf.pdf
		                                                         // probability is now -∞ -> threshold

		if (paramConf.comparison == comparison_op::GTEQ)
		{
			probability = 1 - probability;
		}

		targetInfo->Value(probability);
	}
}
}  // namespace PROB
