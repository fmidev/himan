#pragma once

#include "ensemble.h"
#include "time_duration.h"
#include <string>
#include <utility>
#include <vector>

namespace himan
{
/// @brief Definition of a named ensemble: its type and its members. For ensembles that are
/// not lagged the lag of each member is zero.
struct named_ensemble_definition
{
	HPEnsembleType type;
	std::vector<std::pair<forecast_type, time_duration>> members;
};

/// @brief Return the definition of a named ensemble. Aborts if the name is unknown.
const named_ensemble_definition& GetNamedEnsemble(const std::string& name);
}  // namespace himan
