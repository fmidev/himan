#include "named_ensemble.h"
#include "logger.h"
#include "util.h"
#include <map>

using namespace himan;

namespace
{
using member_list = std::vector<std::pair<forecast_type, time_duration>>;

// Perturbed members first..last, optionally preceded by the control member
member_list Perturbed(int first, int last, bool withControl = false)
{
	member_list m;

	if (withControl)
	{
		m.emplace_back(forecast_type(kEpsControl, 0), time_duration("00:00:00"));
	}

	for (int i = first; i <= last; i++)
	{
		m.emplace_back(forecast_type(kEpsPerturbation, static_cast<float>(i)), time_duration("00:00:00"));
	}

	return m;
}

// Repeat the members of base for each additional lag, lag being added to the member's own lag
member_list Lagged(const member_list& base, const std::vector<time_duration>& additionalLags)
{
	member_list m = base;

	for (const auto& lag : additionalLags)
	{
		for (const auto& p : base)
		{
			m.emplace_back(p.first, p.second + lag);
		}
	}

	return m;
}

member_list MepsMembers()
{
	// MEPS member distribution as of 2020-02-28:
	// Operational member distribution
	// Time (UTC) 	CIRRUS 	STRATUS VOIMA
	// 00,03,…,21 	0 	1,2,12 	9
	// 01,04,…,22 	7 	3,4,13 	10
	// 02,05,…,23 	8,14 	5,6 	11
	//
	// Here we define a single MEPS ensemble to consist
	// of the control member cycle and two cycles *preceding it*.
	//
	// So ensemble where control member is produced at 00 cycle
	// include cycles 23 and 22.

	// TODO: think if this configuration could be stored outside code,
	// for example in database

	// clang-format off

	return member_list({
	     {forecast_type(kEpsControl, 0), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 1), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 2), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 3), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 4), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 5), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 6), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 7), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 8), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 9), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 10), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 11), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 12), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 13), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 14), time_duration("-01:00:00")}
	});

	// clang-format on
}

const std::map<std::string, named_ensemble_definition>& Registry()
{
	static const std::map<std::string, named_ensemble_definition> registry = []() {
		const auto meps = MepsMembers();
		const auto aila = Perturbed(1, 10);

		return std::map<std::string, named_ensemble_definition>{
		    {"ECMWF50", {kPerturbedEnsemble, Perturbed(1, 50)}},
		    {"ECMWF51", {kPerturbedEnsemble, Perturbed(1, 50, true)}},
		    {"AILA10", {kPerturbedEnsemble, aila}},
		    {"AILA10_LAGGED", {kLaggedEnsemble, Lagged(aila, {time_duration("-06:00:00")})}},
		    {"MEPS_SINGLE_ENSEMBLE", {kLaggedEnsemble, meps}},
		    {"MEPS_LAGGED_ENSEMBLE", {kLaggedEnsemble, Lagged(meps, {time_duration("-03:00:00")})}}};
	}();

	return registry;
}
}  // namespace

namespace himan
{
const named_ensemble_definition& GetNamedEnsemble(const std::string& name)
{
	const auto& registry = Registry();
	const auto it = registry.find(name);

	if (it == registry.end())
	{
		std::string allowed;
		for (const auto& p : registry)
		{
			allowed += (allowed.empty() ? "" : ",") + p.first;
		}

		logger log("named_ensemble");
		log.Fatal(fmt::format("Unknown named ensemble: '{}', allowed values are: {}", name, allowed));
		himan::Abort();
	}

	return it->second;
}
}  // namespace himan
