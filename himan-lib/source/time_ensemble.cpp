#include "time_ensemble.h"

#include "plugin_factory.h"

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace himan::plugin;

time_ensemble::time_ensemble(const param& parameter, size_t ensembleSize, HPTimeResolution theTimeSpan)
    : ensemble(parameter, (ensembleSize)), itsTimeSpan(theTimeSpan)
{
}

void time_ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                          const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	try
	{
		forecast_time ftime(time);

		for (size_t i = 0; i < itsPerturbations.size() + 1; i++)
		{
			itsForecasts[i] = f->Fetch(config, ftime, forecastLevel, itsParam);

			ftime.OriginDateTime().Adjust(itsTimeSpan, -1);
			ftime.ValidDateTime().Adjust(itsTimeSpan, -1);
		}
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw std::runtime_error("Ensemble: unable to proceed");
		}
		else
		{
			// NOTE let the plugin decide what to do with missing data
			throw;
		}
	}
}