#include "csv.h"
#include "logger.h"
#include "point_list.h"
#include "util.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>

using namespace std;
using namespace himan::plugin;

csv::csv()
{
	itsLogger = logger("csv");
}

bool csv::ToFile(info<double>& theInfo, string& theOutputFile)
{
	return ToFile<double>(theInfo, theOutputFile);
}

template <typename T>
bool csv::ToFile(info<T>& theInfo, string& theOutputFile)
{
	if (theInfo.Grid()->Class() != kIrregularGrid)
	{
		itsLogger.Error("Only irregular grids can be written to CSV");
		return false;
	}

	timer aTimer;
	aTimer.Start();

	ofstream out(theOutputFile);

	ASSERT(out.is_open());

	out << "#producer_id,origintime,station_id,station_name,longitude,latitude,param_name,level_name,level_value,level_"
	       "value2,"
	       "forecast_period,forecast_type_id,forecast_type_value,value"
	    << endl;

	theInfo.First();
	theInfo.template Reset<param>();

	const auto originTime = theInfo.Time().OriginDateTime().String();
	while (theInfo.Next())
	{
		for (theInfo.ResetLocation(); theInfo.NextLocation();)
		{
			station s = theInfo.Station();

			// If station has some missing elements, skip them in CSV output
			const string stationId = (s.Id() != kHPMissingInt) ? "" : to_string(s.Id());
			const string stationName = (s.Name() != "Himan default station") ? "" : s.Name();

			// boost cast handles floats more elegantly
			const string lon = (IsKHPMissingValue(s.X())) ? "" : to_string(s.X());
			const string lat = (IsKHPMissingValue(s.Y())) ? "" : to_string(s.Y());

			out << theInfo.Producer().Id() << "," << originTime << "," << stationId << "," << stationName << "," << lon
			    << "," << lat << "," << theInfo.Param().Name() << "," << HPLevelTypeToString.at(theInfo.Level().Type())
			    << "," << theInfo.Level().Value() << "," << theInfo.Level().Value2() << ","
			    << util::MakeSQLInterval(theInfo.Time()) << "," << theInfo.ForecastType().Type() << ","
			    << theInfo.ForecastType().Value() << "," << theInfo.Value() << endl;
		}

		out.flush();
	}

	aTimer.Stop();

	double duration = static_cast<double>(aTimer.GetTime());
	double bytes = static_cast<double>(boost::filesystem::file_size(theOutputFile));

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));
	itsLogger.Info("Wrote file '" + theOutputFile + "' (" + to_string(speed) + " MB/s)");

	return true;
}

template bool csv::ToFile<double>(info<double>&, string&);
template bool csv::ToFile<float>(info<float>&, string&);

shared_ptr<himan::info<double>> csv::FromFile(const string& inputFile, const search_options& options,
                                              bool readIfNotMatching) const
{
	return FromFile<double>(inputFile, options, readIfNotMatching);
}

template <typename T>
shared_ptr<himan::info<T>> csv::FromFile(const string& inputFile, const search_options& options,
                                         bool readIfNotMatching) const
{
	shared_ptr<info<T>> all, requested;

	vector<string> lines;
	string line;

	ifstream infile(inputFile);

	while (getline(infile, line))
	{
		lines.push_back(line);
	}

	all = util::CSVToInfo<T>(lines);

	if (readIfNotMatching)
	{
		// CSV file does not have producer information attached.
		// We just have to trust that it came from the producer that was requested.

		all->First();
		all->template Reset<param>();

		while (all->Next())
		{
			all->Producer(options.prod);
		}

		return all;
	}

	vector<forecast_time> times;
	vector<param> params;
	vector<level> levels;
	vector<station> stations;
	vector<forecast_type> ftypes;

	forecast_time optsTime(options.time);

	all->First();
	all->template Reset<param>();

	// Remove those dimensions that are not requested
	while (all->Next())
	{
		if (all->Param() != options.param)
		{
			itsLogger.Debug("Param does not match");
			itsLogger.Debug(options.param.Name() + " vs " + all->Param().Name());
		}
		else if (find(params.begin(), params.end(), all->Param()) == params.end())
		{
			params.push_back(all->Param());
		}

		if (all->Level() != options.level)
		{
			itsLogger.Debug("Level does not match");
			itsLogger.Debug(static_cast<string>(options.level) + " vs " + static_cast<string>(all->Level()));
		}
		else if (find(levels.begin(), levels.end(), all->Level()) == levels.end())
		{
			levels.push_back(all->Level());
		}

		if (all->Time() != options.time)
		{
			itsLogger.Debug("Time does not match");
			itsLogger.Debug("Origin time " + static_cast<string>(optsTime.OriginDateTime()) + " vs " +
			                static_cast<string>(all->Time().OriginDateTime()));
			itsLogger.Debug("Forecast time: " + static_cast<string>(optsTime.ValidDateTime()) + " vs " +
			                static_cast<string>(all->Time().ValidDateTime()));
		}
		else if (find(times.begin(), times.end(), all->Time()) == times.end())
		{
			times.push_back(all->Time());
		}

		if (all->ForecastType() != options.ftype)
		{
			itsLogger.Debug("Forecast type does not match");
			itsLogger.Debug(static_cast<string>(options.ftype) + " vs " + static_cast<string>(all->ForecastType()));
		}
		else if (find(ftypes.begin(), ftypes.end(), all->ForecastType()) == ftypes.end())
		{
			ftypes.push_back(all->ForecastType());
		}

		for (all->ResetLocation(); all->NextLocation();)
		{
			// no station in options struct ...
			if (find(stations.begin(), stations.end(), all->Station()) == stations.end())
			{
				stations.push_back(all->Station());
			}
		}
	}

	if (times.size() == 0 || params.size() == 0 || levels.size() == 0 || ftypes.size() == 0)
	{
		itsLogger.Error("Did not find valid data from file '" + inputFile + "'");
		throw kFileDataNotFound;
	}

	requested = make_shared<info<T>>();
	requested->Producer(options.prod);

	requested->template Set<forecast_time>(times);
	requested->template Set<param>(params);
	requested->template Set<level>(levels);
	requested->template Set<forecast_type>(ftypes);

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(new point_list());  // placeholder

	requested->Create(b, true);
	requested->First();
	requested->template Reset<param>();

	while (requested->Next())
	{
		dynamic_pointer_cast<point_list>(requested->Grid())->Stations(stations);
	}

	itsLogger.Debug("Read " + to_string(times.size()) + " times, " + to_string(levels.size()) + " levels, " +
	                to_string(ftypes.size()) + " forecast types and " + to_string(params.size()) +
	                " params from file '" + inputFile + "'");

	requested->First();
	requested->template Reset<param>();

	while (requested->Next())
	{
		if (!all->template Find<param>(requested->Param()))
			throw runtime_error("Impossible error occurred");
		if (!all->template Find<level>(requested->Level()))
			throw runtime_error("Impossible error occurred");
		if (!all->template Find<forecast_time>(requested->Time()))
			throw runtime_error("Impossible error occurred");
		if (!all->template Find<forecast_type>(requested->ForecastType()))
			throw runtime_error("Impossible error occurred");

		for (requested->ResetLocation(); requested->NextLocation();)
		{
			for (all->ResetLocation(); all->NextLocation();)
			{
				if (requested->Station() == all->Station())
				{
					requested->Value(all->Value());
					break;
				}
			}
		}
	}

	return requested;
}

template shared_ptr<himan::info<double>> csv::FromFile<double>(const string&, const search_options&, bool) const;
template shared_ptr<himan::info<float>> csv::FromFile<float>(const string&, const search_options&, bool) const;
