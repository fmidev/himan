#include "csv.h"
#include "logger_factory.h"
#include "point_list.h"
#include "util.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>

using namespace std;
using namespace himan::plugin;

csv::csv() { itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("csv")); }
bool csv::ToFile(info& theInfo, string& theOutputFile)
{
	if (theInfo.Grid()->Class() != kIrregularGrid)
	{
		itsLogger->Error("Only irregular grids can be written to CSV");
		return false;
	}

	auto aTimer = timer_factory::Instance()->GetTimer();
	aTimer->Start();

	ofstream out(theOutputFile);

	assert(out.is_open());

	out << "station_id,station_name,longitude,latitude,origintime,forecasttime,level_name,level_value,parameter_name,"
	       "value"
	    << endl;

	theInfo.First();
	theInfo.ResetParam();

	const auto originTime = theInfo.Time().OriginDateTime().String();
	while (theInfo.Next())
	{
		for (theInfo.ResetLocation(); theInfo.NextLocation();)
		{
			station s = theInfo.Station();
			out << s.Id() << ","  // << s.X() << "," << s.Y() << ","
			    << originTime << "," << theInfo.Time().ValidDateTime().String() << ","
			    << HPLevelTypeToString.at(theInfo.Level().Type()) << "," << theInfo.Level().Value() << ","
			    << theInfo.Param().Name() << "," << theInfo.Value() << endl;
		}

		out.flush();
	}

	aTimer->Stop();

	double duration = static_cast<double>(aTimer->GetTime());
	double bytes = static_cast<double>(boost::filesystem::file_size(theOutputFile));

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));
	itsLogger->Info("Wrote file '" + theOutputFile + "' (" + boost::lexical_cast<string>(speed) + " MB/s)");

	return true;
}

shared_ptr<himan::info> csv::FromFile(const string& inputFile, const search_options& options,
                                      bool readIfNotMatching) const
{
	info_t all, requested;

	vector<string> lines;
	string line;

	ifstream infile(inputFile);

	while (getline(infile, line))
	{
		lines.push_back(line);
	}

	all = util::CSVToInfo(lines);

	if (readIfNotMatching)
	{
		return all;
	}

	vector<forecast_time> times;
	vector<param> params;
	vector<level> levels;
	vector<station> stations;
	vector<forecast_type> ftypes;

	forecast_time optsTime(options.time);

	all->First();
	all->ResetParam();

	// Remove those dimensions that are not requested
	while (all->Next())
	{
		if (all->Param() != options.param)
		{
			itsLogger->Debug("Param does not match");
			itsLogger->Debug(options.param.Name() + " vs " + all->Param().Name());
		}
		else if (find(params.begin(), params.end(), all->Param()) == params.end())
		{
			params.push_back(all->Param());
		}

		if (all->Level() != options.level)
		{
			itsLogger->Debug("Level does not match");
			itsLogger->Debug(static_cast<string>(options.level) + " vs " + static_cast<string>(all->Level()));
		}
		else if (find(levels.begin(), levels.end(), all->Level()) == levels.end())
		{
			levels.push_back(all->Level());
		}

		if (all->Time() != options.time)
		{
			itsLogger->Debug("Time does not match");
			itsLogger->Debug("Origin time " + static_cast<string>(optsTime.OriginDateTime()) + " vs " +
			                 static_cast<string>(all->Time().OriginDateTime()));
			itsLogger->Debug("Forecast time: " + static_cast<string>(optsTime.ValidDateTime()) + " vs " +
			                 static_cast<string>(all->Time().ValidDateTime()));
		}
		else if (find(times.begin(), times.end(), all->Time()) == times.end())
		{
			times.push_back(all->Time());
		}

		if (all->ForecastType() != options.ftype)
		{
			itsLogger->Debug("Forecast type does not match");
			itsLogger->Debug(static_cast<string>(options.ftype) + " vs " + static_cast<string>(all->ForecastType()));
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
		itsLogger->Error("Did not find valid data from file '" + inputFile + "'");
		throw kFileDataNotFound;
	}

	requested = make_shared<info>();
	requested->Producer(options.prod);

	requested->Times(times);
	requested->Params(params);
	requested->Levels(levels);
	requested->ForecastTypes(ftypes);

	auto base = unique_ptr<grid>(new point_list());  // placeholder

	requested->Create(base.get(), true);
	requested->First();
	requested->ResetParam();

	while (requested->Next())
	{
		dynamic_cast<point_list*>(requested->Grid())->Stations(stations);
	}

	itsLogger->Debug("Read " + boost::lexical_cast<string>(times.size()) + " times, " +
	                 boost::lexical_cast<string>(levels.size()) + " levels, " +
	                 boost::lexical_cast<string>(ftypes.size()) + " forecast types and " +
	                 boost::lexical_cast<string>(params.size()) + " params from file '" + inputFile + "'");

	requested->First();
	requested->ResetParam();

	while (requested->Next())
	{
		if (!all->Param(requested->Param())) throw runtime_error("Impossible error occurred");
		if (!all->Level(requested->Level())) throw runtime_error("Impossible error occurred");
		if (!all->Time(requested->Time())) throw runtime_error("Impossible error occurred");
		if (!all->ForecastType(requested->ForecastType())) throw runtime_error("Impossible error occurred");

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
