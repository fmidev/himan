/**
 * @file csv.cpp
 *
 * @date Nov 27, 2012
 * @author: partio
 */

#include "csv.h"
#include "csv_v3.h"
#include "logger_factory.h"
#include "point_list.h"
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <fstream>

using namespace std;
using namespace himan::plugin;

typedef tuple<int,     // station id
              string,  // station name
              double,  // longitude
              double,  // latitude
              string,  // origintime as timestamp
              string,  // forecasttime as timestamp
              string,  // level name
              double,  // level value
              string,  // parameter name
              double>  // value
    record;

typedef io::CSVReader<10,  // column count
                      io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow, io::no_comment>
    csv_reader;

bool GetLine(csv_reader& in, record& line)
{
	int station_id;
	string station_name;
	double longitude;
	double latitude;
	string origintime;
	string forecasttime;
	string level_name;
	double level_value;
	string parameter_name;
	double value;

	if (in.read_row(station_id, station_name, longitude, latitude, origintime, forecasttime, level_name, level_value,
	                parameter_name, value))
	{
		line = make_tuple(station_id, station_name, longitude, latitude, origintime, forecasttime, level_name,
		                  level_value, parameter_name, value);
		return true;
	}

	return false;
}

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

	for (theInfo.ResetTime(); theInfo.NextTime();)
	{
		forecast_time time = theInfo.Time();

		for (theInfo.ResetLevel(); theInfo.NextLevel();)
		{
			level lev = theInfo.Level();

			for (theInfo.ResetParam(); theInfo.NextParam();)
			{
				param par = theInfo.Param();

				for (theInfo.ResetLocation(); theInfo.NextLocation();)
				{
					station s = theInfo.Station();
					out << s.Id() << "," << s.Name() << "," << s.X() << "," << s.Y() << ","
					    << time.OriginDateTime().String() << "," << time.ValidDateTime().String() << ","
					    << HPLevelTypeToString.at(lev.Type()) << "," << lev.Value() << "," << par.Name() << ","
					    << theInfo.Value() << endl;
				}

				out.flush();
			}
		}
	}

	aTimer->Stop();

	double duration = static_cast<double>(aTimer->GetTime());
	double bytes = static_cast<double>(boost::filesystem::file_size(theOutputFile));

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));
	itsLogger->Info("Wrote file '" + theOutputFile + "' (" + boost::lexical_cast<string>(speed) + " MB/s)");

	return true;
}

shared_ptr<himan::info> csv::FromFile(const string& inputFile, const search_options& options) const
{
	info_t ret = make_shared<info>();

	csv_reader in(inputFile);

	in.read_header(io::ignore_no_column, "station_id", "station_name", "longitude", "latitude", "origintime",
	               "forecasttime", "level_name", "level_value", "parameter_name", "value");

	if (!in.has_column("station_id"))
	{
		itsLogger->Error("CSV file does not have column station_id");
		throw kFileDataNotFound;
	}

	record line;

	vector<forecast_time> times;
	vector<param> params;
	vector<level> levels;
	vector<station> stats;

	forecast_time optsTime(options.time);

	// First create descriptors
	while (GetLine(in, line))
	{
		/* Validate time */

		forecast_time f(get<4>(line), get<5>(line));

		if (f != options.time)
		{
			itsLogger->Debug("Time does not match");
			itsLogger->Debug("Origin time " + static_cast<string>(optsTime.OriginDateTime()) + " vs " +
			                 static_cast<string>(f.OriginDateTime()));
			itsLogger->Debug("Forecast time: " + static_cast<string>(optsTime.ValidDateTime()) + " vs " +
			                 static_cast<string>(f.ValidDateTime()));

			continue;
		}

		string level_name = get<6>(line);
		boost::algorithm::to_lower(level_name);

		/* Validate level */
		level l(HPStringToLevelType.at(level_name), get<7>(line));

		if (l != options.level)
		{
			itsLogger->Debug("Level does not match");
			itsLogger->Debug(static_cast<string>(options.level) + " vs " + static_cast<string>(l));

			continue;
		}

		/* Validate param */

		param p(get<8>(line));

		if (p != options.param)
		{
			itsLogger->Debug("Param does not match");
			itsLogger->Debug(options.param.Name() + " vs " + p.Name());

			continue;
		}

		bool found = false;

		/* Prevent duplicates */

		BOOST_FOREACH (const forecast_time& t, times)
		{
			if (f == t)
			{
				found = true;
				break;
			}
		}

		if (!found) times.push_back(f);

		found = false;

		BOOST_FOREACH (const level& t, levels)
		{
			if (l == t)
			{
				found = true;
				break;
			}
		}

		if (!found) levels.push_back(l);

		found = false;

		BOOST_FOREACH (const param& t, params)
		{
			if (p == t)
			{
				found = true;
				break;
			}
		}

		if (!found) params.push_back(p);

		// Add location information

		station s(get<0>(line), get<1>(line), get<2>(line), get<3>(line));

		found = false;

		BOOST_FOREACH (const station& stat, stats)
		{
			if (stat == s)
			{
				found = true;
				break;
			}
		}

		if (!found) stats.push_back(s);
	}

	if (times.size() == 0 || params.size() == 0 || levels.size() == 0)
	{
		itsLogger->Error("Did not find valid data from file '" + inputFile + "'");
		throw kFileDataNotFound;
	}

	assert(times.size());
	assert(params.size());
	assert(levels.size());

	ret->Times(times);
	ret->Params(params);
	ret->Levels(levels);

	vector<forecast_type> ftypes;
	ftypes.push_back(forecast_type(kDeterministic));

	ret->ForecastTypes(ftypes);

	auto base = unique_ptr<grid>(new point_list());  // placeholder
	base->Type(kLatitudeLongitude);
	base->Class(kIrregularGrid);

	ret->Create(base.get());

	dynamic_cast<point_list*>(ret->Grid())->Stations(stats);

	itsLogger->Debug("Read " + boost::lexical_cast<string>(times.size()) + " times, " +
	                 boost::lexical_cast<string>(levels.size()) + " levels and " +
	                 boost::lexical_cast<string>(params.size()) + " params from file '" + inputFile + "'");

	// Then set grids

	// The csv library used is sub-standard in that it doesn't allow rewinding of the
	// file. It does provide functions to set and get file line number, but that doesn't
	// affect the reading of the file!

	csv_reader in2(inputFile);
	in2.read_header(io::ignore_no_column, "station_id", "station_name", "longitude", "latitude", "origintime",
	                "forecasttime", "level_name", "level_value", "parameter_name", "value");

	int counter = 0;

	while (GetLine(in2, line))
	{
		forecast_time f(get<4>(line), get<5>(line));

		string level_name = get<6>(line);
		boost::algorithm::to_lower(level_name);

		level l(HPStringToLevelType.at(level_name), get<7>(line));
		param p(get<8>(line));

		station s(get<0>(line), get<1>(line), get<2>(line), get<3>(line));

		if (!ret->Param(p)) continue;
		if (!ret->Time(f)) continue;
		if (!ret->Level(l)) continue;

		for (size_t i = 0; i < stats.size(); i++)
		{
			if (s == stats[i])
			{
				// Add the data point
				ret->Grid()->Value(i, get<9>(line));
				counter++;
			}
		}
	}

	itsLogger->Debug("Read " + boost::lexical_cast<string>(counter) + " lines of data");
	return ret;
}
