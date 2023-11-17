#include "csv.h"
#include "filename.h"
#include "logger.h"
#include "point_list.h"
#include "timer.h"
#include "util.h"
#include <algorithm>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace himan::plugin;

csv::csv()
{
	itsLogger = logger("csv");
}

namespace
{

template <typename T>
shared_ptr<himan::info<T>> CSVToInfo(const vector<string>& csv)
{
	using namespace himan;
	vector<forecast_time> times;
	vector<param> params;
	vector<level> levels;
	vector<station> stats;
	vector<forecast_type> ftypes;

	producer prod;

	for (auto& line : csv)
	{
		auto elems = util::Split(line, ",");

		if (elems.size() != 14)
		{
			std::cerr << "Ignoring line '" << line << "'" << std::endl;
			continue;
		}

		// CSV FORMAT
		// 0 producer_id
		// 1 origin time
		// 2 station_id
		// 3 station_name
		// 4 longitude
		// 5 latitude
		// 6 param_name
		// 7 level_name
		// 8 level_value
		// 9 level_value2
		// 10 forecast period
		// 11 forecast_type_id
		// 12 forecast_type_value
		// 13 value

		if (elems[0][0] == '#')
		{
			continue;
		}

		// producer, only single producer per file is supported for now
		prod.Id(stoi(elems[0]));

		// forecast_time
		raw_time originTime(elems[1]), validTime(elems[1]);

		// split HHH:MM:SS and extract hours and minutes
		auto timeparts = util::Split(elems[10], ":");

		validTime.Adjust(kHourResolution, stoi(timeparts[0]));
		validTime.Adjust(kMinuteResolution, stoi(timeparts[1]));

		forecast_time f(originTime, validTime);

		// level
		level l;

		try
		{
			l = level(static_cast<HPLevelType>(HPStringToLevelType.at(boost::algorithm::to_lower_copy(elems[7]))),
			          stod(elems[8]));
		}
		catch (std::out_of_range& e)
		{
			std::cerr << "Level type " << elems[7] << " is not recognized" << std::endl;
			himan::Abort();
		}

		if (!elems[9].empty())
		{
			l.Value2(stod(elems[9]));
		}
		// param
		const param p(boost::algorithm::to_upper_copy(elems[6]));

		// forecast_type
		const forecast_type ftype(static_cast<HPForecastType>(stoi(elems[11])), stod(elems[12]));

		// station
		const int stationId = (elems[2].empty()) ? kHPMissingInt : stoi(elems[2]);
		const double longitude = (elems[4].empty()) ? kHPMissingValue : stod(elems[4]);
		const double latitude = (elems[5].empty()) ? kHPMissingValue : stod(elems[5]);

		const station s(stationId, elems[3], longitude, latitude);

		/* Prevent duplicates */

		if (find(times.begin(), times.end(), f) == times.end())
		{
			times.push_back(f);
		}

		if (find(levels.begin(), levels.end(), l) == levels.end())
		{
			levels.push_back(l);
		}

		if (find(params.begin(), params.end(), p) == params.end())
		{
			params.push_back(p);
		}

		if (find(ftypes.begin(), ftypes.end(), ftype) == ftypes.end())
		{
			ftypes.push_back(ftype);
		}

		if (find(stats.begin(), stats.end(), s) == stats.end())
		{
			stats.push_back(s);
		}
	}

	if (times.size() == 0 || params.size() == 0 || levels.size() == 0 || ftypes.size() == 0)
	{
		return nullptr;
	}

	auto ret = make_shared<info<T>>();

	ret->Producer(prod);
	ret->template Set<forecast_time>(times);
	ret->template Set<param>(params);
	ret->template Set<level>(levels);
	ret->template Set<forecast_type>(ftypes);

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(new point_list(stats));
	ret->Create(b, true);

	for (auto& line : csv)
	{
		auto elems = util::Split(line, ",");

		if (elems.size() != 14)
		{
			std::cerr << "Ignoring line '" << line << "'" << std::endl;
			continue;
		}

		// 0 producer_id
		// 1 origin time
		// 2 station_id
		// 3 station_name
		// 4 longitude
		// 5 latitude
		// 6 param_name
		// 7 level_name
		// 8 level_value
		// 9 level_value2
		// 10 forecast period
		// 11 forecast_type_id
		// 12 forecast_type_value
		// 13 value

		if (elems[0][0] == '#')
		{
			continue;
		}
		// forecast_time
		raw_time originTime(elems[1]), validTime(elems[1]);

		auto timeparts = util::Split(elems[10], ":");

		validTime.Adjust(kHourResolution, stoi(timeparts[0]));
		validTime.Adjust(kMinuteResolution, stoi(timeparts[1]));

		const forecast_time f(originTime, validTime);

		// level
		level l(static_cast<HPLevelType>(HPStringToLevelType.at(boost::algorithm::to_lower_copy(elems[7]))),
		        stod(elems[8]));

		if (!elems[9].empty())
		{
			l.Value2(stod(elems[9]));
		}

		// param
		const param p(elems[6]);

		// forecast_type
		const forecast_type ftype(static_cast<HPForecastType>(stoi(elems[11])), stod(elems[12]));

		// station
		const int stationId = (elems[2].empty()) ? kHPMissingInt : stoi(elems[2]);
		const double longitude = (elems[4].empty()) ? kHPMissingValue : stod(elems[4]);
		const double latitude = (elems[5].empty()) ? kHPMissingValue : stod(elems[5]);

		const station s(stationId, elems[3], longitude, latitude);

		if (!ret->template Find<param>(p))
			continue;
		if (!ret->template Find<forecast_time>(f))
			continue;
		if (!ret->template Find<level>(l))
			continue;
		if (!ret->template Find<forecast_type>(ftype))
			continue;
		for (size_t i = 0; i < stats.size(); i++)
		{
			if (s == stats[i])
			{
				// Add the data point
				ret->Data().Set(i, static_cast<T>(stod(elems[13])));
			}
		}
	}

	return ret;
}

}  // namespace

pair<himan::HPWriteStatus, himan::file_information> csv::ToFile(info<double>& theInfo)
{
	return ToFile<double>(theInfo);
}

template <typename T>
pair<himan::HPWriteStatus, himan::file_information> csv::ToFile(info<T>& theInfo)
{
	if (theInfo.Grid()->Class() != kIrregularGrid)
	{
		itsLogger.Error("Only irregular grids can be written to CSV");
		throw kInvalidWriteOptions;
	}

	timer aTimer;
	aTimer.Start();

	file_information finfo;
	finfo.file_location = util::filename::MakeFileName(theInfo, *itsWriteOptions.configuration);
	finfo.file_type = kCSV;
	finfo.storage_type = itsWriteOptions.configuration->WriteStorageType();

	namespace bf = std::filesystem;

	bf::path pathname(finfo.file_location);

	if (!pathname.parent_path().empty() && !bf::is_directory(pathname.parent_path()))
	{
		bf::create_directories(pathname.parent_path());
	}

	ofstream out;

	const bool writeHeader = (bf::exists(pathname) == false);

	if (itsWriteOptions.configuration->WriteMode() == kAllGridsToAFile ||
	    itsWriteOptions.configuration->WriteMode() == kFewGridsToAFile)
	{
		out.open(finfo.file_location, ios::out | ios::app);
	}
	else
	{
		out.open(finfo.file_location, ios::out);
	}

	if (!out.is_open())
	{
		itsLogger.Fatal(fmt::format("Failed to open file '{}'", finfo.file_location));
		himan::Abort();
	}

	if (writeHeader)
	{
		out << "#producer_id,origintime,station_id,station_name,longitude,latitude,param_name,level_name,level_value,"
		       "level_"
		       "value2,"
		       "forecast_period,forecast_type_id,forecast_type_value,value"
		    << endl;
	}

	const auto originTime = theInfo.Time().OriginDateTime().String();

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

	aTimer.Stop();

	double duration = static_cast<double>(aTimer.GetTime());
	double bytes = static_cast<double>(filesystem::file_size(finfo.file_location));

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));
	itsLogger.Info(fmt::format("Wrote file '{}' ({} MB/s)", finfo.file_location, speed));

	return make_pair(HPWriteStatus::kFinished, finfo);
}

template pair<himan::HPWriteStatus, himan::file_information> csv::ToFile<double>(info<double>&);
template pair<himan::HPWriteStatus, himan::file_information> csv::ToFile<float>(info<float>&);
template pair<himan::HPWriteStatus, himan::file_information> csv::ToFile<short>(info<short>&);
template pair<himan::HPWriteStatus, himan::file_information> csv::ToFile<unsigned char>(info<unsigned char>&);

template <typename T>
shared_ptr<himan::info<T>> csv::FromMemory(const vector<string>& lines) const
{
	return ::CSVToInfo<T>(lines);
}

template shared_ptr<himan::info<double>> csv::FromMemory<double>(const std::vector<string>&) const;
template shared_ptr<himan::info<float>> csv::FromMemory<float>(const std::vector<string>&) const;
template shared_ptr<himan::info<short>> csv::FromMemory<short>(const std::vector<string>&) const;
template shared_ptr<himan::info<unsigned char>> csv::FromMemory<unsigned char>(const std::vector<string>&) const;

shared_ptr<himan::info<double>> csv::FromFile(const string& inputFile, const search_options& options,
                                              bool forceCaching) const
{
	return FromFile<double>(inputFile, options, forceCaching);
}

template <typename T>
shared_ptr<himan::info<T>> csv::FromFile(const string& inputFile, const search_options& options,
                                         bool forceCaching) const
{
	const bool validate = options.configuration->ValidateMetadata();
	shared_ptr<info<T>> all, requested;

	vector<string> lines;
	string line;

	ifstream infile(inputFile);

	while (getline(infile, line))
	{
		lines.push_back(line);
	}

	all = ::CSVToInfo<T>(lines);

	if (forceCaching)
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
	all->template Reset<param>();

	// Remove those dimensions that are not requested
	while (all->Next())
	{
		if (validate && all->Param() != options.param)
		{
			itsLogger.Debug("Param does not match");
			itsLogger.Debug(fmt::format("{} vs {}", options.param.Name(), all->Param().Name()));
		}
		else if (find(params.begin(), params.end(), all->Param()) == params.end())
		{
			params.push_back(all->Param());
		}

		if (validate && all->Level() != options.level)
		{
			itsLogger.Debug("Level does not match");
			itsLogger.Debug(
			    fmt::format("{} vs {}", static_cast<string>(options.level), static_cast<string>(all->Level())));
		}
		else if (find(levels.begin(), levels.end(), all->Level()) == levels.end())
		{
			levels.push_back(all->Level());
		}

		if (validate && all->Time() != options.time)
		{
			itsLogger.Debug("Time does not match");
			itsLogger.Debug(fmt::format("Origin time {} vs {}", static_cast<string>(optsTime.OriginDateTime()),
			                            static_cast<string>(all->Time().OriginDateTime())));
			itsLogger.Debug(fmt::format("Forecast time: {} vs {}", static_cast<string>(optsTime.ValidDateTime()),
			                            static_cast<string>(all->Time().ValidDateTime())));
		}
		else if (find(times.begin(), times.end(), all->Time()) == times.end())
		{
			times.push_back(all->Time());
		}

		if (validate && all->ForecastType() != options.ftype)
		{
			itsLogger.Debug("Forecast type does not match");
			itsLogger.Debug(
			    fmt::format("{} vs {}", static_cast<string>(options.ftype), static_cast<string>(all->ForecastType())));
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
		itsLogger.Error(fmt::format("Did not find valid data from file '{}'", inputFile));
		throw kFileDataNotFound;
	}

	requested = make_shared<info<T>>(ftypes, times, levels, params);
	requested->Producer(options.prod);

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(new point_list());  // placeholder

	requested->Create(b, true);
	requested->First();
	requested->template Reset<param>();

	while (requested->Next())
	{
		dynamic_pointer_cast<point_list>(requested->Grid())->Stations(stations);
	}

	itsLogger.Debug(fmt::format("Read {} times, {} levels, {} forecast types and {} params from file '{}", times.size(),
	                            levels.size(), ftypes.size(), params.size(), inputFile));

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
template shared_ptr<himan::info<short>> csv::FromFile<short>(const string&, const search_options&, bool) const;
template shared_ptr<himan::info<unsigned char>> csv::FromFile<unsigned char>(const string&, const search_options&,
                                                                             bool) const;
