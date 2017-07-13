/**
 * @file plugin_configuration.cpp
 *
 */

#include "plugin_configuration.h"
#include "forecast_time.h"
#include "level.h"
#include "util.h"

#include <algorithm>
#include <iterator>

using namespace himan;
using namespace std;

plugin_configuration::plugin_configuration()
    : itsName(""), itsOptions(), itsPreconfiguredParams(), itsStatistics(new statistics)
{
}

plugin_configuration::plugin_configuration(const configuration& theConfiguration)
    : configuration(theConfiguration),
      itsName(""),
      itsOptions(),
      itsPreconfiguredParams(),
      itsInfo(new info),
      itsStatistics(new statistics)
{
}

plugin_configuration::plugin_configuration(const plugin_configuration& other)
    : configuration(other),
      itsName(other.itsName),
      itsOptions(other.itsOptions),
      itsPreconfiguredParams(other.itsPreconfiguredParams),
      itsInfo(new info(*other.itsInfo)),
      itsStatistics(new statistics(*other.itsStatistics))
{
}

plugin_configuration::plugin_configuration(const string& theName, const map<string, vector<string>>& theOptions)
    : itsName(theName), itsOptions(theOptions), itsPreconfiguredParams(), itsStatistics(new statistics)
{
}

void plugin_configuration::AddOption(const string& key, const string& value) { itsOptions[key].push_back(value); }
void plugin_configuration::Name(const string& theName) { itsName = theName; }
string plugin_configuration::Name() const { return itsName; }
bool plugin_configuration::Exists(const string& key) const
{
	map<string, vector<string>>::const_iterator iter = itsOptions.find(key);
	return (iter != itsOptions.end());
}

string plugin_configuration::GetValue(const string& key) const
{
	map<string, vector<string>>::const_iterator iter = itsOptions.find(key);

	if (iter == itsOptions.end())
	{
		return "";
	}

	if (iter->second.size() > 1)
	{
		throw runtime_error("Key '" + key + "' is a multi-value key");
	}

	assert(iter->second.size() == 1);
	return iter->second[0];
}

const vector<string>& plugin_configuration::GetValueList(const string& key) const { return itsOptions.at(key); }
void plugin_configuration::AddParameter(const string& paramName, const vector<pair<string, string>>& opts)
{
	if (!itsPreconfiguredParams[paramName].empty())
	{
		throw runtime_error(ClassName() + ": duplicate parameter options definition:: '" + paramName + "'");
	}

	for (const auto& p : opts)
	{
		itsPreconfiguredParams[paramName].push_back(p);
	}
}

bool plugin_configuration::ParameterExists(const std::string& paramName) const
{
	auto iter = itsPreconfiguredParams.find(paramName);
	return iter != itsPreconfiguredParams.end();
}

vector<string> plugin_configuration::GetParameterNames() const
{
	vector<string> names;

	for (auto it = itsPreconfiguredParams.begin(); it != itsPreconfiguredParams.end(); ++it)
	{
		names.push_back(it->first);
	}

	return names;
}

const vector<pair<string, string>>& plugin_configuration::GetParameterOptions(const string& paramName) const
{
	auto iter = itsPreconfiguredParams.find(paramName);

	if (iter == itsPreconfiguredParams.end())
	{
		throw runtime_error(ClassName() + ": parameter not found in preconfigured parameter list:: '" + paramName +
		                    "'");
	}

	return iter->second;
}

shared_ptr<info> plugin_configuration::Info() const { return itsInfo; }
void plugin_configuration::Info(shared_ptr<info> theInfo) { itsInfo = theInfo; }
shared_ptr<statistics> plugin_configuration::Statistics() const { return itsStatistics; }
bool plugin_configuration::StatisticsEnabled() const { return !(itsStatisticsLabel.empty()); }
void plugin_configuration::StartStatistics() { itsStatistics->Start(); }
void plugin_configuration::WriteStatistics()
{
	itsStatistics->itsTimer.Stop();

	cout << "*** STATISTICS FOR " << itsStatisticsLabel << " ***" << endl;

	cout << "Plugin:\t\t\t" << itsName << endl;
	cout << "Use cache:\t\t" << (itsUseCache ? "true" : "false") << endl;
	cout << "Use cuda:\t\t" << (itsUseCuda ? "true" : "false") << endl;
	cout << "Use cuda packing:\t" << (itsUseCudaForPacking ? "true" : "false") << endl;
	cout << "Use cuda unpacking:\t" << (itsUseCudaForUnpacking ? "true" : "false") << endl;
	cout << "Use cuda interpolation:\t" << (itsUseCudaForInterpolation ? "true" : "false") << endl;

	cout << "Target geom_name:\t" << itsTargetGeomName << endl;
	cout << "Source geom_name:\t" << util::Join(itsSourceGeomNames, ",") << endl;

	if (itsInfo->DimensionSize() > 0)
	{
		itsInfo->First();

		cout << "Level type:\t\t" << HPLevelTypeToString.at(itsInfo->Level().Type()) << endl;
		cout << "Level count:\t\t" << itsInfo->SizeLevels() << endl;
		cout << "Level order:\t\t" << HPLevelOrderToString.at(itsInfo->LevelOrder()) << endl;

		// assuming even time step

		cout << "Time step:\t\t" << itsInfo->Time().Step() << endl;
		cout << "Time step unit:\t\t" << HPTimeResolutionToString.at(itsInfo->Time().StepResolution()) << endl;
		cout << "Time count:\t\t" << itsInfo->SizeTimes() << endl;
	}

	cout << "Outfile type:\t\t" << HPFileTypeToString.at(itsOutputFileType) << endl;
	cout << "Compression type:\t" << HPFileCompressionToString.at(itsFileCompression) << endl;
	cout << "File write:\t\t" << HPFileWriteOptionToString.at(itsFileWriteOption) << endl;
	cout << "Read from database:\t" << (itsReadDataFromDatabase ? "true" : "false") << endl;

	cout << "Source producer:\t" << SourceProducer().Id() << endl;
	cout << "Target producer:\t" << TargetProducer().Id() << endl;

	// Statistics from class statistics

	// total elapsed time

	size_t elapsedTime = static_cast<size_t>(itsStatistics->itsTimer.GetTime());

	int fetchingTimePercentage =
	    static_cast<int>(100 * static_cast<double>(itsStatistics->itsFetchingTime) /
	                     static_cast<double>(itsStatistics->itsUsedThreadCount) / static_cast<double>(elapsedTime));
	int processingTimePercentage =
	    static_cast<int>(100 * static_cast<double>(itsStatistics->itsProcessingTime -
	                                               itsStatistics->itsFetchingTime / itsStatistics->itsUsedThreadCount) /
	                     static_cast<double>(elapsedTime));
	int initTimePercentage =
	    static_cast<int>(100 * static_cast<double>(itsStatistics->itsInitTime) / static_cast<double>(elapsedTime));

	int writingTimePercentage = 0;

	string writingThreads;

	if (itsFileWriteOption == kSingleFile)
	{
		writingTimePercentage = static_cast<int>(100 * static_cast<double>(itsStatistics->itsWritingTime) /
		                                         static_cast<double>(elapsedTime));
		writingThreads = ", single thread";
	}
	else
	{
		writingTimePercentage = static_cast<int>(
		    100 * static_cast<double>(itsStatistics->itsWritingTime / itsStatistics->itsUsedThreadCount) /
		    static_cast<double>(elapsedTime));
		writingThreads = ", average over used threads";
	}

	assert(itsStatistics->itsValueCount >= itsStatistics->itsMissingValueCount);

	cout << "Thread count:\t\t" << itsStatistics->itsUsedThreadCount << endl
	     << "Used GPU count:\t\t" << itsStatistics->itsUsedGPUCount << endl
	     << "Cache hit count:\t" << itsStatistics->itsCacheHitCount << endl
	     << "Cache miss count:\t" << itsStatistics->itsCacheMissCount << endl
	     << "Elapsed time:\t\t" << elapsedTime << " milliseconds" << endl
	     << "Plugin init time\t" << itsStatistics->itsInitTime << " milliseconds, single thread (" << initTimePercentage
	     << "%)" << endl
	     << "Fetching time:\t\t" << itsStatistics->itsFetchingTime / itsStatistics->itsUsedThreadCount
	     << " milliseconds, average over used threads (" << fetchingTimePercentage << "%)" << endl
	     << "Process time:\t\t" << itsStatistics->itsProcessingTime / itsStatistics->itsUsedThreadCount
	     << " milliseconds, total over used threads (" << processingTimePercentage << "%)" << endl
	     << "Writing time:\t\t" << itsStatistics->itsWritingTime / itsStatistics->itsUsedThreadCount << " milliseconds"
	     << writingThreads << " (" << writingTimePercentage << "%)" << endl
	     << "Values:\t\t\t" << itsStatistics->itsValueCount << endl
	     << "Missing values:\t\t" << itsStatistics->itsMissingValueCount << " ("
	     << static_cast<int>(100 * static_cast<double>(itsStatistics->itsMissingValueCount) /
	                         static_cast<double>(itsStatistics->itsValueCount))
	     << "%)" << endl
	     << "Million PPS:\t\t"
	     << static_cast<double>(itsStatistics->itsValueCount) / (static_cast<double>(elapsedTime) * 1e3)
	     << endl;  // million points per second
}

ostream& plugin_configuration::Write(ostream& file) const
{
	// configuration::Write();

	file << "<" << ClassName() << ">" << endl;
	file << "__itsName__ " << itsName << endl;

	for (map<string, vector<string>>::const_iterator iter = itsOptions.begin(); iter != itsOptions.end(); ++iter)
	{
		file << "__" << iter->first << "__ ";

		for (size_t i = 0; i < iter->second.size(); i++)
		{
			if (i > 1)
			{
				file << ",";
			}

			file << iter->second[i] << endl;
		}
	}

	configuration::Write(file);

	return file;
}
