/**
 * @file plugin_configuration.cpp
 *
 */

#include "plugin_configuration.h"
#include "forecast_time.h"
#include "level.h"
#include "statistics.h"
#include "util.h"

#include <algorithm>
#include <iterator>

using namespace himan;
using namespace std;

plugin_configuration::plugin_configuration()
    : itsName(""),
      itsOptions(),
      itsPreconfiguredParams(),
      itsStatistics(new statistics),
      itsOrdinalNumber(0),
      itsRelativeOrdinalNumber(0)
{
}

plugin_configuration::plugin_configuration(const configuration& theConfiguration)
    : configuration(theConfiguration),
      itsName(""),
      itsOptions(),
      itsPreconfiguredParams(),
      itsStatistics(new statistics),
      itsOrdinalNumber(0),
      itsRelativeOrdinalNumber(0)
{
}

plugin_configuration::plugin_configuration(const plugin_configuration& other)
    : configuration(other),
      itsName(other.itsName),
      itsOptions(other.itsOptions),
      itsPreconfiguredParams(other.itsPreconfiguredParams),
      itsStatistics(new statistics(*other.itsStatistics)),
      itsBaseGrid(other.itsBaseGrid->Clone()),
      itsOrdinalNumber(other.itsOrdinalNumber),
      itsRelativeOrdinalNumber(other.itsRelativeOrdinalNumber)
{
}

plugin_configuration::plugin_configuration(const string& theName, const map<string, vector<string>>& theOptions)
    : itsName(theName),
      itsOptions(theOptions),
      itsPreconfiguredParams(),
      itsStatistics(new statistics),
      itsOrdinalNumber(0),
      itsRelativeOrdinalNumber(0)
{
}

void plugin_configuration::AddOption(const string& key, const string& value)
{
	itsOptions[key].push_back(value);
}
void plugin_configuration::Name(const string& theName)
{
	itsName = theName;
}
string plugin_configuration::Name() const
{
	return itsName;
}
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

	ASSERT(iter->second.size() == 1);
	return iter->second[0];
}

const vector<string>& plugin_configuration::GetValueList(const string& key) const
{
	return itsOptions.at(key);
}
void plugin_configuration::AddParameter(const string& paramName, const vector<pair<string, string>>& opts)
{
	if (!itsPreconfiguredParams[paramName].empty())
	{
		throw runtime_error(ClassName() + ": duplicate parameter options definition: '" + paramName + "'");
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

shared_ptr<statistics> plugin_configuration::Statistics() const
{
	return itsStatistics;
}
bool plugin_configuration::StatisticsEnabled() const
{
	return !(itsStatisticsLabel.empty());
}
void plugin_configuration::WriteStatistics()
{
	cout << "*** STATISTICS ***" << endl;

	cout << setw(30) << left << "Plugin:" << itsName << endl
	     << setw(30) << left << "Use cache for reads:" << (itsUseCacheForReads ? "true" : "false") << endl
	     << setw(30) << left << "Use cache for writes:" << (itsUseCacheForWrites ? "true" : "false") << endl
	     << setw(30) << left << "Use cuda:" << (itsUseCuda ? "true" : "false") << endl
	     << setw(30) << left << "Use cuda unpacking:" << (itsUseCudaForUnpacking ? "true" : "false") << endl
	     << setw(30) << left << "Target geom_name:" << itsTargetGeomName << endl
	     << setw(30) << left << "Source geom_name:" << util::Join(itsSourceGeomNames, ",") << endl
	     << setw(30) << left << "Outfile type:" << HPFileTypeToString.at(itsOutputFileType) << endl
	     << setw(30) << left << "Compression type:" << HPFileCompressionToString.at(itsFileCompression) << endl
	     << setw(30) << left << "Write mode:" << HPWriteModeToString.at(itsWriteMode) << endl
	     << setw(30) << left << "Read from database:" << (itsReadFromDatabase ? "true" : "false") << endl
	     << setw(30) << left << "Write to database:" << (itsWriteToDatabase ? "true" : "false") << endl
	     << setw(30) << left << "Write storage type: " << HPFileStorageTypeToString.at(itsWriteStorageType) << endl;

	string sourceProducers = "";

	for (const auto& prod : itsSourceProducers)
	{
		sourceProducers += to_string(prod.Id()) + ",";
	}

	sourceProducers.pop_back();

	cout << setw(30) << left << "Source producer:" << sourceProducers << endl;
	cout << setw(30) << left << "Target producer:" << TargetProducer().Id() << endl;

	// Statistics from class statistics

	ASSERT(itsStatistics->itsValueCount >= itsStatistics->itsMissingValueCount);

	const double totalTime = static_cast<double>(itsStatistics->itsInitTime + itsStatistics->itsFetchingTime +
	                                             itsStatistics->itsProcessingTime + itsStatistics->itsWritingTime);

	const int initP = static_cast<int>(100. * static_cast<double>(itsStatistics->itsInitTime) / totalTime);
	const int fetchP = static_cast<int>(100. * static_cast<double>(itsStatistics->itsFetchingTime) / totalTime);
	const int procP = static_cast<int>(100. * static_cast<double>(itsStatistics->itsProcessingTime) / totalTime);
	const int writeP = static_cast<int>(100. * static_cast<double>(itsStatistics->itsWritingTime) / totalTime);

	cout << setw(30) << left << "Thread count:" << itsStatistics->itsUsedThreadCount << endl
	     << setw(30) << left << "Cache hit count:" << itsStatistics->itsCacheHitCount << endl
	     << setw(30) << left << "Cache miss count:" << itsStatistics->itsCacheMissCount << endl
	     << setw(30) << left << "Elapsed wall time:" << setw(7) << right << itsStatistics->itsTotalTime << " ms" << endl
	     << setw(30) << left << "Plugin init time:" << setw(7) << right << itsStatistics->itsInitTime << " ms ("
	     << setw(2) << initP << "%)" << endl
	     << setw(30) << left << "Fetching time:" << setw(7) << right << itsStatistics->itsFetchingTime << " ms ("
	     << setw(2) << fetchP << "%)" << endl
	     << setw(30) << left << "Process time:" << setw(7) << right << itsStatistics->itsProcessingTime << " ms ("
	     << setw(2) << procP << "%)" << endl
	     << setw(30) << left << "Writing time:" << setw(7) << right << itsStatistics->itsWritingTime << " ms ("
	     << setw(2) << writeP << "%)" << endl
	     << setw(30) << left << "Values:" << itsStatistics->itsValueCount << endl
	     << setw(30) << left << "Missing values:" << itsStatistics->itsMissingValueCount << " ("
	     << static_cast<int>(100 * static_cast<double>(itsStatistics->itsMissingValueCount) /
	                         static_cast<double>(itsStatistics->itsValueCount))
	     << "%)" << endl
	     << setw(30) << left << "Million PPS:"
	     << static_cast<double>(itsStatistics->itsValueCount) / (static_cast<double>(itsStatistics->itsTotalTime) * 1e3)
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

unsigned int plugin_configuration::OrdinalNumber() const
{
	return itsOrdinalNumber;
}

void plugin_configuration::OrdinalNumber(unsigned int theOrdinalNumber)
{
	itsOrdinalNumber = theOrdinalNumber;
}

unsigned int plugin_configuration::RelativeOrdinalNumber() const
{
	return itsRelativeOrdinalNumber;
}

void plugin_configuration::RelativeOrdinalNumber(unsigned int theRelativeOrdinalNumber)
{
	itsRelativeOrdinalNumber = theRelativeOrdinalNumber;
}
