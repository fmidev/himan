/**
 * @file plugin_configuration.cpp
 *
 * @date Feb 11, 2013
 * @author partio
 */

#include "plugin_configuration.h"

using namespace himan;
using namespace std;

plugin_configuration::plugin_configuration() 
	: itsName("")
	, itsOptions()
	, itsStatistics(new statistics)
{
}

plugin_configuration::plugin_configuration(shared_ptr<configuration> conf) 
	: configuration(*conf)
	, itsName("")
	, itsOptions()
	, itsStatistics(new statistics)
{
}

plugin_configuration::plugin_configuration(const string& theName, const map<string,string>& theOptions)
	: itsName(theName)
	, itsOptions(theOptions)
	, itsStatistics(new statistics)
{
}

void plugin_configuration::AddOption(const string& key, const string& value)
{
	itsOptions[key] = value;
}

void plugin_configuration::Options(const map<string,string>& theOptions)
{
	itsOptions = theOptions;
}

const map<string,string>& plugin_configuration::Options() const
{
	return itsOptions;
}

void plugin_configuration::Name(const string& theName)
{
	itsName = theName;
}

string plugin_configuration::Name() const
{
	return itsName;
}

bool plugin_configuration::Exists(const string & key) const
{
	return !(GetValue(key).empty());
}

string plugin_configuration::GetValue(const string & key) const
{

	map<string,string>::const_iterator iter = itsOptions.find(key);

	if (iter == itsOptions.end())
	{
		return "";
	}

	return iter->second;
}

shared_ptr<info> plugin_configuration::Info() const
{
	return itsInfo;
}

void plugin_configuration::Info(shared_ptr<info> theInfo) 
{
	itsInfo = theInfo;
}

shared_ptr<statistics> plugin_configuration::Statistics() const
{
	return itsStatistics;
}

bool plugin_configuration::StatisticsEnabled() const
{
	return !(itsStatisticsLabel.empty());

}

void plugin_configuration::StartStatistics()
{
	itsStatistics->Start();
}

void plugin_configuration::WriteStatistics()
{
	itsStatistics->itsTimer->Stop();

	cout << "*** STATISTICS FOR " << itsStatisticsLabel << " ***" << endl;

	cout << "Use cache:\t\t" << (itsUseCache ? "true" : "false") << endl;
	cout << "Use cuda:\t\t" << (itsUseCuda ? "true" : "false") << endl;
	cout << "Use cuda packing:\t" << (itsUseCudaForPacking ? "true" : "false") << endl;

	cout << "Origin time:\t\t" << itsInfo->OriginDateTime().String() << endl;

	cout << "geom_name:\t\t" << itsGeomName << endl;
	
	// Hoping we have iterators set

	itsInfo->First();

	cout << "Level type:\t\t" << HPLevelTypeToString.at(itsInfo->Level().Type()) << endl;
	cout << "Level count:\t\t" << itsInfo->SizeLevels() << endl;

	// assuming even time step

	cout << "Time step:\t\t" << itsInfo->Time().Step() << endl;
	cout << "Time step unit:\t\t" << HPTimeResolutionToString.at(itsInfo->Time().StepResolution()) << endl;
	cout << "Time count:\t\t" << itsInfo->SizeTimes() << endl;

	cout << "Outfile type:\t\t" << HPFileTypeToString.at(itsOutputFileType) << endl;
	cout << "File write:\t\t" << HPFileWriteOptionToString.at(itsFileWriteOption) << endl;
	cout << "Read from database:\t" << (itsReadDataFromDatabase ? "true" : "false") << endl;
	cout << "Leading dimension:\t" << HPDimensionTypeToString.at(itsLeadingDimension) << endl;

	cout << "Source producer:\t" << SourceProducer().Id() << endl;
	cout << "Target producer:\t" << itsInfo->Producer().Id() << endl;

	cout << "Plugin:\t\t\t" << itsName << endl;

	// Statistics from class statistics
	
	size_t elapsedTime = itsStatistics->itsTimer->GetTime();

	size_t threadCountDivisor = itsStatistics->itsUsedThreadCount;

	if (itsLeadingDimension == kTimeDimension && itsInfo->SizeTimes() < itsStatistics->itsUsedThreadCount)
	{
		threadCountDivisor = itsInfo->SizeTimes();
	}
	else if (itsLeadingDimension == kLevelDimension && itsInfo->SizeLevels() < itsStatistics->itsUsedThreadCount)
	{
		threadCountDivisor = itsInfo->SizeLevels();
	}

	if (threadCountDivisor == 0)
	{
		itsLogger->Warning("Unable to print statistics due to invalid value for threadcount (0) -- somebody forgot to configure their plugin?");
		return;
	}
	
	int fetchingTimePercentage = static_cast<int> (100*static_cast<double> (itsStatistics->itsFetchingTime)/static_cast<double>(threadCountDivisor)/static_cast<double> (elapsedTime));
	int processingTimePercentage = static_cast<int> (100*static_cast<double> (itsStatistics->itsProcessingTime-itsStatistics->itsFetchingTime/threadCountDivisor)/static_cast<double> (elapsedTime));
	int initTimePercentage = static_cast<int> (100*static_cast<double> (itsStatistics->itsInitTime)/static_cast<double> (elapsedTime));

	int writingTimePercentage = 0;

	string writingThreads;

	if (itsFileWriteOption == kSingleFile)
	{
		writingTimePercentage = static_cast<int> (100*static_cast<double> (itsStatistics->itsWritingTime)/static_cast<double> (elapsedTime));
		writingThreads = ", single thread";
	}
	else
	{
		writingTimePercentage = static_cast<int> (100*static_cast<double> (itsStatistics->itsWritingTime/threadCountDivisor)/static_cast<double> (elapsedTime));
		writingThreads = ", average over used threads";
	}

	cout << "Thread count:\t\t" <<  itsStatistics->itsUsedThreadCount << endl
		<< "Used GPU count:\t\t" << itsStatistics->itsUsedGPUCount << endl
		<< "Cache hit count:\t" << itsStatistics->itsCacheHitCount << endl
		<< "Cache miss count:\t" << itsStatistics->itsCacheMissCount << endl
		<< "Elapsed time:\t\t" <<  elapsedTime << " microseconds" << endl
		<< "Plugin init time\t" << itsStatistics->itsInitTime << " microseconds, single thread (" << initTimePercentage << "%)" << endl
		<< "Fetching time:\t\t" << itsStatistics->itsFetchingTime/threadCountDivisor << " microseconds, average over used threads (" << fetchingTimePercentage << "%)" << endl
		<< "Process time:\t\t" << itsStatistics->itsProcessingTime/threadCountDivisor << " microseconds, total over used threads (" << processingTimePercentage << "%)" << endl
		<< "Writing time:\t\t" << itsStatistics->itsWritingTime/threadCountDivisor << " microseconds" << writingThreads << " (" << writingTimePercentage << "%)" << endl
		<< "Values:\t\t\t" << itsStatistics->itsValueCount << endl
		<< "Missing values:\t\t" << itsStatistics->itsMissingValueCount << " (" << static_cast<int> (100*static_cast<double>(itsStatistics->itsMissingValueCount)/static_cast<double>(itsStatistics->itsValueCount)) << "%)" << endl
		<< "PPS:\t\t\t" << 1000*1000*static_cast<double>(itsStatistics->itsValueCount)/static_cast<double>(elapsedTime) << endl;

}

ostream& plugin_configuration::Write(ostream& file) const
{

	// configuration::Write();
	
	file << "<" << ClassName() << " " << Version() << ">" << endl;
	file << "__itsName__ " << itsName << endl;

	for(map<string, string>::const_iterator iter = itsOptions.begin(); iter != itsOptions.end(); ++iter)
	{
		file << "__" << iter->first << "__ " << iter->second << endl;
	}

	configuration::Write(file);

	return file;
}
