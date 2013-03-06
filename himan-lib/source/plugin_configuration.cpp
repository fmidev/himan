/**
 * @file plugin_configuration.cpp
 *
 * @date Feb 11, 2013
 * @author partio
 */

#include "plugin_configuration.h"

using namespace himan;

plugin_configuration::plugin_configuration() 
	: itsName("")
	, itsOptions()
	, itsStatistics(new statistics)
{
}

plugin_configuration::plugin_configuration(std::shared_ptr<configuration> conf) 
	: configuration(*conf)
	, itsName("")
	, itsOptions()
	, itsStatistics(new statistics)
{
}

plugin_configuration::plugin_configuration(const std::string& theName, const std::map<std::string,std::string>& theOptions)
	: itsName(theName)
	, itsOptions(theOptions)
	, itsStatistics(new statistics)
{
}

void plugin_configuration::AddOption(const std::string& key, const std::string& value)
{
	itsOptions[key] = value;
}

void plugin_configuration::Options(const std::map<std::string,std::string>& theOptions)
{
	itsOptions = theOptions;
}

const std::map<std::string,std::string>& plugin_configuration::Options() const
{
	return itsOptions;
}

void plugin_configuration::Name(const std::string& theName)
{
	itsName = theName;
}

std::string plugin_configuration::Name() const
{
	return itsName;
}

bool plugin_configuration::Exists(const std::string & key) const
{
	return !(GetValue(key).empty());
}

std::string plugin_configuration::GetValue(const std::string & key) const
{

	std::map<std::string,std::string>::const_iterator iter = itsOptions.find(key);

	if (iter == itsOptions.end())
	{
		return "";
	}

	return iter->second;
}

std::shared_ptr<info> plugin_configuration::Info() const
{
	return itsInfo;
}

void plugin_configuration::Info(std::shared_ptr<info> theInfo) 
{
	itsInfo = theInfo;
}

std::shared_ptr<statistics> plugin_configuration::Statistics() const
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
	std::cout << "*** STATISTICS FOR " << itsStatisticsLabel << " ***" << std::endl;

	std::cout << "use cuda:\t" << itsUseCuda << std::endl;
	std::cout << "origin time:\t" << itsInfo->OriginDateTime().String() << std::endl;

	std::cout << "geom_name\t" << itsGeomName << std::endl;
	
	// Hoping we have iterators set

	std::cout << "level type:\t" << HPLevelTypeToString.at(itsInfo->Level().Type()) << std::endl;
	std::cout << "level count:\t" << itsInfo->SizeLevels() << std::endl;

	// assuming even time step

	std::cout << "time step:\t" << itsInfo->Time().Step() << std::endl;
	std::cout << "time step unit:\t" << itsInfo->Time().StepResolution() << std::endl;
	std::cout << "time count:\t" << itsInfo->SizeTimes() << std::endl;

	std::cout << "file type:\t" << itsOutputFileType << std::endl;
	std::cout << "whole_fw:\t" << itsWholeFileWrite << std::endl;
	std::cout << "read_from_db:\t" << itsReadDataFromDatabase << std::endl;
	std::cout << "leading_dim:\t" << itsLeadingDimension << std::endl;

	std::cout << "plugin:\t\t" << itsName << std::endl;

	itsStatistics->Write();
}

std::ostream& plugin_configuration::Write(std::ostream& file) const
{

	// configuration::Write();
	
    file << "<" << ClassName() << " " << Version() << ">" << std::endl;
    file << "__itsName__ " << itsName << std::endl;

    for(std::map<std::string, std::string>::const_iterator iter = itsOptions.begin(); iter != itsOptions.end(); ++iter)
    {
    	file << "__" << iter->first << "__ " << iter->second << std::endl;
    }

    return file;
}
