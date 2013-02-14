/**
 * @file plugin_configuration.cpp
 *
 * @date Feb 11, 2013
 * @author partio
 */

#include "plugin_configuration.h"

using namespace himan;

plugin_configuration::plugin_configuration() : itsName(""), itsOptions() {} ;

plugin_configuration::plugin_configuration(const std::string& theName, const std::map<std::string,std::string>& theOptions)
	: itsName(theName), itsOptions(theOptions)
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

std::ostream& plugin_configuration::Write(std::ostream& file) const
{

    file << "<" << ClassName() << " " << Version() << ">" << std::endl;
    file << "__itsName__ " << itsName << std::endl;

    for(std::map<std::string, std::string>::const_iterator iter = itsOptions.begin(); iter != itsOptions.end(); ++iter)
    {
    	file << "__" << iter->first << "__ " << iter->second << std::endl;
    }

    return file;
}
