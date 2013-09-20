/**
 * @file plugin_configuration.h
 *
 * @date Feb 11, 2013
 * @author partio
 */

#ifndef PLUGIN_CONFIGURATION_H
#define PLUGIN_CONFIGURATION_H

#include "himan_common.h"
#include <vector>
#include <map>
#include "statistics.h"
#include "configuration.h"

namespace himan
{

class plugin_configuration : public configuration
{

public:

	plugin_configuration();
	plugin_configuration(const plugin_configuration& other);
	plugin_configuration& operator=(const plugin_configuration& other) = delete;

	plugin_configuration(std::shared_ptr<configuration> conf);
	plugin_configuration(const std::string& theName, const std::map<std::string,std::string>& theOptions);

	~plugin_configuration() {};

    /**
     * @return Class name
     */

    std::string ClassName() const
    {
        return "himan::plugin_configuration";
    }

    std::ostream& Write(std::ostream& file) const;

    void Name(const std::string& theName);
    std::string Name() const;

    const std::map<std::string,std::string>& Options() const;
    void Options(const std::map<std::string,std::string>& theOptions);

    void AddOption(const std::string& key, const std::string& value);

    bool Exists(const std::string & key) const;
    std::string GetValue(const std::string & key) const;
    
	void Info(std::shared_ptr<info> theInfo);
	std::shared_ptr<info> Info() const;

	std::shared_ptr<statistics> Statistics() const;

	bool StatisticsEnabled() const;
	void StartStatistics();
	void WriteStatistics();
	
private:

	std::string itsName;
	std::map<std::string,std::string> itsOptions;
	std::shared_ptr<info> itsInfo;
	std::shared_ptr<statistics> itsStatistics;
};

inline
std::ostream& operator<<(std::ostream& file, const plugin_configuration& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* PLUGIN_CONFIGURATION_H */
