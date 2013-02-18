/**
 * @file plugin_configuration.h
 *
 * @date Feb 11, 2013
 * @author partio
 */

#ifndef PLUGIN_CONFIGURATION_H
#define PLUGIN_CONFIGURATION_H

#include "param.h"
#include "level.h"
#include "himan_common.h"
#include <vector>
#include <map>

namespace himan
{

class plugin_configuration
{
public:
	plugin_configuration();
	plugin_configuration(const std::string& theName, const std::map<std::string,std::string>& theOptions);

	~plugin_configuration() {};

    /**
     * @return Class name
     */

    std::string ClassName() const
    {
        return "himan::plugin_configuration";
    }

    HPVersionNumber Version() const
    {
	    return HPVersionNumber(0, 1);
    }

    std::ostream& Write(std::ostream& file) const;

    void Name(const std::string& theName);
    std::string Name() const;

    const std::map<std::string,std::string>& Options() const;
    void Options(const std::map<std::string,std::string>& theOptions);

    void AddOption(const std::string& key, const std::string& value);

    bool Exists(const std::string & key) const;
    std::string GetValue(const std::string & key) const;

private:

	std::string itsName;
	std::map<std::string,std::string> itsOptions;

};

inline
std::ostream& operator<<(std::ostream& file, const plugin_configuration& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* PLUGIN_CONFIGURATION_H */
