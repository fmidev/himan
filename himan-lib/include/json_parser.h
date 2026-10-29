/*
 * json_parser.h
 *
 */

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include "plugin_configuration.h"
#include <boost/property_tree/ptree.hpp>
#include <vector>

namespace himan
{
/**
 * @brief Read the options of one entry of 'param_list'
 *
 * Options with a string value are returned as key-value pairs in 'opts'. Options with a nested
 * structure (an array of objects or an object) are returned in 'nested'. The option 'name'
 * (or its alias 'producer') is returned in 'name'.
 *
 * @throws runtime_error if an option is empty or if an array contains something else than objects
 */

void ParseParameterOptions(const boost::property_tree::ptree& param, std::string& name,
                           std::vector<std::pair<std::string, std::string>>& opts, option_block& nested);

class json_parser
{
   public:
	json_parser();
	json_parser(const json_parser& other) = delete;
	~json_parser() = default;

	json_parser& operator=(const json_parser& other) = delete;

	std::vector<std::shared_ptr<plugin_configuration>> Parse(std::shared_ptr<configuration> conf);

	std::string ClassName() const
	{
		return "himan::json_parser";
	}

   private:
	std::vector<std::shared_ptr<plugin_configuration>> ParseConfigurationFile(std::shared_ptr<configuration> conf);
};

}  // namespace himan

#endif /* JSON_PARSER_H */
