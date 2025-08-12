/*
 * json_parser.h
 *
 */

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include "plugin_configuration.h"
#include <vector>

namespace himan
{
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
