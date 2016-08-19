/*
 * json_parser.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include "plugin_configuration.h"
#include <boost/property_tree/ptree.hpp>
#include <string>
#include <vector>

namespace himan
{
class json_parser
{
   public:
	static json_parser* Instance();

	json_parser(const json_parser& other) = delete;
	json_parser& operator=(const json_parser& other) = delete;

	std::vector<std::shared_ptr<plugin_configuration>> Parse(std::shared_ptr<configuration> conf);

	std::string ClassName() const { return "himan::json_parser"; }
	~json_parser() = default;

   private:
	json_parser();

	std::vector<std::shared_ptr<plugin_configuration>> ParseConfigurationFile(std::shared_ptr<configuration> conf);
	std::unique_ptr<grid> ParseAreaAndGrid(std::shared_ptr<configuration> conf, const boost::property_tree::ptree& pt);
	void ParseTime(std::shared_ptr<configuration> conf, std::shared_ptr<info> baseInfo,
	               const boost::property_tree::ptree& pt);

	static std::unique_ptr<json_parser> itsInstance;
};

}  // namespace himan

#endif /* JSON_PARSER_H */
