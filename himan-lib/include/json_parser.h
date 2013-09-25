/*
 * json_parser.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <string>
#include <vector>
#include "plugin_configuration.h"
#include <boost/property_tree/ptree.hpp>

namespace himan
{

class json_parser
{

public:

    static json_parser* Instance();

    json_parser(const json_parser& other) = delete;
    json_parser& operator=(const json_parser& other) = delete;

    std::vector<std::shared_ptr<plugin_configuration>> Parse(std::shared_ptr<configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::json_parser";
    }

private:

    json_parser();
    ~json_parser() {}

    std::vector<std::shared_ptr<plugin_configuration>> ParseConfigurationFile(std::shared_ptr<configuration> conf);
    void ParseAreaAndGrid(std::shared_ptr<configuration> conf, std::shared_ptr<info> baseInfo, const boost::property_tree::ptree& pt);
    void ParseTime(const producer& sourceProducer, std::shared_ptr<info> baseInfo, const boost::property_tree::ptree& pt);
    void ParseProducers(std::shared_ptr<configuration> conf, std::shared_ptr<info> anInfo, const boost::property_tree::ptree& pt);
    void ParseLevels(std::shared_ptr<info> anInfo, const boost::property_tree::ptree& pt);

    std::vector<level> LevelsFromString(const std::string& levelType, const std::string& levelValues) const;
    bool ParseBoolean(std::string& booleanValue);

    static json_parser* itsInstance;
    std::unique_ptr<logger> itsLogger;

};

} // namespace himan

#endif /* JSON_PARSER_H */
