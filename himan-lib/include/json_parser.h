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
#include "configuration.h"
#include <boost/property_tree/ptree.hpp>

namespace himan
{

class json_parser
{

public:

    static json_parser* Instance();

    json_parser(const json_parser& other) = delete;
    json_parser& operator=(const json_parser& other) = delete;

    void Parse(std::shared_ptr<configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::json_parser";
    }
    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

private:

    json_parser();
    ~json_parser() {}

    void ParseConfigurationFile(std::shared_ptr<configuration> conf);
    void ParseAreaAndGrid(std::shared_ptr<configuration> conf, const boost::property_tree::ptree& pt);
    void ParseTime(std::shared_ptr<configuration> conf, const boost::property_tree::ptree& pt);
    bool ParseBoolean(std::string& booleanValue);

    static json_parser* itsInstance;
    std::unique_ptr<logger> itsLogger;

};

} // namespace himan

#endif /* JSON_PARSER_H */
