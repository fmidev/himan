/*
 * ini_parser.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef INI_PARSER_H
#define INI_PARSER_H

#include <string>
#include <vector>
#include "configuration.h"

namespace himan
{

class ini_parser : public boost::noncopyable
{

	public:

		static ini_parser* Instance();

		std::shared_ptr<configuration> Parse(int argc, char** argv);
		std::shared_ptr<configuration> GetConfiguration() const;

		virtual std::string ClassName() const
		{
			return "himan::ini_parser";
		}
		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

	private:

		ini_parser();
		~ini_parser() {}
		ini_parser(const ini_parser& other) = delete;
		ini_parser& operator=(const ini_parser& other) = delete;

		void ParseAndCreateInfo(int argc, char** argv);
		void ParseCommandLine(int argc, char** argv);
		void ParseConfigurationFile(const std::string& theConfigurationFile);
		bool ParseBoolean(std::string& booleanValue);

		std::vector<std::string> Split(const std::string& s, char delim, bool fill = false);
		std::vector<std::string> &split(const std::string& s, char delim, std::vector<std::string> &elems);

		static ini_parser* itsInstance;
		std::unique_ptr<logger> itsLogger;

		std::shared_ptr<configuration> itsConfiguration;

};

} // namespace himan

#endif /* INI_PARSER_H */
