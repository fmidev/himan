/*
 * util.cpp
 *
 *  Created on: Dec  1, 2012
 *      Author: partio
 */

#include "util.h"
#include <boost/filesystem/operations.hpp>
#include "logger_factory.h"
#include <sstream>
#include <iomanip>

using namespace himan::plugin;

util::util()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("util"));
}

std::string util::MakeNeonsFileName(std::shared_ptr<const info> info) const
{

	std::ostringstream neonsFileName;

	std::string base = "/tmp"; //"/cluster/hiladata/BDAP/REFStorage/";

	neonsFileName 	<< base
					<< "/"
					<< info->Producer().Centre()
					<< "_"
					<< info->Producer().Process()
					<< "/"
	                << info->Time()->ValidDateTime()->String("%Y%m%d%H%M")
	                << "/"
	                << info->Param()->Name()
	                << "_"
	                << HPLevelTypeToString.at(info->Level()->Type())
	                << "_"
	                << info->Level()->Value()
	                << "_"
	                << HPProjectionTypeToString.at(info->Projection())
	                << "_"
	                << info->Ni()
	                << "_"
	                << info->Nj()
	                << "_0_"
	                << std::setw(3)
                    << std::setfill('0')
                    << info->Time()->Step()
	                ;

	return neonsFileName.str();

}

himan::HPFileType util::FileType(const std::string& theFile) const
{

	using namespace std;

	if (!boost::filesystem::exists(theFile))
	{
		cerr << "Input file '" + theFile + "' does not exist" << endl;
		exit(1);
	}

	ifstream f(theFile.c_str(), ios::in | ios::binary);

	char* content;

	short keywordLength = 4;

	content = static_cast<char*> (malloc((keywordLength + 1) * sizeof(char)));

	f.read(content, keywordLength);

	HPFileType ret = kUnknownFile;

	if (strncmp(content, "GRIB", 4) == 0)
	{
		ret = kGRIB;
	}
	else if (strncmp(content, "CDF", 3) == 0)
	{
		ret = kNetCDF;
	}
	else
	{
		// Not GRIB or NetCDF, keep on searching

		keywordLength = 5;

		free(content);

		content = static_cast<char*> (malloc((keywordLength + 1) * sizeof(char)));

		f.read(content, keywordLength);

		if (strncmp(content, "QINFO", 5) == 0)
		{
			ret = kQueryData;
		}

	}

	free (content);

	return ret;
}
