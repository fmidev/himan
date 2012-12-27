/*
 * writer.cpp
 *
 *  Created on: Nov 26, 2012
 *      Author: partio
 */

#include "writer.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <fstream>
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "grib.h"
#include "querydata.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan::plugin;

writer::writer()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("writer"));
}

bool writer::ToFile(std::shared_ptr<info> theInfo,
                    const std::string& theOutputFile,
                    HPFileType theFileType,
                    bool theActiveOnly)
{

	bool ret = false;

	boost::filesystem::path p(theOutputFile);
	std::string theCorrectOutputFile = p.stem().string();

	switch (theFileType)
	{

		case kGRIB:
		case kGRIB1:
		case kGRIB2:
			{

				theCorrectOutputFile += ".grib";

				std::shared_ptr<grib> theGribWriter = std::dynamic_pointer_cast<grib> (plugin_factory::Instance()->Plugin("grib"));

				ret = theGribWriter->ToFile(theInfo, theCorrectOutputFile, theFileType, theActiveOnly);

				break;
			}
		case kQueryData:
			{
				theCorrectOutputFile += ".fqd";

				std::shared_ptr<querydata> theWriter = std::dynamic_pointer_cast<querydata> (plugin_factory::Instance()->Plugin("querydata"));

				if (!theWriter)
				{
					throw std::runtime_error("Unable to load QueryData plugin");
				}

				ret = theWriter->ToFile(theInfo, theCorrectOutputFile, theActiveOnly);

				break;
			}
		case kNetCDF:
			break;

			// Must have this or compiler complains
		default:
			throw std::runtime_error(ClassName() + ": Invalid file type: " + boost::lexical_cast<std::string> (theFileType));
			break;

	}

	if (ret)
	{
		itsLogger->Info("Wrote file '" + theCorrectOutputFile + "'");
	}

	return ret;
}
