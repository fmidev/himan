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
//#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "grib.h"
#include "querydata.h"
#include "neons.h"
#include "util.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan::plugin;

writer::writer()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("writer"));
}

bool writer::ToFile(std::shared_ptr<info> theInfo,
                    HPFileType theFileType,
                    bool theActiveOnly,
                    const std::string& theOutputFile)
{

	bool ret = false;

	std::string correctFileName = theOutputFile;

	if (correctFileName.empty())
	{
		std::shared_ptr<util> u = std::dynamic_pointer_cast<util> (plugin_factory::Instance()->Plugin("util"));


		correctFileName = u->MakeNeonsFileName(theInfo);
	}

	switch (theFileType)
	{

		case kGRIB:
		case kGRIB1:
		case kGRIB2:
			{

				std::shared_ptr<grib> theGribWriter = std::dynamic_pointer_cast<grib> (plugin_factory::Instance()->Plugin("grib"));

				correctFileName += ".grib";

				ret = theGribWriter->ToFile(theInfo, correctFileName, theFileType, theActiveOnly);

				break;
			}
		case kQueryData:
			{
				std::shared_ptr<querydata> theWriter = std::dynamic_pointer_cast<querydata> (plugin_factory::Instance()->Plugin("querydata"));

				correctFileName += ".fqd";

				ret = theWriter->ToFile(theInfo, correctFileName, theActiveOnly);

				break;
			}
		case kNetCDF:
			break;

			// Must have this or compiler complains
		default:
			throw std::runtime_error(ClassName() + ": Invalid file type: " + boost::lexical_cast<std::string> (theFileType));
			break;

	}

	if (ret && theActiveOnly)
	{
		std::shared_ptr<neons> n = std::dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		// Save file information to neons

		ret = n->Save(theInfo);

		if (ret)
		{
			itsLogger->Info("Wrote file '" + correctFileName + "'");
		}
		else
		{
			itsLogger->Warning("Saving file information to neons failed");
			unlink(correctFileName.c_str());
		}

	}

	return ret;
}
