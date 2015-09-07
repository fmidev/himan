/*
 * writer.cpp
 *
 *  Created on: Nov 26, 2012
 *	  Author: partio
 */

#include "writer.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "util.h"
#include "timer_factory.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "grib.h"
#include "querydata.h"
#include "neons.h"
#include "radon.h"
#include "cache.h"
#include "csv.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan::plugin;

writer::writer() : itsWriteOptions()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("writer"));
}

bool writer::ToFile(info& theInfo,
					const plugin_configuration& conf,
					const std::string& theOutputFile)
{

	std::unique_ptr<himan::timer> t = std::unique_ptr<himan::timer> (timer_factory::Instance()->GetTimer());

	if (conf.StatisticsEnabled())
	{
		t->Start();
	}

	namespace fs = boost::filesystem;

	bool ret = false;

	std::string correctFileName = theOutputFile;
	
	HPFileWriteOption fileWriteOption = conf.FileWriteOption();
	HPFileType fileType = conf.OutputFileType();
	HPFileCompression fileCompression = conf.FileCompression();

	if ((fileWriteOption == kDatabase || fileWriteOption == kMultipleFiles) || correctFileName.empty())
	{
		correctFileName = util::MakeFileName(fileWriteOption, theInfo);
	}

	fs::path pathname(correctFileName);

	if (!pathname.parent_path().empty() && !fs::is_directory(pathname.parent_path()))
	{
		fs::create_directories(pathname.parent_path());
	}

	switch (fileType)
	{

		case kGRIB:
		case kGRIB1:
		case kGRIB2:
		{

			auto theGribWriter = GET_PLUGIN(grib);

			correctFileName += ".grib";

			if (fileType == kGRIB2)
			{
				correctFileName += "2";
			}

			if (fileCompression == kGZIP)
			{
				correctFileName += ".gz";
			}
			else if (fileCompression == kBZIP2)
			{
				correctFileName += ".bz2";
			}

			theGribWriter->WriteOptions(itsWriteOptions);
			ret = theGribWriter->ToFile(theInfo, correctFileName, fileType, fileCompression, fileWriteOption);

			break;
		}
		case kQueryData:
		{
			auto theWriter = GET_PLUGIN(querydata);
			theWriter->WriteOptions(itsWriteOptions);

			correctFileName += ".fqd";

			ret = theWriter->ToFile(theInfo, correctFileName, fileWriteOption);

			break;
		}
		case kNetCDF:
			break;

		case kCSV:
		{
			auto theWriter = GET_PLUGIN(csv);
			theWriter->WriteOptions(itsWriteOptions);

			correctFileName += ".csv";

			ret = theWriter->ToFile(theInfo, correctFileName, fileWriteOption);
			break;
		}
			// Must have this or compiler complains
		default:
			throw std::runtime_error(ClassName() + ": Invalid file type: " + HPFileTypeToString.at(fileType));
			break;

	}

	if (ret && fileWriteOption == kDatabase)
	{
		HPDatabaseType dbtype = conf.DatabaseType();
		
		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = GET_PLUGIN(neons);
			
			ret = n->Save(theInfo, correctFileName);

			if (!ret)
			{
				itsLogger->Warning("Saving file information to neons failed");
			}
		}
		
		if (dbtype == kRadon || dbtype == kNeonsAndRadon)
		{
			auto r = GET_PLUGIN(radon);

			// Try to save file information to radon
			try
			{
				ret = r->Save(theInfo, correctFileName);
			}
			catch(...)
			{
				itsLogger->Error("Writing to radon failed"); 
			}
		}
	}

	bool activeOnly = (conf.FileWriteOption() == kSingleFile) ? false : true;

	if (conf.UseCache())
	{
		std::shared_ptr<cache> c = GET_PLUGIN(cache);

		c->Insert(theInfo, activeOnly);
	}

	if (conf.StatisticsEnabled())
	{
		t->Stop();

		conf.Statistics()->AddToWritingTime(t->GetTime());
	}

	return ret;
}

write_options writer::WriteOptions() const
{
	return itsWriteOptions;
}

void writer::WriteOptions(const write_options& theWriteOptions)
{
	itsWriteOptions = theWriteOptions;
}