/*
 * writer.cpp
 *
 */

#include "writer.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include "timer_factory.h"
#include "util.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>

#include "cache.h"
#include "csv.h"
#include "grib.h"
#include "neons.h"
#include "querydata.h"
#include "radon.h"

using namespace himan::plugin;

writer::writer() : itsWriteOptions()
{
	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("writer"));
}

bool writer::ToFile(info& theInfo, std::shared_ptr<const plugin_configuration> conf, const std::string& theOutputFile)
{
	std::unique_ptr<himan::timer> t = std::unique_ptr<himan::timer>(timer_factory::Instance()->GetTimer());

	if (conf->StatisticsEnabled())
	{
		t->Start();
	}

	bool ret = true;

	if (conf->FileWriteOption() != kCacheOnly)
	{
		namespace fs = boost::filesystem;

		std::string correctFileName = theOutputFile;

		itsWriteOptions.configuration = conf;

		if (correctFileName.empty())
		{
			correctFileName = util::MakeFileName(itsWriteOptions.configuration->FileWriteOption(), theInfo);
		}

		fs::path pathname(correctFileName);

		if (!pathname.parent_path().empty() && !fs::is_directory(pathname.parent_path()))
		{
			fs::create_directories(pathname.parent_path());
		}

		switch (itsWriteOptions.configuration->OutputFileType())
		{
			case kGRIB:
			case kGRIB1:
			case kGRIB2:
			{
				auto theGribWriter = GET_PLUGIN(grib);

				correctFileName += ".grib";

				if (itsWriteOptions.configuration->OutputFileType() == kGRIB2)
				{
					correctFileName += "2";
				}

				if (itsWriteOptions.configuration->FileCompression() == kGZIP)
				{
					correctFileName += ".gz";
				}
				else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
				{
					correctFileName += ".bz2";
				}

				theGribWriter->WriteOptions(itsWriteOptions);
				ret = theGribWriter->ToFile(
				    theInfo, correctFileName,
				    (itsWriteOptions.configuration->FileWriteOption() == kSingleFile) ? true : false);

				break;
			}
			case kQueryData:
			{
				if (theInfo.Grid()->Type() == kReducedGaussian)
				{
					itsLogger->Error("Reduced gaussian grid cannot be written to querydata");
					return false;
				}

				auto theWriter = GET_PLUGIN(querydata);
				theWriter->WriteOptions(itsWriteOptions);

				correctFileName += ".fqd";

				ret = theWriter->ToFile(theInfo, correctFileName);

				break;
			}
			case kNetCDF:
				break;

			case kCSV:
			{
				auto theWriter = GET_PLUGIN(csv);
				theWriter->WriteOptions(itsWriteOptions);

				correctFileName += ".csv";

				ret = theWriter->ToFile(theInfo, correctFileName);
				break;
			}
			// Must have this or compiler complains
			default:
				throw std::runtime_error(ClassName() + ": Invalid file type: " +
				                         HPFileTypeToString.at(itsWriteOptions.configuration->OutputFileType()));
				break;
		}

		if (ret && itsWriteOptions.configuration->FileWriteOption() == kDatabase)
		{
			HPDatabaseType dbtype = conf->DatabaseType();

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
					if (!ret) itsLogger->Error("Writing to radon failed");
				}
				catch (const std::exception& e)
				{
					itsLogger->Error("Writing to radon failed: " + std::string(e.what()));
				}
				catch (...)
				{
					itsLogger->Error("Writing to radon failed: general exception");
				}
			}
		}
	}

	if (conf->UseCache())
	{
		std::shared_ptr<cache> c = GET_PLUGIN(cache);

		c->Insert(theInfo);
	}

	if (conf->StatisticsEnabled())
	{
		t->Stop();

		conf->Statistics()->AddToWritingTime(t->GetTime());
	}

	return ret;
}

write_options writer::WriteOptions() const { return itsWriteOptions; }

void writer::WriteOptions(const write_options& theWriteOptions) { itsWriteOptions = theWriteOptions; }
