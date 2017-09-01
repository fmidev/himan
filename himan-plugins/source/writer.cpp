/*
 * writer.cpp
 *
 */

#include "writer.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <boost/filesystem.hpp>
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
	itsLogger = logger("writer");
}

bool writer::CreateFile(info& theInfo, std::shared_ptr<const plugin_configuration> conf, std::string& theOutputFile)
{
	namespace fs = boost::filesystem;

	itsWriteOptions.configuration = conf;

	if (theOutputFile.empty())
	{
		theOutputFile = util::MakeFileName(itsWriteOptions.configuration->FileWriteOption(), theInfo);
	}

	fs::path pathname(theOutputFile);

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

			theOutputFile += ".grib";

			if (itsWriteOptions.configuration->OutputFileType() == kGRIB2)
			{
				theOutputFile += "2";
			}

			if (itsWriteOptions.configuration->FileCompression() == kGZIP)
			{
				theOutputFile += ".gz";
			}
			else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
			{
				theOutputFile += ".bz2";
			}

			theGribWriter->WriteOptions(itsWriteOptions);
			return theGribWriter->ToFile(
			    theInfo, theOutputFile,
			    (itsWriteOptions.configuration->FileWriteOption() == kSingleFile) ? true : false);
		}
		case kQueryData:
		{
			if (theInfo.Grid()->Type() == kReducedGaussian)
			{
				itsLogger.Error("Reduced gaussian grid cannot be written to querydata");
				return false;
			}

			auto theWriter = GET_PLUGIN(querydata);
			theWriter->WriteOptions(itsWriteOptions);

			theOutputFile += ".fqd";

			return theWriter->ToFile(theInfo, theOutputFile);
		}
		case kNetCDF:
			break;

		case kCSV:
		{
			auto theWriter = GET_PLUGIN(csv);
			theWriter->WriteOptions(itsWriteOptions);

			theOutputFile += ".csv";

			return theWriter->ToFile(theInfo, theOutputFile);
		}
		// Must have this or compiler complains
		default:
			throw std::runtime_error(ClassName() + ": Invalid file type: " +
			                         HPFileTypeToString.at(itsWriteOptions.configuration->OutputFileType()));
			break;
	}

	return false;
}

bool writer::ToFile(info& theInfo, std::shared_ptr<const plugin_configuration> conf,
                    const std::string& theOriginalOutputFile)
{
	timer t;

	if (conf->StatisticsEnabled())
	{
		t.Start();
	}

	bool ret = true;
	std::string theOutputFile = theOriginalOutputFile;  // This is modified

	if (conf->FileWriteOption() != kCacheOnly)
	{
		// When writing previ to database, no file is needed. In all other cases we have to create
		// a file.

		if (theInfo.Producer().Class() == kGridClass ||
		    (theInfo.Producer().Class() == kPreviClass && conf->FileWriteOption() != kDatabase))
		{
			ret = CreateFile(theInfo, conf, theOutputFile);
		}

		if (ret && conf->FileWriteOption() == kDatabase)
		{
			HPDatabaseType dbtype = conf->DatabaseType();

			if (dbtype == kNeons || dbtype == kNeonsAndRadon)
			{
				auto n = GET_PLUGIN(neons);

				ret = n->Save(theInfo, theOutputFile);

				if (!ret)
				{
					itsLogger.Warning("Saving file information to neons failed");
				}
			}

			if (dbtype == kRadon || dbtype == kNeonsAndRadon)
			{
				auto r = GET_PLUGIN(radon);

				// Try to save file information to radon
				try
				{
					ret = r->Save(theInfo, theOutputFile);

					if (!ret)
					{
						itsLogger.Error("Writing to radon failed");
					}
				}
				catch (const std::exception& e)
				{
					itsLogger.Error("Writing to radon failed: " + std::string(e.what()));
				}
				catch (...)
				{
					itsLogger.Error("Writing to radon failed: general exception");
				}
			}
		}
	}

	if (conf->UseCache())
	{
		std::shared_ptr<cache> c = GET_PLUGIN(cache);

		// Pin those items that are not written to file at all
		// so they can't be removed from cache if cache size is limited
		c->Insert(theInfo, (conf->FileWriteOption() == kCacheOnly));
	}

	if (conf->StatisticsEnabled())
	{
		t.Stop();

		conf->Statistics()->AddToWritingTime(t.GetTime());
	}

	return ret;
}

write_options writer::WriteOptions() const { return itsWriteOptions; }
void writer::WriteOptions(const write_options& theWriteOptions) { itsWriteOptions = theWriteOptions; }
