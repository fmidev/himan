#include "writer.h"
#include "logger.h"
#include "plugin_factory.h"
#include "statistics.h"
#include "timer.h"
#include "util.h"
#include <boost/filesystem.hpp>
#include <fstream>

#include "cache.h"
#include "csv.h"
#include "grib.h"
#include "querydata.h"
#include "radon.h"

using namespace himan::plugin;

writer::writer() : itsWriteOptions()
{
	itsLogger = logger("writer");
}

template <typename T>
bool writer::CreateFile(info<T>& theInfo, std::shared_ptr<const plugin_configuration> conf, std::string& theOutputFile)
{
	namespace fs = boost::filesystem;

	itsWriteOptions.configuration = conf;

	if (theOutputFile.empty())
	{
		theOutputFile = util::MakeFileName(itsWriteOptions.configuration->FileWriteOption(), theInfo, *conf);
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
			return theGribWriter->ToFile<T>(
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

			return theWriter->ToFile<T>(theInfo, theOutputFile);
		}
		case kNetCDF:
			break;

		case kCSV:
		{
			auto theWriter = GET_PLUGIN(csv);
			theWriter->WriteOptions(itsWriteOptions);

			theOutputFile += ".csv";

			return theWriter->ToFile<T>(theInfo, theOutputFile);
		}
		// Must have this or compiler complains
		default:
			throw std::runtime_error(ClassName() + ": Invalid file type: " +
			                         HPFileTypeToString.at(itsWriteOptions.configuration->OutputFileType()));
			break;
	}

	return false;
}

template bool writer::CreateFile<double>(info<double>&, std::shared_ptr<const plugin_configuration>, std::string&);
template bool writer::CreateFile<float>(info<float>&, std::shared_ptr<const plugin_configuration>, std::string&);

bool writer::ToFile(std::shared_ptr<info<double>> theInfo, std::shared_ptr<const plugin_configuration> conf,
                    const std::string& theOriginalOutputFile)
{
	return ToFile<double>(theInfo, conf, theOriginalOutputFile);
}

template <typename T>
bool writer::ToFile(std::shared_ptr<info<T>> theInfo, std::shared_ptr<const plugin_configuration> conf,
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

		if (theInfo->Producer().Class() == kGridClass ||
		    (theInfo->Producer().Class() == kPreviClass && conf->FileWriteOption() != kDatabase))
		{
			ret = CreateFile<T>(*theInfo, conf, theOutputFile);
		}

		if (ret && conf->FileWriteOption() == kDatabase)
		{
			HPDatabaseType dbtype = conf->DatabaseType();

			if (dbtype == kRadon)
			{
				auto r = GET_PLUGIN(radon);

				// Try to save file information to radon
				try
				{
					ret = r->Save<T>(*theInfo, theOutputFile, conf->TargetGeomName());

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
		auto c = GET_PLUGIN(cache);

		// Pin those items that are not written to file at all
		// so they can't be removed from cache if cache size is limited
		c->Insert<T>(theInfo, (conf->FileWriteOption() == kCacheOnly));
	}

	if (conf->StatisticsEnabled())
	{
		t.Stop();

		conf->Statistics()->AddToWritingTime(t.GetTime());
	}

	return ret;
}

template bool writer::ToFile<double>(std::shared_ptr<info<double>>, std::shared_ptr<const plugin_configuration>,
                                     const std::string&);
template bool writer::ToFile<float>(std::shared_ptr<info<float>>, std::shared_ptr<const plugin_configuration>,
                                    const std::string&);

write_options writer::WriteOptions() const
{
	return itsWriteOptions;
}
void writer::WriteOptions(const write_options& theWriteOptions)
{
	itsWriteOptions = theWriteOptions;
}
