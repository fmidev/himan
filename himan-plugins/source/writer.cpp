#include "writer.h"
#include "logger.h"
#include "plugin_factory.h"
#include "statistics.h"
#include "timer.h"
#include "util.h"
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
himan::file_information writer::CreateFile(info<T>& theInfo, std::shared_ptr<const plugin_configuration> conf)
{
	itsWriteOptions.configuration = conf;

	switch (itsWriteOptions.configuration->OutputFileType())
	{
		case kGRIB:
		case kGRIB1:
		case kGRIB2:
		{
			auto theGribWriter = GET_PLUGIN(grib);

			theGribWriter->WriteOptions(itsWriteOptions);
			return theGribWriter->ToFile<T>(theInfo);
		}
		case kQueryData:
		{
			if (theInfo.Grid()->Type() == kReducedGaussian)
			{
				itsLogger.Error("Reduced gaussian grid cannot be written to querydata");
				throw kInvalidWriteOptions;
			}

			auto theWriter = GET_PLUGIN(querydata);
			theWriter->WriteOptions(itsWriteOptions);

			return theWriter->ToFile<T>(theInfo);
		}
		case kNetCDF:
			break;

		case kCSV:
		{
			auto theWriter = GET_PLUGIN(csv);
			theWriter->WriteOptions(itsWriteOptions);

			return theWriter->ToFile<T>(theInfo);
		}
		// Must have this or compiler complains
		default:
			throw std::runtime_error(ClassName() + ": Invalid file type: " +
			                         HPFileTypeToString.at(itsWriteOptions.configuration->OutputFileType()));
			break;
	}

	throw kInvalidWriteOptions;
}

template himan::file_information writer::CreateFile<double>(info<double>&, std::shared_ptr<const plugin_configuration>);
template himan::file_information writer::CreateFile<float>(info<float>&, std::shared_ptr<const plugin_configuration>);

bool writer::ToFile(std::shared_ptr<info<double>> theInfo, std::shared_ptr<const plugin_configuration> conf)

{
	return ToFile<double>(theInfo, conf);
}

template <typename T>
bool writer::ToFile(std::shared_ptr<info<T>> theInfo, std::shared_ptr<const plugin_configuration> conf)
{
	if (!itsWriteOptions.write_empty_grid)
	{
		if (theInfo->Data().MissingCount() == theInfo->Data().Size())
		{
			itsLogger.Info("Not writing empty grid for param " + theInfo->Param().Name() + " time " +
			               theInfo->Time().OriginDateTime().String() + " step " +
			               static_cast<std::string>(theInfo->Time().Step()) + " level " +
			               static_cast<std::string>(theInfo->Level()));
			return false;
		}
	}

	timer t;

	if (conf->StatisticsEnabled())
	{
		t.Start();
	}

	bool ret = true;

	if (conf->WriteMode() != kNoFileWrite)
	{
		// When writing previ to database, no file is needed. In all other cases we have to create
		// a file.

		file_information finfo;

		if (theInfo->Producer().Class() == kGridClass ||
		    (theInfo->Producer().Class() == kPreviClass && conf->WriteToDatabase() == false))
		{
			finfo = CreateFile<T>(*theInfo, conf);
		}

		if (conf->WriteToDatabase() == true)
		{
			HPDatabaseType dbtype = conf->DatabaseType();

			if (dbtype == kRadon)
			{
				auto r = GET_PLUGIN(radon);

				// Try to save file information to radon
				try
				{
					if (!r->Save<T>(*theInfo, finfo, conf->TargetGeomName()))
					{
						itsLogger.Error("Writing to radon failed");
						ret = false;
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

	if (conf->UseCacheForWrites())
	{
		auto c = GET_PLUGIN(cache);

		// Pin those items that are not written to file at all
		// so they can't be removed from cache if cache size is limited
		c->Insert<T>(theInfo, (conf->WriteMode() == kNoFileWrite));
	}

	if (conf->StatisticsEnabled())
	{
		t.Stop();

		conf->Statistics()->AddToWritingTime(t.GetTime());
	}

	return ret;
}

template bool writer::ToFile<double>(std::shared_ptr<info<double>>, std::shared_ptr<const plugin_configuration>);
template bool writer::ToFile<float>(std::shared_ptr<info<float>>, std::shared_ptr<const plugin_configuration>);

write_options writer::WriteOptions() const
{
	return itsWriteOptions;
}
void writer::WriteOptions(const write_options& theWriteOptions)
{
	itsWriteOptions = theWriteOptions;
}
