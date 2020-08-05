#include "writer.h"
#include "logger.h"
#include "plugin_factory.h"
#include "s3.h"
#include "statistics.h"
#include "timer.h"
#include "util.h"
#include <fstream>

#include "cache.h"
#include "csv.h"
#include "grib.h"
#include "querydata.h"
#include "radon.h"

static std::vector<std::string> pendingWrites;
static std::mutex pendingMutex;

using namespace himan::plugin;

writer::writer() : itsWriteOptions()
{
	itsLogger = logger("writer");
}

void writer::AddToPending(const std::vector<std::string>& names)
{
	std::lock_guard<std::mutex> lock(pendingMutex);
	pendingWrites.reserve(pendingWrites.size() + names.size());
	pendingWrites.insert(pendingWrites.end(), names.begin(), names.end());
}

template <typename T>
std::pair<himan::HPWriteStatus, himan::file_information> writer::CreateFile(
    info<T>& theInfo, std::shared_ptr<const plugin_configuration> conf)
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

template std::pair<himan::HPWriteStatus, himan::file_information> writer::CreateFile<double>(
    info<double>&, std::shared_ptr<const plugin_configuration>);
template std::pair<himan::HPWriteStatus, himan::file_information> writer::CreateFile<float>(
    info<float>&, std::shared_ptr<const plugin_configuration>);

himan::HPWriteStatus writer::ToFile(std::shared_ptr<info<double>> theInfo,
                                    std::shared_ptr<const plugin_configuration> conf)

{
	return ToFile<double>(theInfo, conf);
}

template <typename T>
himan::HPWriteStatus writer::ToFile(std::shared_ptr<info<T>> theInfo, std::shared_ptr<const plugin_configuration> conf)
{
	if (!itsWriteOptions.write_empty_grid)
	{
		if (theInfo->Data().MissingCount() == theInfo->Data().Size())
		{
			itsLogger.Info("Not writing empty grid for param " + theInfo->Param().Name() + " time " +
			               theInfo->Time().OriginDateTime().String() + " step " +
			               static_cast<std::string>(theInfo->Time().Step()) + " level " +
			               static_cast<std::string>(theInfo->Level()));
			return himan::HPWriteStatus::kFailed;
		}
	}

	timer t;

	if (conf->StatisticsEnabled())
	{
		t.Start();
	}

	HPWriteStatus status = HPWriteStatus::kFinished;

	if (conf->WriteMode() != kNoFileWrite)
	{
		// When writing previ to database, no file is needed. In all other cases we have to create
		// a file.

		file_information finfo;

		if (theInfo->Producer().Class() == kGridClass ||
		    (theInfo->Producer().Class() == kPreviClass && conf->WriteToDatabase() == false))
		{
			auto ret = CreateFile<T>(*theInfo, conf);
			status = ret.first;
			finfo = ret.second;
		}

		if (status == HPWriteStatus::kFinished)
		{
			WriteToRadon(conf, finfo, theInfo);
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

	return status;
}

template himan::HPWriteStatus writer::ToFile<double>(std::shared_ptr<info<double>>,
                                                     std::shared_ptr<const plugin_configuration>);
template himan::HPWriteStatus writer::ToFile<float>(std::shared_ptr<info<float>>,
                                                    std::shared_ptr<const plugin_configuration>);

write_options writer::WriteOptions() const
{
	return itsWriteOptions;
}
void writer::WriteOptions(const write_options& theWriteOptions)
{
	itsWriteOptions = theWriteOptions;
}

void writer::WritePendingInfos(std::shared_ptr<const plugin_configuration> conf)
{
	if (conf->WriteStorageType() == kS3ObjectStorageSystem)
	{
		std::lock_guard<std::mutex> lock(pendingMutex);
		itsLogger.Info("Writing " + std::to_string(pendingWrites.size()) + " pending infos to file");

		// The only case when we have pending writes is (currently) when
		// writing to s3

		// First get the infos from cache that match the names given
		// to us by the caller

		auto c = GET_PLUGIN(cache);

		std::vector<std::shared_ptr<himan::info<double>>> infos;
		for (const auto& name : pendingWrites)
		{
			auto ret = c->GetInfo<double>(name);

			if (ret.empty())
			{
				itsLogger.Fatal("Failed to find pending write from cache with key: " + name);
				himan::Abort();
			}

			infos.insert(infos.end(), ret.begin(), ret.end());
		}

		// Next create a grib message of each info and store them sequentially
		// in a buffer. All infos that have the same filename will end up in the
		// same buffer.

		std::map<std::string, himan::buffer> list;
		std::map<std::string, int> count;
		std::vector<std::pair<std::shared_ptr<info<double>>, file_information>> finfos;

		auto g = GET_PLUGIN(grib);
		itsWriteOptions.configuration = conf;
		g->WriteOptions(itsWriteOptions);

		ASSERT(conf);

		for (const auto& info : infos)
		{
			auto ret = g->CreateGribMessage(*info);
			file_information& finfo = ret.first;
			NFmiGribMessage& msg = ret.second;
			const size_t griblength = msg.GetLongKey("totalLength");

			himan::buffer& buff = list[finfo.file_location];
			int& message_no = count[finfo.file_location];

			buff.data = static_cast<unsigned char*>(realloc(buff.data, buff.length + griblength));

			msg.GetMessage(buff.data + buff.length, griblength);

			finfo.offset = buff.length;
			finfo.length = griblength;
			finfo.message_no = message_no;

			buff.length += griblength;
			message_no++;

			finfos.push_back(make_pair(info, finfo));
		}

		// Write the buffers to s3

		for (const auto& p : list)
		{
			s3::WriteObject(p.first, p.second);
		}

		// And finally update radon

		for (const auto& elem : finfos)
		{
			WriteToRadon(conf, elem.second, elem.first);
		}
	}
	else if (pendingWrites.empty() == false)
	{
		itsLogger.Fatal(
		    "Pending write started with invalid conditions: write_mode=" + HPWriteModeToString.at(conf->WriteMode()) +
		    ", storage_type=" + HPFileStorageTypeToString.at(conf->WriteStorageType()));
		himan::Abort();
	}
}

template <typename T>
bool writer::WriteToRadon(std::shared_ptr<const plugin_configuration> conf, const file_information& finfo,
                          std::shared_ptr<info<T>> theInfo)
{
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
					return false;
				}
			}
			catch (const std::exception& e)
			{
				itsLogger.Error("Writing to radon failed: " + std::string(e.what()));
				return false;
			}
			catch (...)
			{
				itsLogger.Error("Writing to radon failed: general exception");
				return false;
			}
		}
	}

	return true;
}

template bool writer::WriteToRadon(std::shared_ptr<const plugin_configuration>, const file_information&,
                                   std::shared_ptr<info<double>>);
