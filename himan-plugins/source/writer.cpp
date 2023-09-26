#include "writer.h"
#include "logger.h"
#include "plugin_factory.h"
#include "s3.h"
#include "statistics.h"
#include "timer.h"
#include "util.h"
#include <fstream>
#include <thread>

#include "cache.h"
#include "csv.h"
#include "geotiff.h"
#include "grib.h"
#include "querydata.h"
#include "radon.h"

static std::vector<std::pair<std::string, himan::HPWriteStatus>> writeStatuses;
static std::mutex writeStatusMutex;

using namespace himan::plugin;

writer::writer() : itsWriteOptions()
{
	itsLogger = logger("writer");
}

void writer::ClearPending()
{
	auto c = GET_PLUGIN(cache);
	std::lock_guard<std::mutex> lock(writeStatusMutex);
	for (const auto& m : writeStatuses)
	{
		if (m.second == himan::HPWriteStatus::kPending)
		{
			c->Remove(m.first);
		}
	}
	writeStatuses.erase(std::remove_if(writeStatuses.begin(), writeStatuses.end(),
	                                   [](const std::pair<std::string, himan::HPWriteStatus>& v)
	                                   { return v.second == himan::HPWriteStatus::kPending; }),
	                    writeStatuses.end());
}

void ReadConfigurationWriteOptions(write_options& writeOptions)
{
	// Other options in write_options struct could also be checked here,
	// but they are really not that interesting hence I'm skipping them
	// at this stage.

	const std::string precision = writeOptions.configuration->GetValue("write_options.precision");

	if (precision.empty() == false)
	{
		writeOptions.precision = std::stoi(precision);
	}

	std::string extra = writeOptions.configuration->GetValue("write_options.extra_metadata");

	if (extra.empty())
	{
		extra = writeOptions.configuration->GetValue("extra_file_metadata");
	}

	if (extra.empty() == false)
	{
		himan::logger logr("writer");
		std::vector<std::pair<std::string, std::string>> options;
		const auto list = himan::util::Split(extra, ",");
		for (const auto& e : list)
		{
			const auto kv = himan::util::Split(e, "=");
			if (kv.size() != 2)
			{
				logr.Warning(fmt::format("Invalid extra_file_metadata option: {}", e));
				continue;
			}

			options.emplace_back(kv[0], kv[1]);
		}

		writeOptions.extra_metadata = options;
	}
}

template <typename T>
std::pair<himan::HPWriteStatus, himan::file_information> writer::CreateFile(info<T>& theInfo)
{
	// do not modify write configuration of fetcher instance,
	// as it may be shared among many writes
	auto wo = itsWriteOptions;

	ReadConfigurationWriteOptions(wo);

	switch (wo.configuration->OutputFileType())
	{
		case kGRIB:
		case kGRIB1:
		case kGRIB2:
		{
			auto theGribWriter = GET_PLUGIN(grib);

			theGribWriter->WriteOptions(wo);
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
			theWriter->WriteOptions(wo);

			return theWriter->ToFile<T>(theInfo);
		}

		case kCSV:
		{
			auto theWriter = GET_PLUGIN(csv);
			theWriter->WriteOptions(wo);

			return theWriter->ToFile<T>(theInfo);
		}

		case kGeoTIFF:
		{
			auto theWriter = GET_PLUGIN(geotiff);
			theWriter->WriteOptions(wo);

			return theWriter->ToFile<T>(theInfo);
		}
		// Must have this or compiler complains
		case kNetCDF:
		case kNetCDFv4:
		default:
			itsLogger.Error(
			    fmt::format("Invalid file type: {}", HPFileTypeToString.at(wo.configuration->OutputFileType())));
			break;
	}

	throw kInvalidWriteOptions;
}

template std::pair<himan::HPWriteStatus, himan::file_information> writer::CreateFile<double>(info<double>&);
template std::pair<himan::HPWriteStatus, himan::file_information> writer::CreateFile<float>(info<float>&);

himan::HPWriteStatus writer::ToFile(std::shared_ptr<info<double>> theInfo,
                                    std::shared_ptr<const plugin_configuration> conf)

{
	return ToFile<double>(theInfo, conf);
}

template <typename T>
himan::HPWriteStatus writer::ToFile(std::shared_ptr<info<T>> theInfo, std::shared_ptr<const plugin_configuration> conf)
{
	itsWriteOptions.configuration = conf;

	bool writeEmptyGrid = itsWriteOptions.write_empty_grid;

	if (conf->Exists("write_empty_grid"))
	{
		writeEmptyGrid = util::ParseBoolean(conf->GetValue("write_empty_grid"));
	}

	if (writeEmptyGrid == false && theInfo->Data().MissingCount() == theInfo->Data().Size())
	{
		itsLogger.Info(fmt::format("Not writing empty grid for param {} time {} step {} level {}",
		                           theInfo->Param().Name(), theInfo->Time().OriginDateTime().String(),
		                           static_cast<std::string>(theInfo->Time().Step()),
		                           static_cast<std::string>(theInfo->Level())));
		return himan::HPWriteStatus::kFailed;
	}

	const size_t allowedMissing = itsWriteOptions.configuration->AllowedMissingValues();
	if (allowedMissing < std::numeric_limits<size_t>::max() && allowedMissing < theInfo->Data().MissingCount())
	{
		itsLogger.Fatal(fmt::format("Parameter {} for leadtime {} contains more missing values ({}) than allowed ({})",
		                            theInfo->Param().Name(), theInfo->Time().Step(), theInfo->Data().MissingCount(),
		                            allowedMissing));
		exit(1);
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
			auto ret = CreateFile<T>(*theInfo);
			status = ret.first;
			finfo = ret.second;
		}

		if (status == HPWriteStatus::kFinished)
		{
			auto ret = WriteToRadon(conf, finfo, theInfo);

			if (ret.first)
			{
				conf->Statistics()->AddToSummaryRecords(summary_record(finfo, ret.second, theInfo->Producer(),
				                                                       theInfo->ForecastType(), theInfo->Time(),
				                                                       theInfo->Level(), theInfo->Param()));
			}
		}
	}

	if (conf->UseCacheForWrites())
	{
		auto c = GET_PLUGIN(cache);

		// Pin those items that
		// * are not written to file at all
		// * are written in s3 at the end of the execution
		//
		// so they can't be removed from cache if cache size is limited
		c->Insert<T>(theInfo,
		             (conf->WriteMode() == kNoFileWrite) || (conf->WriteStorageType() == kS3ObjectStorageSystem));
	}

	const std::string uName = util::UniqueName<T>(*theInfo);

	{
		std::lock_guard<std::mutex> lock(writeStatusMutex);
		writeStatuses.push_back(make_pair(uName, status));
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
	auto FetchPendingFromCache =
	    [](const std::vector<std::string>& pending) -> std::vector<std::shared_ptr<himan::info<double>>>
	{
		auto c = GET_PLUGIN(cache);
		logger logr("writer");

		std::vector<std::shared_ptr<himan::info<double>>> infos;
		for (const auto& name : pending)
		{
			auto ret = c->GetInfo<double>(name);

			if (ret.empty())
			{
				logr.Fatal(fmt::format("Failed to find pending write from cache with key: {}", name));
				himan::Abort();
			}

			infos.insert(infos.end(), ret.begin(), ret.end());
		}
		return infos;
	};

	std::vector<std::string> pendingWrites;

	for (const auto& m : writeStatuses)
	{
		if (m.second == himan::HPWriteStatus::kPending)
		{
			pendingWrites.push_back(m.first);
		}
	}

	if (conf->WriteStorageType() == kS3ObjectStorageSystem &&
	    (conf->OutputFileType() == kGRIB || conf->OutputFileType() == kGRIB1 || conf->OutputFileType() == kGRIB2))
	{
		std::lock_guard<std::mutex> lock(writeStatusMutex);
		itsLogger.Info(fmt::format("Writing {} pending infos to file", pendingWrites.size()));

		// The only case when we have pending writes is (currently) when
		// writing to s3

		// First get the infos from cache that match the names given
		// to us by the caller

		auto infos = FetchPendingFromCache(pendingWrites);

		// Next create a grib message of each info and store them sequentially
		// in a buffer. All infos that have the same filename will end up in the
		// same buffer.

		std::map<std::string, himan::buffer> list;
		std::map<std::string, int> count;
		std::vector<std::pair<std::shared_ptr<info<double>>, file_information>> finfos;

		itsWriteOptions.configuration = conf;

		ASSERT(conf);

		// Building grib messages from infos is time consuming, mostly cpu intensive
		// work. Parallelize it.

		std::vector<std::thread> threads;
		std::mutex m_;

		std::atomic_int infonum{-1};

		auto GetInfo = [&]() -> std::shared_ptr<info<double>>
		{
			int myinfonum = ++infonum;

			if (myinfonum >= static_cast<int>(infos.size()))
			{
				return nullptr;
			}
			return infos[myinfonum];
		};

		auto ProcessInfo = [&](std::shared_ptr<info<double>> info)
		{
			auto g = GET_PLUGIN(grib);
			g->WriteOptions(itsWriteOptions);

			// create grib message
			auto ret = g->CreateGribMessage(*info);
			NFmiGribMessage& msg = ret.second;
			const size_t griblength = msg.GetLongKey("totalLength");
			file_information& finfo = ret.first;

			{
				std::lock_guard<std::mutex> lockb(m_);
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
		};

		int threadCount = conf->ThreadCount();

		if (threadCount == -1)
		{
			threadCount = 4;
		}

		itsLogger.Trace(fmt::format("Starting {} threads to convert infos to grib messages", threadCount));
		timer tmr(true);

		for (size_t i = 0; i < static_cast<size_t>(threadCount); i++)
		{
			threads.push_back(std::thread(
			    [&]()
			    {
				    logger logr("grib");
				    while (true)
				    {
					    auto info = GetInfo();
					    if (info == nullptr)
					    {
						    break;
					    }
					    logr.Trace(fmt::format("Creating grib message from '{}'", util::UniqueName(*info)));

					    ProcessInfo(info);
				    }
			    }));
		}

		for (auto& t : threads)
		{
			t.join();
		}
		tmr.Stop();

		itsLogger.Debug(fmt::format("Converted {} infos to gribs in {:.1f}s", infos.size(),
		                            static_cast<float>(tmr.GetTime()) / 1000.f));

		// Write the buffers to s3

		for (const auto& p : list)
		{
			s3::WriteObject(p.first, p.second);
		}

		// And finally update radon

		for (const auto& elem : finfos)
		{
			const auto& finfo = elem.second;
			const auto& info = elem.first;

			auto ret = WriteToRadon(conf, finfo, info);

			if (ret.first)
			{
				conf->Statistics()->AddToSummaryRecords(summary_record(finfo, ret.second, info->Producer(),
				                                                       info->ForecastType(), info->Time(),
				                                                       info->Level(), info->Param()));
			}
		}
	}
	else if (conf->OutputFileType() == kGeoTIFF)
	{
		std::lock_guard<std::mutex> lock(writeStatusMutex);
		itsLogger.Info(fmt::format("Writing {} pending infos to file", pendingWrites.size()));
		auto infos = FetchPendingFromCache(pendingWrites);

		auto g = GET_PLUGIN(geotiff);
		itsWriteOptions.configuration = conf;
		g->WriteOptions(itsWriteOptions);

		std::vector<info<double>> plain;
		for (const auto& x : infos)
		{
			plain.push_back(*x);
		}

		auto finfos = g->ToFile<double>(plain);

		for (size_t i = 0; i < finfos.size(); i++)
		{
			const auto& finfo = finfos[i].second;
			const auto& info = infos[i];

			if (finfos[i].first == HPWriteStatus::kFinished)
			{
				auto ret = WriteToRadon(conf, finfo, info);

				if (ret.first)
				{
					conf->Statistics()->AddToSummaryRecords(summary_record(finfo, ret.second, info->Producer(),
					                                                       info->ForecastType(), info->Time(),
					                                                       info->Level(), info->Param()));
				}
			}
		}
	}
	else if (pendingWrites.empty() == false)
	{
		itsLogger.Fatal(fmt::format("Pending write started with invalid conditions: write_mode: {}, storage_type: {}",
		                            HPWriteModeToString.at(conf->WriteMode()),
		                            HPFileStorageTypeToString.at(conf->WriteStorageType())));
		himan::Abort();
	}
}

template <typename T>
std::pair<bool, radon_record> writer::WriteToRadon(std::shared_ptr<const plugin_configuration> conf,
                                                   const file_information& finfo, std::shared_ptr<info<T>> theInfo)
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
				auto ret = r->Save<T>(*theInfo, finfo, conf->TargetGeomName());

				if (!ret.first)
				{
					itsLogger.Error("Writing to radon failed");
					return ret;
				}

				return ret;
			}
			catch (const std::exception& e)
			{
				itsLogger.Error(fmt::format("Writing to radon failed: {}", e.what()));
				return std::make_pair(false, radon_record());
			}
		}
	}

	return std::make_pair(true, radon_record());
}

template std::pair<bool, radon_record> writer::WriteToRadon(std::shared_ptr<const plugin_configuration>,
                                                            const file_information&, std::shared_ptr<info<double>>);
