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
#include <filesystem>

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

#ifdef HAVE_CEREAL
std::string SpillDirectory()
{
	std::string tmpdir = "/tmp";

	try
	{
		tmpdir = himan::util::GetEnv("HIMAN_TEMP_DIRECTORY");
	}
	catch (...)
	{
	}

	tmpdir = fmt::format("{}/himan-spill-{}", tmpdir, getpid());

	if (std::filesystem::is_directory(tmpdir) == false)
	{
		std::filesystem::create_directory(tmpdir);
	}

	return tmpdir;
}

template <typename T>
std::string writer::SpillToDisk(std::shared_ptr<himan::info<T>> info)
{
	const std::string spillFile = fmt::format("{}/{:010d}", SpillDirectory(), rand());
	std::ofstream outfile(spillFile, std::ios::binary);
	cereal::BinaryOutputArchive archive(outfile);

	// extract only currently active info and set file type to double
	auto newInfo =
	    std::make_shared<himan::info<double>>(info->ForecastType(), info->Time(), info->Level(), info->Param());
	newInfo->Producer(info->Producer());

	auto b = std::make_shared<himan::base<double>>();
	b->grid = std::shared_ptr<himan::grid>(info->Grid()->Clone());
	b->data = info->Data();
	newInfo->Base(b);
	archive(newInfo);
	itsLogger.Debug(fmt::format("Cache is full, spilling {} to file {}", util::UniqueName(*info), spillFile));

	return spillFile;
}
#endif

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

	// write status name, either file name if data was spilled to disk,
	// or unique name of the info otherwise
	std::string wsName = util::UniqueName(*theInfo);

	if (conf->UseCacheForWrites())
	{
		auto c = GET_PLUGIN(cache);

		// Pin those items that
		// * are not written to file at all
		// * are written in s3 at the end of the execution
		//
		// so they can't be removed from cache if cache size is limited
		auto ret = c->Insert<T>(
		    theInfo, (conf->WriteMode() == kNoFileWrite) || (conf->WriteStorageType() == kS3ObjectStorageSystem));
#ifdef HAVE_CEREAL
		if (ret == HPWriteStatus::kFailed)
		{
			status = HPWriteStatus::kSpilled;
			wsName = SpillToDisk(theInfo);
		}
#endif
	}

	{
		{
			std::lock_guard<std::mutex> lock(writeStatusMutex);
			writeStatuses.push_back(make_pair(wsName, status));
		}
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

std::vector<write_information> writer::WritePendingGribs(const std::vector<std::shared_ptr<himan::info<double>>>& infos)
{
	// Next create a grib message of each info and store them sequentially
	// in a buffer. All infos that have the same filename will end up in the
	// same buffer.

	std::map<std::string, himan::buffer> list;
	std::map<std::string, int> count;
	std::vector<std::pair<std::shared_ptr<info<double>>, file_information>> finfos;

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

	int threadCount = itsWriteOptions.configuration->ThreadCount();

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

	std::vector<write_information> ret;

	for (size_t i = 0; i < infos.size(); i++)
	{
		ret.push_back(std::make_tuple(HPWriteStatus::kFinished, finfos[i].second, finfos[i].first));
	}

	return ret;
}

std::vector<write_information> writer::WritePendingGeotiffs(
    const std::vector<std::shared_ptr<himan::info<double>>>& infos)
{
	auto g = GET_PLUGIN(geotiff);
	g->WriteOptions(itsWriteOptions);

	std::vector<info<double>> plain;
	for (const auto& x : infos)
	{
		plain.push_back(*x);
	}

	auto finfos = g->ToFile<double>(plain);

	std::vector<write_information> ret;

	for (size_t i = 0; i < infos.size(); i++)
	{
		ret.push_back(std::make_tuple(finfos[i].first, finfos[i].second, infos[i]));
	}

	return ret;
}

void writer::WritePendingToRadon(std::vector<write_information>& list)
{
	for (const auto& elem : list)
	{
		const HPWriteStatus status = std::get<0>(elem);
		const file_information& finfo = std::get<1>(elem);
		const std::shared_ptr<himan::info<double>>& info = std::get<2>(elem);

		if (status == HPWriteStatus::kFinished)
		{
			auto& conf = itsWriteOptions.configuration;
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

void writer::WritePendingInfos(std::shared_ptr<const plugin_configuration> conf)
{
	itsWriteOptions.configuration = conf;

	ASSERT(conf);

	using std::string, std::vector;
	auto Filter = [](const vector<std::pair<string, himan::HPWriteStatus>>& ws,
	                 himan::HPWriteStatus req) -> vector<string>
	{
		vector<string> ret;
		for (const auto& m : ws)
		{
			if (m.second == req)
			{
				ret.push_back(m.first);
			}
		}
		return ret;
	};

	auto FetchPendingFromCache = [&Filter]() -> vector<std::shared_ptr<himan::info<double>>>
	{
		vector<string> pending = Filter(writeStatuses, himan::HPWriteStatus::kPending);

		auto c = GET_PLUGIN(cache);
		logger logr("writer");

		vector<std::shared_ptr<himan::info<double>>> infos;
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

#ifdef HAVE_CEREAL
	auto FetchSpilledFromDisk = [&Filter]() -> vector<std::shared_ptr<himan::info<double>>>
	{
		vector<string> spilled = Filter(writeStatuses, himan::HPWriteStatus::kSpilled);

		// read serialized infos from disk and de-serialize (is there a word for that?)
		// remove the files after they are read to memory
		vector<std::shared_ptr<himan::info<double>>> infos;
		logger logr("writer");
		for (const auto& name : spilled)
		{
			logr.Trace(fmt::format("De-serializing spilled file {}", name));
			auto info = std::make_shared<himan::info<double>>();
			{
				std::ifstream infile(name, std::ios::binary);
				cereal::BinaryInputArchive iarchive(infile);

				iarchive(info);
			}
			infos.push_back(info);

			std::filesystem::remove(name);
		}
		if (spilled.size() > 0)
		{
			std::filesystem::remove_all(std::filesystem::path{spilled[0]}.parent_path().string());
		}
		return infos;
	};
#endif

	std::vector<std::shared_ptr<himan::info<double>>> infos;
	size_t pendingSize = 0;
#ifdef HAVE_CEREAL
	size_t spilledSize = 0;  // just for logging
#endif

	{
		std::lock_guard<std::mutex> lock(writeStatusMutex);

		// The only case when we have pending writes is (currently) when
		// writing to s3

		// First get the infos from cache that match the names given
		// to us by the caller

		infos = FetchPendingFromCache();
		pendingSize = infos.size();

#ifdef HAVE_CEREAL
		auto spilledInfos = FetchSpilledFromDisk();

		auto c = GET_PLUGIN(cache);
		c->Clean(CleanType::kAll);

		infos.insert(infos.end(), spilledInfos.begin(), spilledInfos.end());
		spilledSize = spilledInfos.size();
#endif
	}

	if (infos.size() == 0)
	{
		return;
	}

#ifdef HAVE_CEREAL
	itsLogger.Info(fmt::format("Writing {} pending and {} spilled infos to file", pendingSize, spilledSize));
#else
	itsLogger.Info(fmt::format("Writing {} pending infos to file", pendingSize));
#endif

	if (conf->WriteStorageType() == kS3ObjectStorageSystem &&
	    (conf->OutputFileType() == kGRIB || conf->OutputFileType() == kGRIB1 || conf->OutputFileType() == kGRIB2))
	{
		auto ret = WritePendingGribs(infos);

		WritePendingToRadon(ret);
	}
	else if (conf->OutputFileType() == kGeoTIFF)
	{
		auto ret = WritePendingGeotiffs(infos);

		WritePendingToRadon(ret);
	}
	else
	{
		auto pending = Filter(writeStatuses, HPWriteStatus::kPending);

		if (pending.empty() == false)
		{
			itsLogger.Fatal(fmt::format(
			    "Pending write started with invalid conditions: write_mode: {}, storage_type: {}",
			    HPWriteModeToString.at(conf->WriteMode()), HPFileStorageTypeToString.at(conf->WriteStorageType())));
			himan::Abort();
		}
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
