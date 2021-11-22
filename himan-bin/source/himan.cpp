/**
 * @file himan.cpp
 *
 * @brief himan main program
 *
 */

#include "auxiliary_plugin.h"
#include "compiled_plugin.h"
#include "cuda_helper.h"
#include "himan_common.h"
#include "himan_plugin.h"
#include "json_parser.h"
#include "logger.h"
#include "plugin_factory.h"
#include "radon.h"
#include "statistics.h"
#include "timer.h"
#include "util.h"
#include "writer.h"
#include <boost/program_options.hpp>
#include <future>
#include <iostream>
#include <vector>

using namespace himan;
using namespace std;

void banner();
void ParseCommandLine(shared_ptr<configuration>& conf, int argc, char** argv);
shared_ptr<configuration> ReadEnvironment();

struct plugin_timing
{
	std::string plugin_name;
	int order_number;      // plugin order number (if called more than once))
	int64_t time_elapsed;  // elapsed time in ms
};

void UploadRunStatisticsToDatabase(const shared_ptr<configuration>& conf, const vector<plugin_timing>& pluginTimes)
{
	stringstream json, query;

	const string content = conf->ConfigurationFileContent();
	string configname;

	// read configuration file
	if (conf->ConfigurationFileName() == "-")
	{
		configname = conf->StatisticsLabel();
	}
	else
	{
		configname = conf->ConfigurationFileName();
	}

	// create json out of timings
	json << "{ \"plugins\" : [";

	for (const auto& t : pluginTimes)
	{
		string name = t.plugin_name;

		if (t.order_number > 1)
		{
			name += " #" + to_string(t.order_number);
		}

		json << " { \"name\" : \"" << name << "\", \"elapsed_ms\" : \"" << (t.time_elapsed) << "\" },";
	}

	json.seekp(-1, json.cur);  // remove comma from last element
	json << " ] }";

	char* host = getenv("HOSTNAME");

	auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
	char timestr[80];
	strftime(timestr, 80, "%Y-%m-%d %H:%M:%S %Z", localtime(&now));

	// clang-format off

	query << "INSERT INTO himan_run_statistics (hostname, finish_time, configuration_name, configuration, statistics) VALUES ("
	      << "'" << host << "', "
	      << "'" << timestr << "', "
	      << "'" << configname << "', "
	      << "'" << content << "'::json, "
	      << "'" << json.str() << "'::json)";

	// clang-format on

	auto r = GET_PLUGIN(radon);

	logger log("himan");

	try
	{
		r->RadonDB().Query(query.str());
	}
#if PQXX_VERSION_MAJOR < 7
	catch (const pqxx::pqxx_exception& e)
	{
		log.Error(e.base().what());
	}
#else
	catch (const pqxx::failure& e)
	{
		log.Error(e.what());
	}
#endif
	catch (const exception& e)
	{
		log.Error(e.what());
	}
}

int HighestOrderNumber(const vector<plugin_timing>& timingList, const std::string& pluginName)
{
	int highest = 1;

	for (const auto& timing : timingList)
	{
		if (timing.plugin_name == pluginName && timing.order_number >= highest)
		{
			highest = timing.order_number + 1;
		}
	}

	return highest;
}

void UpdateSSState(const shared_ptr<const plugin_configuration>& pc)
{
	logger log("himan");

	if (pc->DatabaseType() != kRadon || pc->WriteToDatabase() == false || pc->UpdateSSStateTable() == false)
	{
		log.Trace("ss_state table update disabled");
		return;
	}

	const auto& summaryRecords = pc->Statistics()->SummaryRecords();

	auto r = GET_PLUGIN(radon);

	int inserts = 0, updates = 0;

	for (const auto& record : summaryRecords)
	{
		const auto& producerId = record.producer.Id();
		const auto& geometryId = record.rrecord.geometry_id;
		const auto& analysisTime = record.ftime.OriginDateTime().String();
		const auto& period = util::MakeSQLInterval(record.ftime);
		const auto& forecastTypeId = record.ftype.Type();
		const auto& forecastTypeValue = (forecastTypeId >= 3 && forecastTypeId <= 4) ? record.ftype.Value() : -1;
		const auto& tableName = pc->SSStateTableName().empty()
		                            ? fmt::format("{}.{}", record.rrecord.schema_name, record.rrecord.table_name)
		                            : pc->SSStateTableName();

		try
		{
			const string query = fmt::format(
			    "INSERT INTO ss_state (producer_id, geometry_id, analysis_time, forecast_period, forecast_type_id, "
			    "forecast_type_value, table_name) VALUES ({}, {}, '{}', '{}', {}, {}, '{}')",
			    producerId, geometryId, analysisTime, period, forecastTypeId, forecastTypeValue, tableName);

			r->RadonDB().Execute(query);
			inserts++;
		}
		catch (const pqxx::unique_violation& e)
		{
			const string query = fmt::format(
			    "UPDATE ss_state SET table_name = '{}', last_updated = now() WHERE producer_id = {} AND geometry_id = "
			    "{} AND analysis_time = '{}' AND forecast_period = '{}' AND forecast_type_id = '{}' AND "
			    "forecast_type_value = '{}'",
			    tableName, producerId, geometryId, analysisTime, period, forecastTypeId, forecastTypeValue);

			r->RadonDB().Execute(query);
			updates++;
		}
		catch (const exception& e)
		{
			log.Error("Caught unexpected exception: " + string(e.what()));
			return;
		}

		r->RadonDB().Commit();
	}

	log.Debug(fmt::format("Update of ss_state: {} inserts, {} updates", inserts, updates));
}

void ExecutePlugin(const shared_ptr<plugin_configuration>& pc, vector<plugin_timing>& pluginTimes)
{
	timer aTimer(true);
	logger aLogger("himan");

	auto aPlugin = dynamic_pointer_cast<plugin::compiled_plugin>(plugin_factory::Instance()->Plugin(pc->Name()));

	if (!aPlugin)
	{
		aLogger.Error("Unable to declare plugin " + pc->Name());
		return;
	}

	aLogger.Info("Calculating " + pc->Name());

	try
	{
		aPlugin->Process(pc);
	}
	catch (const exception& e)
	{
		aLogger.Fatal(string("Caught exception: ") + e.what());
		exit(1);
	}

	if (pc->StatisticsEnabled())
	{
		aTimer.Stop();
		const auto totalTime = aTimer.GetTime();
		pc->Statistics()->AddToTotalTime(totalTime);
		pc->WriteStatistics();

		plugin_timing t;
		t.plugin_name = pc->Name();
		t.time_elapsed = totalTime;
		t.order_number = HighestOrderNumber(pluginTimes, pc->Name());

		pluginTimes.push_back(t);
	}

#if defined DEBUG and defined HAVE_CUDA
	// For 'cuda-memcheck --leak-check full'
	CUDA_CHECK(cudaDeviceReset());
#endif

	if (pc->WriteToObjectStorageBetweenPluginCalls())
	{
		auto w = GET_PLUGIN(writer);
		w->WritePendingInfos(pc);
		plugin::writer::ClearPending();
		UpdateSSState(pc);
	}
	else if (pc->WriteStorageType() != kS3ObjectStorageSystem)
	{
		UpdateSSState(pc);
	}
}

int main(int argc, char** argv)
{
	shared_ptr<configuration> conf;

	SignalHandlerInit();

	try
	{
		conf = ReadEnvironment();
		ParseCommandLine(conf, argc, argv);
		conf->ProgramName(kHiman);
	}
	catch (const std::exception& e)
	{
		cerr << e.what() << endl;
		exit(1);
	}

	logger aLogger = logger("himan");

	/*
	 * Initialize plugin factory before parsing configuration file. This prevents himan from
	 * terminating suddenly with SIGSEGV on RHEL5 environments.
	 */

	shared_ptr<plugin::auxiliary_plugin> c =
	    dynamic_pointer_cast<plugin::auxiliary_plugin>(plugin_factory::Instance()->Plugin("cache"));

	std::vector<shared_ptr<plugin_configuration>> plugins;

	try
	{
		json_parser parser;
		plugins = parser.Parse(conf);
	}
	catch (std::runtime_error& e)
	{
		aLogger.Fatal(e.what());
		exit(1);
	}

	banner();

	aLogger.Debug("Processqueue size: " + std::to_string(plugins.size()));

	vector<future<void>> asyncs;
	vector<plugin_timing> pluginTimes;
	shared_ptr<const plugin_configuration> lastConf;  // For pending writes we need to have some plugin's configuration

	while (plugins.size() > 0)
	{
		auto pc = plugins[0];

		plugins.erase(plugins.begin());

		if (plugins.size() == 0)
		{
			lastConf = pc;
		}

		if (pc->AsyncExecution())
		{
			aLogger.Info("Asynchronous launch for " + pc->Name());
			asyncs.push_back(async(
			    launch::async,
			    [&pluginTimes](shared_ptr<plugin_configuration> _pc) { ExecutePlugin(_pc, pluginTimes); }, pc));

			continue;
		}

		ExecutePlugin(pc, pluginTimes);
	}

	for (auto& fut : asyncs)
	{
		fut.wait();
	}

	if (lastConf->WriteStorageType() == kS3ObjectStorageSystem &&
	    lastConf->WriteToObjectStorageBetweenPluginCalls() == false)
	{
		auto w = GET_PLUGIN(writer);
		w->WritePendingInfos(lastConf);
		UpdateSSState(lastConf);
	}

	if (!conf->StatisticsLabel().empty())
	{
		// bubble sort

		bool passed;

		do
		{
			passed = true;

			for (size_t i = 1; i < pluginTimes.size(); i++)
			{
				plugin_timing prev = pluginTimes[i - 1];
				plugin_timing cur = pluginTimes[i];

				if (prev.time_elapsed < cur.time_elapsed)
				{
					pluginTimes[i - 1] = cur;
					pluginTimes[i] = prev;
					passed = false;
				}
			}
		} while (!passed);

		int64_t totalTime = 0;

		for (const auto& time : pluginTimes)
		{
			totalTime += time.time_elapsed;
		}

		cout << endl << "*** TOTAL timings for himan ***" << endl;

		for (size_t i = 0; i < pluginTimes.size(); i++)
		{
			plugin_timing t = pluginTimes[i];

			// c++ string formatting really is unnecessarily hard
			stringstream ss;

			ss << t.plugin_name;

			if (t.order_number > 1)
			{
				ss << " #" << t.order_number;
			}

			cout << setw(25) << left << ss.str();

			ss.str("");

			ss << "("
			   << static_cast<int>(((static_cast<double>(t.time_elapsed) / static_cast<double>(totalTime)) * 100))
			   << "%)";

			cout << setw(8) << right << t.time_elapsed << " ms " << setw(5) << right << ss.str() << endl;
		}

		cout << "-------------------------------------------" << endl;
		cout << setw(25) << left << "Total duration:" << setw(8) << right << totalTime << " ms" << endl;

		if (conf->DatabaseType() == kRadon && conf->WriteToDatabase())
		{
			UploadRunStatisticsToDatabase(conf, pluginTimes);
		}
	}

	return 0;
}

void banner()
{
	cout << endl
	     << "************************************************" << endl
	     << "* By the Power of Grayskull, I Have the Power! *" << endl
	     << "************************************************" << endl
	     << endl;
}
#ifdef HAVE_CUDA
void CudaCapabilities()
{
	int devCount;

	cudaError_t err = cudaGetDeviceCount(&devCount);

	if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver)
	{
		// No device or no driver present

		devCount = 0;
	}
	else if (err != cudaSuccess)
	{
		std::cout << "cudaGetDeviceCount() returned error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	if (devCount == 0)
	{
		std::cout << "No CUDA devices found" << std::endl;
		return;
	}

	int runtimeVersion, libraryVersion;

	CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
	CUDA_CHECK(cudaDriverGetVersion(&libraryVersion));

	std::cout << "#----------------------------------------------#" << std::endl;
	std::cout << "CUDA library version " << libraryVersion / 1000 << "." << (libraryVersion % 100) / 10 << std::endl;
	std::cout << "CUDA runtime version " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
	std::cout << "There are " << devCount << " CUDA device(s)" << std::endl;
	std::cout << "#----------------------------------------------#" << std::endl;

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		std::cout << "CUDA Device #" << i << std::endl;

		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);

		std::cout << "Major revision number:\t\t" << devProp.major << std::endl
		          << "Minor revision number:\t\t" << devProp.minor << std::endl
		          << "Device name:\t\t\t" << devProp.name << std::endl
		          << "Total global memory:\t\t" << devProp.totalGlobalMem << std::endl
		          << "Total shared memory per block:\t" << devProp.sharedMemPerBlock << std::endl
		          << "Total registers per block:\t" << devProp.regsPerBlock << std::endl
		          << "Warp size:\t\t\t" << devProp.warpSize << std::endl
		          << "Maximum memory pitch:\t\t" << devProp.memPitch << std::endl
		          << "Maximum threads per block:\t" << devProp.maxThreadsPerBlock << std::endl;

		for (int j = 0; j < 3; ++j)
		{
			std::cout << "Maximum dimension " << j << " of block:\t" << devProp.maxThreadsDim[j] << std::endl;
		}

		for (int j = 0; j < 3; ++j)
		{
			std::cout << "Maximum dimension " << j << " of grid:\t" << devProp.maxGridSize[j] << std::endl;
		}

		std::cout << "Clock rate:\t\t\t" << devProp.clockRate << std::endl
		          << "Total constant memory:\t\t" << devProp.totalConstMem << std::endl
		          << "Texture alignment:\t\t" << devProp.textureAlignment << std::endl
		          << "Concurrent copy and execution:\t" << (devProp.deviceOverlap ? "Yes" : "No") << std::endl
		          << "Number of multiprocessors:\t" << devProp.multiProcessorCount << std::endl
		          << "Kernel execution timeout:\t" << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl
		          << std::endl;
	}
	std::cout << "#----------------------------------------------#" << std::endl;
}
#endif

void SetConfigurationFile(shared_ptr<configuration>& conf, const string& confFile)
{
	conf->ConfigurationFileName(confFile);
	if (confFile == "-")
	{
		stringstream ss;
		for (string line; getline(cin, line);)
		{
			ss << line;
		}
		conf->ConfigurationFileContent(ss.str());
	}
	else
	{
		ifstream ifs(confFile);
		string content = string((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));
		conf->ConfigurationFileContent(content);
	}
}

void SetOutputFileType(shared_ptr<configuration>& conf, const string& outfileType)
{
	if (outfileType == "grib")
	{
		conf->OutputFileType(kGRIB1);
	}
	else if (outfileType == "grib2")
	{
		conf->OutputFileType(kGRIB2);
	}
	else if (outfileType == "netcdf")
	{
		conf->OutputFileType(kNetCDF);
	}
	else if (outfileType == "querydata")
	{
		conf->OutputFileType(kQueryData);
	}
	else if (outfileType == "csv")
	{
		conf->OutputFileType(kCSV);
	}
	else if (outfileType == "geotiff")
	{
		conf->OutputFileType(kGeoTIFF);
	}
	else
	{
		cerr << "Invalid file type: " << outfileType << endl;
		himan::Abort();
	}
}

void SetCompression(shared_ptr<configuration>& conf, const string& outfileCompression)
{
	if (outfileCompression == "gz")
	{
		conf->FileCompression(kGZIP);
	}
	else if (outfileCompression == "bzip2")
	{
		conf->FileCompression(kBZIP2);
	}
	else
	{
		cerr << "Invalid file compression type: " << outfileCompression << endl;
		himan::Abort();
	}
}

void SetLogLevel(shared_ptr<configuration>& conf, int logLevel)
{
	himan::HPDebugState debugState = himan::kDebugMsg;

	switch (logLevel)
	{
		default:
			cerr << "Invalid debug level: " << logLevel << endl;
			exit(1);
		case 0:
			debugState = kFatalMsg;
			break;
		case 1:
			debugState = kErrorMsg;
			break;
		case 2:
			debugState = kWarningMsg;
			break;
		case 3:
			debugState = kInfoMsg;
			break;
		case 4:
			debugState = kDebugMsg;
			break;
		case 5:
			debugState = kTraceMsg;
			break;
	}

	logger::MainDebugState = debugState;
}

void SetCudaDeviceId(shared_ptr<configuration>& conf, int cudaDeviceId)
{
#ifdef HAVE_CUDA

	if (cudaDeviceId >= conf->CudaDeviceCount() || cudaDeviceId < 0)
	{
		cerr << "cuda device id " << cudaDeviceId << " requested, allowed maximum cuda device id is "
		     << conf->CudaDeviceCount() - 1 << endl;
		cerr << "cuda mode is disabled" << endl;
		conf->UseCuda(false);
		conf->UseCudaForPacking(false);
	}

	conf->CudaDeviceId(cudaDeviceId);
#endif
}

shared_ptr<configuration> ReadEnvironment()
{
	const vector<string> keys{"HIMAN_OUTPUT_FILE_TYPE",
	                          "HIMAN_COMPRESSION",
	                          "HIMAN_CONFIGURATION_FILE",
	                          "HIMAN_THREADS",
	                          "HIMAN_DEBUG_LEVEL",
	                          "HIMAN_STATISTICS",
	                          "HIMAN_CUDA_DEVICE_ID",
	                          "HIMAN_NO_CUDA",
	                          "HIMAN_NO_CUDA_UNPACKING",
	                          "HIMAN_NO_CUDA_PACKING",
	                          "HIMAN_NO_DATABASE",
	                          "HIMAN_PARAM_FILE",
	                          "HIMAN_NO_AUXILIARY_FILE_FULL_CACHE_READ",
	                          "HIMAN_NO_SS_STATE_UPDATE",
	                          "HIMAN_NO_STATISTICS_UPLOAD",
	                          "HIMAN_AUXILIARY_FILES"};

	shared_ptr<configuration> conf = make_shared<configuration>();

	for (const string& key : keys)
	{
		try
		{
			const string val = util::GetEnv(key);

			if (key == "HIMAN_OUTPUT_FILE_TYPE")
				SetOutputFileType(conf, val);
			else if (key == "HIMAN_COMPRESSION")
				SetCompression(conf, val);
			else if (key == "HIMAN_CONFIGURATION_FILE")
				SetConfigurationFile(conf, val);
			else if (key == "HIMAN_THREADS")
				conf->ThreadCount(static_cast<short>(stoi(val)));
			else if (key == "HIMAN_DEBUG_LEVEL")
				SetLogLevel(conf, stoi(val));
			else if (key == "HIMAN_STATISTICS")
				conf->StatisticsLabel(val);
			else if (key == "HIMAN_NO_CUDA")
			{
				conf->UseCuda(false);
				conf->UseCudaForUnpacking(false);
				conf->UseCudaForPacking(false);
			}
			else if (key == "HIMAN_NO_CUDA_UNPACKING")
				conf->UseCudaForUnpacking(false);
			else if (key == "HIMAN_NO_CUDA_PACKING")
				conf->UseCudaForPacking(false);
			else if (key == "HIMAN_NO_DATABASE")
				conf->DatabaseType(kNoDatabase);
			else if (key == "HIMAN_PARAM_FILE")
				conf->ParamFile(val);
			else if (key == "HIMAN_NO_AUXILIARY_FILE_FULL_CACHE_READ")
				conf->ReadAllAuxiliaryFilesToCache(false);
			else if (key == "HIMAN_NO_SS_STATE_UPDATE")
				conf->UpdateSSStateTable(false);
			else if (key == "HIMAN_NO_STATISTIC_UPLOAD")
				conf->UploadStatistics(false);
			else if (key == "HIMAN_AUXILIARY_FILES")
			{
				auto files = util::Split(val, " ");
				conf->AuxiliaryFiles(files);
			}
		}
		catch (const invalid_argument& e)
		{
		}
	}

	return conf;
}

void ParseCommandLine(shared_ptr<configuration>& conf, int argc, char** argv)
{
	namespace po = boost::program_options;

	po::options_description desc("Allowed options", 100);

	// Can't use required() since it doesn't allow us to use --list-plugins

	string outfileType;
	string outfileCompression;
	string confFile, paramFile;
	string statisticsLabel;
	vector<string> auxFiles;
#ifdef HAVE_CUDA
	short int cudaDeviceId = 0;
#endif

	int logLevel = 0;
	short int threadCount = -1;

	// clang-format off

	desc.add_options()
		("help,h", "print out help message")
		("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, csv, querydata, geotiff")
		("compression,c", po::value(&outfileCompression), "output file compression, one of: gz, bzip2")
		("version,v", "display version number")
		("configuration-file,f", po::value(&confFile), "configuration file")
		("auxiliary-files,a", po::value<vector<string>>(&auxFiles), "file(s) containing source data for calculation")
		("threads,j", po::value(&threadCount), "number of started threads")
		("list-plugins,l", "list all defined plugins")
		("debug-level,d", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
		("statistics,s", po::value(&statisticsLabel)->implicit_value("Himan"), "record statistics information")
#ifdef HAVE_CUDA
		("cuda-device-id", po::value(&cudaDeviceId), "use a specific cuda device (default: 0)")
		("cuda-properties", "print cuda device properties of platform (if any)")
		("no-cuda", "disable all cuda extensions")
		("no-cuda-packing", "disable cuda packing of grib data")
		("no-cuda-unpacking", "disable cuda unpacking of grib data")
#endif
		("no-database", "disable database access")
		("param-file", po::value(&paramFile), "parameter definition file for no-database mode (syntax: shortName,paramName)")
		("no-auxiliary-file-full-cache-read", "disable the initial reading of all auxiliary files to cache")
		("no-ss_state-update,X", "do not update ss_state table information")
		("no-statistics-upload", "do not upload statistics to database")
	;

	// clang-format on

	po::positional_options_description p;
	p.add("auxiliary-files", -1);

	po::variables_map opt;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), opt);

	po::notify(opt);

	if (threadCount)
	{
		conf->ThreadCount(threadCount);
	}

	if (auxFiles.size())
	{
		conf->AuxiliaryFiles(auxFiles);
	}

	if (logLevel)
	{
		SetLogLevel(conf, logLevel);
	}

	if (opt.count("version"))
	{
		cout << "himan-bin version " << __DATE__ << " " << __TIME__ << endl;
		exit(1);
	}

#ifndef HAVE_CUDA
	conf->UseCuda(false);
	conf->UseCudaForPacking(false);
	conf->UseCudaForUnpacking(false);
	conf->CudaDeviceCount(0);
#else
	if (opt.count("cuda-properties"))
	{
		CudaCapabilities();
		exit(1);
	}

	if (opt.count("no-cuda-packing"))
	{
		conf->UseCudaForPacking(false);
	}

	if (opt.count("no-cuda-unpacking"))
	{
		conf->UseCudaForUnpacking(false);
	}

	if (opt.count("no-cuda"))
	{
		conf->UseCuda(false);
		conf->UseCudaForPacking(false);
		conf->UseCudaForUnpacking(false);
	}

	// get cuda device count for this server

	int devCount;

	cudaError_t err = cudaGetDeviceCount(&devCount);

	switch (err)
	{
		case cudaSuccess:
			break;

		case cudaErrorNoDevice:
			// No device

			devCount = 0;

			conf->UseCuda(false);
			conf->UseCudaForPacking(false);
			conf->UseCudaForUnpacking(false);

			break;

		default:
			// No driver present or other problems
			// with installation

			devCount = 0;

			conf->UseCuda(false);
			conf->UseCudaForPacking(false);
			conf->UseCudaForUnpacking(false);

			cerr << fmt::format("Error from cuda library: {} ({})\n", cudaGetErrorName(err), err);
			break;
	}

	conf->CudaDeviceCount(devCount);

	if (opt.count("cuda-device-id"))
	{
		SetCudaDeviceId(conf, cudaDeviceId);
	}
#endif

	if (opt.count("no-ss_state-update"))
	{
		conf->UpdateSSStateTable(false);
	}

	if (opt.count("no-statistics-upload"))
	{
		conf->UploadStatistics(false);
	}

	if (!outfileType.empty())
	{
		SetOutputFileType(conf, outfileType);
	}

	if (!outfileCompression.empty())
	{
		SetCompression(conf, outfileCompression);
	}

	if (opt.count("help"))
	{
		cout << "usage: himan [ options ]" << endl;
		cout << desc;
		cout << endl << "Examples:" << endl;
		cout << "  himan -f etc/tpot.json" << endl;
		cout << "  himan -f etc/vvmms.json -a file.grib -t querydata" << endl << endl;
		exit(1);
	}

	if (opt.count("list-plugins"))
	{
		vector<shared_ptr<plugin::himan_plugin>> thePlugins = plugin_factory::Instance()->Plugins();

		for (const auto& plugin : thePlugins)
		{
			cout << "Plugin '" << plugin->ClassName() << "'" << endl;

			switch (plugin->PluginClass())
			{
				case kCompiled:
					if (dynamic_pointer_cast<plugin::compiled_plugin>(plugin)->CudaEnabledCalculation())
					{
						cout << "\tcuda-enabled\n";
					}
					cout << "\ttype compiled" << endl;
					break;

				case kAuxiliary:
					cout << "\ttype aux" << endl;
					break;

				default:
					cout << " has unknown plugin type" << endl;
					exit(1);
			}
		}

		exit(1);
	}

	if (opt.count("no-database"))
	{
		conf->DatabaseType(kNoDatabase);
	}

	if (!paramFile.empty())
	{
		conf->ParamFile(paramFile);
	}

	if (conf->DatabaseType() == kNoDatabase && conf->ParamFile().empty())
	{
		cerr << "param-file options missing" << endl;
		exit(1);
	}

	if (!confFile.empty())
	{
		SetConfigurationFile(conf, confFile);
	}

	if (conf->ConfigurationFileName().empty())
	{
		cerr << "himan: Configuration file not defined" << endl << desc;
		exit(1);
	}

	if (!statisticsLabel.empty())
	{
		conf->StatisticsLabel(statisticsLabel);
	}

	if (opt.count("no-auxiliary-file-full-cache-read"))
	{
		conf->ReadAllAuxiliaryFilesToCache(false);
	}
}
