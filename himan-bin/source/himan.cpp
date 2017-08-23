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
#include <boost/program_options.hpp>
#include <future>
#include <iostream>
#include <vector>

using namespace himan;
using namespace std;

void banner();
shared_ptr<configuration> ParseCommandLine(int argc, char** argv);

struct plugin_timing
{
	std::string plugin_name;
	int order_number;     // plugin order number (if called more than once))
	size_t time_elapsed;  // elapsed time in ms
};

vector<plugin_timing> pluginTimes;

int HighestOrderNumber(const vector<plugin_timing>& timingList, const std::string& pluginName)
{
	int highest = 1;

	for (size_t i = 0; i < timingList.size(); i++)
	{
		if (timingList[i].plugin_name == pluginName)
		{
			if (timingList[i].order_number >= highest)
			{
				highest = timingList[i].order_number + 1;
			}
		}
	}

	return highest;
}

void ExecutePlugin(shared_ptr<plugin_configuration> pc)
{
	timer aTimer;
	logger aLogger("himan");
	if (pc->StatisticsEnabled())
	{
		aTimer.Start();
	}

	auto aPlugin = dynamic_pointer_cast<plugin::compiled_plugin>(plugin_factory::Instance()->Plugin(pc->Name()));

	if (!aPlugin)
	{
		aLogger.Error("Unable to declare plugin " + pc->Name());
		return;
	}

	if (pc->StatisticsEnabled())
	{
		pc->StartStatistics();
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
		pc->WriteStatistics();

		aTimer.Stop();
		plugin_timing t;
		t.plugin_name = pc->Name();
		t.time_elapsed = aTimer.GetTime();
		t.order_number = HighestOrderNumber(pluginTimes, pc->Name());

		pluginTimes.push_back(t);
	}

#if defined DEBUG and defined HAVE_CUDA
	// For 'cuda-memcheck --leak-check full'
	CUDA_CHECK(cudaDeviceReset());
#endif
}

int main(int argc, char** argv)
{
	shared_ptr<configuration> conf;

	try
	{
		conf = ParseCommandLine(argc, argv);
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

	while (plugins.size() > 0)
	{
		auto pc = plugins[0];

		plugins.erase(plugins.begin());

		if (pc->AsyncExecution())
		{
			aLogger.Info("Asynchronous launch for " + pc->Name());
			asyncs.push_back(async(launch::async, [](shared_ptr<plugin_configuration> pc) { ExecutePlugin(pc); }, pc));

			continue;
		}

		ExecutePlugin(pc);
	}

	for (auto& fut : asyncs)
	{
		fut.wait();
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

		size_t totalTime = 0;

		for (const auto& time : pluginTimes)
		{
			totalTime += time.time_elapsed;
		}

		cout << endl << "*** TOTAL timings for himan ***" << endl;

		for (size_t i = 0; i < pluginTimes.size(); i++)
		{
			plugin_timing t = pluginTimes[i];

			cout << t.plugin_name;

			if (t.order_number > 1)
			{
				cout << " #" << t.order_number;
			}

			string indent = "\t\t\t\t";

			if (t.plugin_name.length() < 6)
			{
				indent = "\t\t\t";
			}
			else if (t.plugin_name.length() < 12)
			{
				indent = "\t\t";
			}
			else if (t.plugin_name.length() < 18)
			{
				indent = "\t";
			}

			cout << indent << t.time_elapsed << " ms\t("
			     << static_cast<int>(((static_cast<double>(t.time_elapsed) / static_cast<double>(totalTime)) * 100))
			     << "%)" << endl;
		}

		cout << "------------------------------------" << endl;
		cout << "Total duration:\t\t" << totalTime << " ms" << endl;
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

		for (int i = 0; i < 3; ++i)
		{
			std::cout << "Maximum dimension " << i << " of block:\t" << devProp.maxThreadsDim[i] << std::endl;
		}

		for (int i = 0; i < 3; ++i)
		{
			std::cout << "Maximum dimension " << i << " of grid:\t" << devProp.maxGridSize[i] << std::endl;
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
shared_ptr<configuration> ParseCommandLine(int argc, char** argv)
{
	shared_ptr<configuration> conf = make_shared<configuration>();

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

	himan::HPDebugState debugState = himan::kDebugMsg;

	int logLevel = 0;
	short int threadCount = -1;

	// clang-format off

	desc.add_options()
		("help,h", "print out help message")
		("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, csv, querydata")
		("compression,c", po::value(&outfileCompression), "output file compression, one of: gz, bzip2")
		("version,v", "display version number")
		("configuration-file,f", po::value(&confFile), "configuration file")
		("auxiliary-files,a", po::value<vector<string>>(&auxFiles), "auxiliary (helper) file(s)")
		("threads,j", po::value(&threadCount), "number of started threads")
		("list-plugins,l", "list all defined plugins")
		("debug-level,d", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
		("statistics,s", po::value(&statisticsLabel)->implicit_value("Himan"), "record statistics information")
		("radon,R", "use only radon database")
		("neons,N", "use only neons database")
#ifdef HAVE_CUDA
		("cuda-device-id", po::value(&cudaDeviceId), "use a specific cuda device (default: 0)")
		("cuda-properties", "print cuda device properties of platform (if any)")
		("no-cuda", "disable all cuda extensions")
		("no-cuda-packing", "disable cuda packing of grib data")
		("no-cuda-unpacking", "disable cuda unpacking of grib data")
		("no-cuda-interpolation", "disable cuda grid interpolation")
#endif
		("no-database", "disable database access")
		("param-file", po::value(&paramFile), "parameter definition file for no-database mode (syntax: shortName,paramName)")
		("no-auxiliary-file-full-cache-read", "disable the initial reading of all auxiliary files to cache")
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
		switch (logLevel)
		{
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
	}

	logger::MainDebugState = debugState;

	if (opt.count("version"))
	{
		cout << "himan-bin version " << __DATE__ << " " << __TIME__ << endl;
		exit(1);
	}

#ifndef HAVE_CUDA
	conf->UseCuda(false);
	conf->UseCudaForPacking(false);
	conf->UseCudaForUnpacking(false);
	conf->UseCudaForInterpolation(false);
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

	if (opt.count("no-cuda-interpolation"))
	{
		conf->UseCudaForInterpolation(false);
	}

	if (opt.count("no-cuda"))
	{
		conf->UseCuda(false);
		conf->UseCudaForPacking(false);
		conf->UseCudaForUnpacking(false);
		conf->UseCudaForInterpolation(false);
	}

	// get cuda device count for this server

	int devCount;

	cudaError_t err = cudaGetDeviceCount(&devCount);

	if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver)
	{
		// No device or no driver present

		devCount = 0;

		conf->UseCuda(false);
		conf->UseCudaForPacking(false);
		conf->UseCudaForUnpacking(false);
		conf->UseCudaForInterpolation(false);
	}

	conf->CudaDeviceCount(devCount);

	if (opt.count("cuda-device-id"))
	{
		if (cudaDeviceId >= conf->CudaDeviceCount() || cudaDeviceId < 0)
		{
			cerr << "cuda device id " << cudaDeviceId << " requested, allowed maximum cuda device id is "
			     << conf->CudaDeviceCount() - 1 << endl;
			cerr << "cuda mode is disabled" << endl;
			conf->UseCuda(false);
			conf->UseCudaForPacking(false);
		}

		conf->CudaDeviceId(cudaDeviceId);
	}
#endif

	if (!outfileType.empty())
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
		else
		{
			cerr << "Invalid file type: " << outfileType << endl;
			exit(1);
		}
	}

	if (!outfileCompression.empty())
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
		}
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

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "Plugin '" << thePlugins[i]->ClassName() << "'" << endl
			     << "\tversion " << thePlugins[i]->Version() << endl;

			switch (thePlugins[i]->PluginClass())
			{
				case kCompiled:
					if (dynamic_pointer_cast<plugin::compiled_plugin>(thePlugins[i])->CudaEnabledCalculation())
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

	if (opt.count("radon"))
	{
		conf->DatabaseType(kRadon);
	}
	else if (opt.count("neons"))
	{
		conf->DatabaseType(kNeons);
	}
	else if (opt.count("neons") && opt.count("radon"))
	{
		cerr << "Both radon and neons options cannot be selected" << endl;
		exit(1);
	}
	else if (opt.count("no-database"))
	{
		conf->DatabaseType(kNoDatabase);
		if (opt.count("param-file") == 0)
		{
			cerr << "param-file options missing" << endl;
			exit(1);
		}
	}

	if (!confFile.empty())
	{
		conf->ConfigurationFile(confFile);
	}
	else
	{
		cerr << "himan: Configuration file not defined" << endl << desc;
		exit(1);
	}

	if (!paramFile.empty())
	{
		conf->ParamFile(paramFile);
	}

	if (!statisticsLabel.empty())
	{
		conf->StatisticsLabel(statisticsLabel);
	}

	if (opt.count("no-auxiliary-file-full-cache-read"))
	{
		conf->ReadAllAuxiliaryFilesToCache(false);
	}
	return conf;
}
