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
#include "logger_factory.h"
#include "plugin_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>

using namespace himan;
using namespace std;

void banner();
shared_ptr<configuration> ParseCommandLine(int argc, char** argv);

struct plugin_timing
{
	std::string plugin_name;
	unsigned short order_number;  // plugin order number (if called more than once))
	size_t time_elapsed;          // elapsed time in ms
};

unsigned short HighestOrderNumber(const vector<plugin_timing>& timingList, const std::string& pluginName)
{
	unsigned short highest = 1;

	for (size_t i = 0; i < timingList.size(); i++)
	{
		if (timingList[i].plugin_name == pluginName)
		{
			if (timingList[i].order_number >= highest)
			{
				highest = static_cast<unsigned short>(timingList[i].order_number + 1);
			}
		}
	}

	return highest;
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

	unique_ptr<logger> aLogger = unique_ptr<logger>(logger_factory::Instance()->GetLog("himan"));
	unique_ptr<timer> aTimer;

	if (!conf->StatisticsLabel().empty())
	{
		// This timer is used to measure time elapsed for each plugin call
		aTimer = unique_ptr<timer>(timer_factory::Instance()->GetTimer());
	}

	/*
	 * Initialize plugin factory before parsing configuration file. This prevents himan from
	 * terminating suddenly with SIGSEGV on RHEL5 environments.
	 */

	shared_ptr<plugin::auxiliary_plugin> c =
	    dynamic_pointer_cast<plugin::auxiliary_plugin>(plugin_factory::Instance()->Plugin("cache"));

	std::vector<shared_ptr<plugin_configuration>> plugins;

	try
	{
		plugins = json_parser::Instance()->Parse(conf);
	}
	catch (std::runtime_error& e)
	{
		aLogger->Fatal(e.what());
		exit(1);
	}

	banner();

	vector<shared_ptr<plugin::himan_plugin>> thePlugins = plugin_factory::Instance()->Plugins();

	aLogger->Info("Found " + boost::lexical_cast<string>(thePlugins.size()) + " plugins");

	aLogger->Debug("Processqueue size: " + boost::lexical_cast<string>(plugins.size()));

	vector<plugin_timing> pluginTimes;
	size_t totalTime = 0;

	while (plugins.size() > 0)
	{
		auto pc = plugins[0];

		if (pc->StatisticsEnabled())
		{
			aTimer->Start();
		}

		if (pc->Name() == "cloud_type")
		{
			aLogger->Warning("Plugin 'cloud_type' is deprecated -- use 'cloud_code' instead'");
			pc->Name("cloud_code");
		}
		else if (pc->Name() == "fmi_weather_symbol_1")
		{
			aLogger->Warning("Plugin 'fmi_weather_symbol_1' is deprecated -- use 'weather_code_2' instead'");
			pc->Name("weather_code_2");
		}
		else if (pc->Name() == "rain_type")
		{
			aLogger->Warning("Plugin 'rain_type' is deprecated -- use 'weather_code_1' instead'");
			pc->Name("weather_code_1");
		}

		auto aPlugin = dynamic_pointer_cast<plugin::compiled_plugin>(plugin_factory::Instance()->Plugin(pc->Name()));

		if (!aPlugin)
		{
			aLogger->Error("Unable to declare plugin " + pc->Name());
			continue;
		}

		if (pc->StatisticsEnabled())
		{
			pc->StartStatistics();
		}

		aLogger->Info("Calculating " + pc->Name());

		try
		{
			aPlugin->Process(pc);
		}
		catch (const exception& e)
		{
			aLogger->Fatal(string("Caught exception: ") + e.what());
			exit(1);
		}

		if (pc->StatisticsEnabled())
		{
			pc->WriteStatistics();

			aTimer->Stop();
			plugin_timing t;
			t.plugin_name = pc->Name();
			t.time_elapsed = aTimer->GetTime();
			t.order_number = HighestOrderNumber(pluginTimes, pc->Name());

			totalTime += t.time_elapsed;
			pluginTimes.push_back(t);
		}

		plugins.erase(plugins.begin());  // remove configuration and resize container

#if defined DEBUG and defined HAVE_CUDE
		// For 'cuda-memcheck --leak-check full'
		CUDA_CHECK(cudaDeviceReset());
#endif
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

	po::options_description desc("Allowed options");

	// Can't use required() since it doesn't allow us to use --list-plugins

	string outfileType = "";
	string outfileCompression = "";
	string confFile = "";
	string statisticsLabel = "";
	vector<string> auxFiles;
	short int cudaDeviceId = 0;

	himan::HPDebugState debugState = himan::kDebugMsg;

	int logLevel = 0;
	short int threadCount = -1;

// clang-format off

	desc.add_options()
		("help,h", "print out help message")
		("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, netcdf, querydata")
		("compression,c", po::value(&outfileCompression), "output file compression, one of: gz, bzip2")
		("version,v", "display version number")
		("configuration-file,f", po::value(&confFile), "configuration file")
		("auxiliary-files,a", po::value<vector<string>>(&auxFiles), "auxiliary (helper) file(s)")
		("threads,j", po::value(&threadCount), "number of started threads")
		("list-plugins,l", "list all defined plugins")
		("debug-level,d", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
		("statistics,s", po::value(&statisticsLabel), "record statistics information")
		("radon,R", "use only radon database")
		("neons,N", "use only neons database")
		("cuda-device-id", po::value(&cudaDeviceId), "use a specific cuda device (default: 0)")
		("cuda-properties", "print cuda device properties of platform (if any)")
		("no-cuda", "disable all cuda extensions")
		("no-cuda-packing", "disable cuda packing of grib data")
		("no-cuda-unpacking", "disable cuda unpacking of grib data")
		("no-cuda-interpolation", "disable cuda grid interpolation");

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

	logger_factory::Instance()->DebugState(debugState);

	if (opt.count("version"))
	{
		cout << "himan-bin version " << __DATE__ << " " << __TIME__ << endl;
		exit(1);
	}

#ifndef HAVE_CUDA
	if (opt.count("cuda-properties"))
	{
		cout << "CUDA support turned off at compile time" << endl;
		exit(1);
	}

	conf->UseCuda(false);
	conf->UseCudaForPacking(false);
	conf->UseCudaForUnpacking(false);
	conf->UseCudaForInterpolation(false);
	conf->CudaDeviceCount(0);

	if (opt.count("cuda-device-id"))
	{
		cerr << "CUDA support turned off at compile time" << endl;
	}

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
	}

	conf->CudaDeviceCount(static_cast<short>(devCount));

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
					cout << "\ttype compiled --> "
					     << dynamic_pointer_cast<plugin::compiled_plugin>(thePlugins[i])->Formula() << endl;
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
	if (!confFile.empty())
	{
		conf->ConfigurationFile(confFile);
	}
	else
	{
		cerr << "himan: Configuration file not defined" << endl;
		exit(1);
	}

	if (!statisticsLabel.empty())
	{
		conf->StatisticsLabel(statisticsLabel);
	}

	return conf;
}
