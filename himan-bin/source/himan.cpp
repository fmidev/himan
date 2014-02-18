/**
 * @file himan.cpp
 *
 * @brief himan main program
 *
 * @author partio
 *
 */

#include <iostream>
#include "himan_common.h"
#include "plugin_factory.h"
#include "json_parser.h"
#include "himan_plugin.h"
#include "compiled_plugin.h"
#include "auxiliary_plugin.h"
#include "logger_factory.h"
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "pcuda.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

void banner();
shared_ptr<configuration> ParseCommandLine(int argc, char** argv);

int main(int argc, char** argv)
{

	shared_ptr<configuration> conf;
	
	try
	{
		conf = ParseCommandLine(argc, argv);
	}
	catch (const std::exception &e)
	{
		cerr << e.what() << endl;
		exit(1);
	}


	unique_ptr<logger> aLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("himan"));

	/*
	 * Initialize plugin factory before parsing configuration file. This prevents himan from
	 * terminating suddenly with SIGSEGV on RHEL5 environments.
	 *
	 * Also, it may be good to have neons -plugin declared at main level of program. This
	 * goes also for the cache plugin.
	 *
	 * Note that we don't actually do anything with the plugin here.
	 */

	shared_ptr<plugin::auxiliary_plugin> n = dynamic_pointer_cast<plugin::auxiliary_plugin> (plugin_factory::Instance()->Plugin("neons"));
	shared_ptr<plugin::auxiliary_plugin> c = dynamic_pointer_cast<plugin::auxiliary_plugin> (plugin_factory::Instance()->Plugin("cache"));

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

	conf.reset(); // we don't need this conf anymore, it was only used as a base for json_parser
	
	banner();

	vector<shared_ptr<plugin::himan_plugin>> thePlugins = plugin_factory::Instance()->Plugins();

	aLogger->Info("Found " + boost::lexical_cast<string> (thePlugins.size()) + " plugins");

	aLogger->Debug("Processqueue size: " + boost::lexical_cast<string> (plugins.size()));

	for (size_t i = 0; i < plugins.size(); i++)
	{

		shared_ptr<plugin_configuration> pc = plugins[i];

		if (pc->Name() == "precipitation")
		{
			aLogger->Warning("Plugin 'precipitation' is deprecated -- use 'split_sum' instead'");
			pc->Name("split_sum");
		}
		
		auto aPlugin = dynamic_pointer_cast<plugin::compiled_plugin > (plugin_factory::Instance()->Plugin(pc->Name()));

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
		}

	}

	return 0;

}  


void banner()
{
	cout << endl
			  << "************************************************" << endl
			  << "* By the Power of Grayskull, I Have the Power! *" << endl
			  << "************************************************" << endl << endl;

}

shared_ptr<configuration> ParseCommandLine(int argc, char** argv)
{

	shared_ptr<configuration> conf = make_shared<configuration> ();

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");

	// Can't use required() since it doesn't allow us to use --list-plugins

	string outfileType = "";
	string confFile = "";
	string statisticsLabel = "";
	vector<string> auxFiles;
	short int cudaDeviceId = 0;
	
	himan::HPDebugState debugState = himan::kDebugMsg;

	int logLevel = 0;
	short int threadCount = -1;

	desc.add_options()
	("help,h", "print out help message")
	("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, netcdf, querydata")
	("version,v", "display version number")
	("configuration-file,f", po::value(&confFile), "configuration file")
	("auxiliary-files,a", po::value<vector<string> > (&auxFiles), "auxiliary (helper) file(s)")
	("threads,j", po::value(&threadCount), "number of started threads")
	("list-plugins,l", "list all defined plugins")
	("debug-level,d", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
	("statistics,s", po::value(&statisticsLabel), "record statistics information")
	("no-cuda", "disable all cuda extensions")
	("no-cuda-packing", "disable cuda packing and unpacking of grib data")
	("cuda-device-id", po::value(&cudaDeviceId), "use a specific cuda device (default: 0)")
	("cuda-properties", "print cuda device properties of platform (if any)")
	;

	po::positional_options_description p;
	p.add("auxiliary-files", -1);

	po::variables_map opt;
	po::store(po::command_line_parser(argc, argv)
			  .options(desc)
			  .positional(p)
			  .run(),
			  opt);

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

	shared_ptr<plugin::pcuda> cuda;

#ifndef HAVE_CUDA
	if (opt.count("cuda-properties"))
	{
		cout << "CUDA support turned off at compile time" << endl;
		exit(1);
	}

	conf->UseCuda(false);
	conf->UseCudaForPacking(false);
	conf->CudaDeviceCount(0);

	if (opt.count("cuda-device-id"))
	{
		cerr << "CUDA support turned off at compile time" << endl;
	}

#else
	cuda = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));
	
	if (opt.count("cuda-properties"))
	{
		cuda->Capabilities();
		exit(1);
	}

	if (opt.count("no-cuda-packing"))
	{
		conf->UseCudaForPacking(false);
	}

	if (opt.count("no-cuda"))
	{
		conf->UseCuda(false);
		conf->UseCudaForPacking(false);
	}

	conf->CudaDeviceCount(static_cast<short> (cuda->DeviceCount()));

	if (opt.count("cuda-device-id"))
	{
		if (cudaDeviceId < conf->CudaDeviceCount())
		{
			cerr << "cuda device id " << cudaDeviceId << " requested, number of available cuda devices is " << conf->CudaDeviceCount() << endl;
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
		else
		{
			cerr << "Invalid file type: " << outfileType << endl;
			exit(1);
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
			cout << "Plugin '"  << thePlugins[i]->ClassName() << "'" << endl << "\tversion " << thePlugins[i]->Version() << endl; 

			switch (thePlugins[i]->PluginClass())
			{

				case kCompiled:
					if (dynamic_pointer_cast<plugin::compiled_plugin> (thePlugins[i])->CudaEnabledCalculation())
					{
						cout << "\tcuda-enabled\n";
					}
					cout << "\ttype compiled (hard-coded) --> " << dynamic_pointer_cast<plugin::compiled_plugin> (thePlugins[i])->Formula() << endl;
					break;

				case kAuxiliary:
					cout << "\ttype aux" << endl;
					break;

				case kInterpreted:
					cout << "\ttype interpreted" << endl;
					break;

				default:
					cout << " has unknown plugin type" << endl;
					exit(1);
			}
		}

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
