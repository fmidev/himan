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

#ifdef DEBUG
#include "timer_factory.h"
#endif

#define HIMAN_AUXILIARY_INCLUDE
#include "pcuda.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

void banner();
std::shared_ptr<configuration> ParseCommandLine(int argc, char** argv);

int main(int argc, char** argv)
{

	shared_ptr<configuration> conf = ParseCommandLine(argc, argv);

	std::unique_ptr<logger> theLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("himan"));

	/*
	 * Initialize plugin factory before parsing configuration file. This prevents himan from
	 * terminating suddenly with SIGSEGV on RHEL5 environments.
	 *
	 * Also, it may be good to have neons -plugin declared at main level of program. This
	 * goes also for the cache plugin.
	 *
	 * Note that we don't actually do anything with the plugin here.
	 */

	std::shared_ptr<plugin::auxiliary_plugin> n = std::dynamic_pointer_cast<plugin::auxiliary_plugin> (plugin_factory::Instance()->Plugin("neons"));
	std::shared_ptr<plugin::auxiliary_plugin> c = std::dynamic_pointer_cast<plugin::auxiliary_plugin> (plugin_factory::Instance()->Plugin("cache"));
 
	json_parser::Instance()->Parse(conf);

	banner();

	std::vector<std::shared_ptr<plugin::himan_plugin>> thePlugins = plugin_factory::Instance()->Plugins();

	theLogger->Info("Found " + boost::lexical_cast<std::string> (thePlugins.size()) + " plugins");

	for (size_t i = 0; i < thePlugins.size(); i++)
	{
		switch (thePlugins[i]->PluginClass())
		{
		case kCompiled:
			theLogger->Info("Plugin '"  + thePlugins[i]->ClassName() + "'" + " \ttype compiled (hard-coded) --> " + std::dynamic_pointer_cast<plugin::compiled_plugin> (thePlugins[i])->Formula());
			break;

		case kAuxiliary:
			theLogger->Debug("Plugin '"  + thePlugins[i]->ClassName() + "'" + "\ttype aux");
			break;

		case kInterpreted:
			theLogger->Debug("Plugin '"  + thePlugins[i]->ClassName() + "'" + "\ttype interpreted");
			break;

		default:
			theLogger->Fatal("Unknown plugin type");
			exit(1);
		}
	}

	std::vector<std::shared_ptr<info>> queues = conf->Infos();

	theLogger->Debug("Processqueue size: " + boost::lexical_cast<std::string> (queues.size()));

	for (size_t i = 0; i < queues.size(); i++)
	{

		theLogger->Debug("Number of plugins for processqueue element " + boost::lexical_cast<string> (i) + ": " + boost::lexical_cast<string> (queues[i]->Plugins().size()));

		for (size_t j = 0; j < queues[i]->Plugins().size(); j++)
		{

		 	plugin_configuration pc = queues[i]->Plugins()[j];
            
			theLogger->Info("Calculating " + pc.Name());
           
			std::shared_ptr<plugin::compiled_plugin> thePlugin = std::dynamic_pointer_cast<plugin::compiled_plugin > (plugin_factory::Instance()->Plugin(pc.Name()));

			if (!thePlugin)
			{
				theLogger->Error("Unable to declare plugin " + pc.Name());
				continue;
			}

#ifdef DEBUG
			std::unique_ptr<timer> t = std::unique_ptr<timer> (timer_factory::Instance()->GetTimer());
			t->Start();
#endif

			conf->PluginConfiguration(pc);

			thePlugin->Process(conf, queues[i]);

#ifdef DEBUG
			t->Stop();
			theLogger->Debug("Processing " + pc.Name() + " took " + boost::lexical_cast<std::string> (static_cast<long> (t->GetTime()/1000)) + " milliseconds");
#endif
		}

	}

	return 0;

}  


void banner()
{
	std::cout << std::endl
			  << "************************************************" << std::endl
			  << "* By the Power of Grayskull, I Have the Power! *" << std::endl
			  << "************************************************" << std::endl << std::endl;

	sleep(1);

}

std::shared_ptr<configuration> ParseCommandLine(int argc, char** argv)
{

	std::shared_ptr<configuration> conf(new configuration());

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");

	// Can't use required() since it doesn't allow us to use --list-plugins

	std::string outfileType = "";
	std::string confFile = "";
	std::vector<std::string> auxFiles;
	
	himan::HPDebugState debugState = himan::kDebugMsg;

	int logLevel = 0;
	short int threadCount = -1;

	desc.add_options()
	("help,h", "print out help message")
	("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, netcdf, querydata")
	("version,v", "display version number")
	("configuration-file,f", po::value(&confFile), "configuration file")
	("auxiliary-files,a", po::value<std::vector<std::string> > (&auxFiles), "auxiliary (helper) file(s)")
	("threads,j", po::value(&threadCount), "number of started threads")
	("list-plugins,l", "list all defined plugins")
	("debug-level,d", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
	("no-cuda", "disable cuda extensions")
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
		std::cout << "himan version ???" << endl;
		exit(1);
	}

	if (opt.count("cuda-properties"))
	{
#ifndef HAVE_CUDA
		std::cout << "CUDA support turned off at compile time" << std::endl;
#else
		shared_ptr<plugin::pcuda> p = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));
		p->Capabilities();
#endif
		exit(1);
	}

	if (opt.count("no-cuda"))
	{
		conf->UseCuda(false);
	}

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
			throw runtime_error("Invalid file type: " + outfileType);
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

		vector<shared_ptr<plugin::himan_plugin> > thePlugins = plugin_factory::Instance()->CompiledPlugins();

		cout << "Compiled plugins" << endl;

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << endl;
		}

		thePlugins = plugin_factory::Instance()->InterpretedPlugins();

		cout << "Interpreted plugins" << endl;

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << endl;
		}

		cout << "Auxiliary plugins" << endl;

		thePlugins = plugin_factory::Instance()->AuxiliaryPlugins();

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << endl;
		}

		exit(1);
	}


	if (!confFile.empty())
	{
		conf->ConfigurationFile(confFile);
	}
	else
	{
		throw runtime_error("himan: Configuration file not defined");
	}

	return conf;
}
