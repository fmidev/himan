//============================================================================
// Name        : hp.cpp
// Author      : Mikko Partio
// Version     : 0.1
// Copyright   : My copyright notice
// Description : Main program for hilpee program
//============================================================================

#include <iostream>
#include "plugin_factory.h"
#include "ini_parser.h"
#include "hilpee_plugin.h"
#include "compiled_plugin.h"
#include "auxiliary_plugin.h"
#include "logger_factory.h"
#include <vector>
#include <boost/lexical_cast.hpp>
#include <memory> // for std::unique_ptr

using namespace hilpee;

int main(int argc, char ** argv) {

	std::shared_ptr<configuration> theConfiguration = ini_parser::Instance()->Parse(argc, argv);

	std::unique_ptr<logger> theLogger = logger_factory::Instance()->GetLog("himan");

	std::cout << "************************************************" << std::endl;
	std::cout << "* By the power of Greyskull, I Have The Power! *" << std::endl;
	std::cout << "************************************************" << std::endl;

	std::vector<std::shared_ptr<plugin::hilpee_plugin>> thePlugins = plugin_factory::Instance()->Plugins();

	theLogger->Info("Found " + boost::lexical_cast<std::string> (thePlugins.size()) + " plugins");

	for (size_t i = 0; i < thePlugins.size(); i++) {
		std::string stub = "Plugin '"  + thePlugins[i]->ClassName() + "'";

		switch (thePlugins[i]->PluginClass()) {
		case kCompiled:
			theLogger->Info(stub + " \ttype compiled (hard-coded) --> " + std::dynamic_pointer_cast<plugin::compiled_plugin> (thePlugins[i])->Formula());
			break;

		case kAuxiliary:
			theLogger->Info(stub + "\ttype aux");
			break;

		case kInterpreted:
			theLogger->Info(stub + "\ttype interpreted");
			break;

		default:
			theLogger->Warning("Unknown plugin type");
			break;
		}
	}


	theLogger->Debug("Requested plugin(s):");

	std::vector<std::string> theRequestedPlugins = theConfiguration->Plugins();

	for (std::string theName : theRequestedPlugins)
		theLogger->Debug(theName);

	for (std::string theName : theRequestedPlugins) {
		theLogger->Info("Calculating " + theName);

		std::shared_ptr<plugin::compiled_plugin> thePlugin = std::dynamic_pointer_cast<plugin::compiled_plugin > (plugin_factory::Instance()->Plugin(theName));

		if (!thePlugin) {
			theLogger->Error("Unable to declare plugin " + theName);
			continue;
		}

		thePlugin->Process(theConfiguration);

	}

	return 0;
}
