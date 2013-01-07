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
#include "ini_parser.h"
#include "himan_plugin.h"
#include "compiled_plugin.h"
#include "auxiliary_plugin.h"
#include "logger_factory.h"
#include <vector>
#include <boost/lexical_cast.hpp>

#ifdef DEBUG
#include "timer_factory.h"
#endif

using namespace himan;

void banner();

int main(int argc, char** argv)
{

    std::shared_ptr<configuration> theConfiguration = ini_parser::Instance()->Parse(argc, argv);

    banner();

    std::unique_ptr<logger> theLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("himan"));

    std::vector<std::shared_ptr<plugin::himan_plugin>> thePlugins = plugin_factory::Instance()->Plugins();

    theLogger->Info("Found " + boost::lexical_cast<std::string> (thePlugins.size()) + " plugins");

    for (size_t i = 0; i < thePlugins.size(); i++)
    {
        std::string stub = "Plugin '"  + thePlugins[i]->ClassName() + "'";

        switch (thePlugins[i]->PluginClass())
        {
        case kCompiled:
            theLogger->Debug(stub + " \ttype compiled (hard-coded) --> " + std::dynamic_pointer_cast<plugin::compiled_plugin> (thePlugins[i])->Formula());
            break;

        case kAuxiliary:
            theLogger->Debug(stub + "\ttype aux");
            break;

        case kInterpreted:
            theLogger->Debug(stub + "\ttype interpreted");
            break;

        default:
            theLogger->Fatal("Unknown plugin type");
            exit(1);
        }
    }


    theLogger->Debug("Requested plugin(s):");

    std::vector<std::string> theRequestedPlugins = theConfiguration->Plugins();

    for (size_t i = 0; i < theRequestedPlugins.size(); i++)
    {
        theLogger->Debug(theRequestedPlugins[i]);
    }

    for (size_t i = 0; i < theRequestedPlugins.size(); i++)
    {
        std::string theName = theRequestedPlugins[i];

        theLogger->Info("Calculating " + theName);

        std::shared_ptr<plugin::compiled_plugin> thePlugin = std::dynamic_pointer_cast<plugin::compiled_plugin > (plugin_factory::Instance()->Plugin(theName));

        if (!thePlugin)
        {
            theLogger->Error("Unable to declare plugin " + theName);
            continue;
        }

#ifdef DEBUG
        std::unique_ptr<timer> t = std::unique_ptr<timer> (timer_factory::Instance()->GetTimer());
        t->Start();
#endif

        thePlugin->Process(theConfiguration);

#ifdef DEBUG
        t->Stop();
        theLogger->Debug("Processing " + theName + " took " + boost::lexical_cast<std::string> (static_cast<long> (t->GetTime()/1000)) + " milliseconds");
#endif
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
