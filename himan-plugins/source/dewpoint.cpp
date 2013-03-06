/**
 * @file dewpoint.cpp
 *
 * @date Jan 21, 2012
 * @author partio
 */

#include "dewpoint.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "neons.h"
#include "pcuda.h"

#undef HIMAN_AUXILIARY_INCLUDE

#ifdef DEBUG
#include "timer_factory.h"
#endif

using namespace std;
using namespace himan::plugin;

const double RW = 461.5; // Vesihoyryn kaasuvakio (J / K kg)
const double L = 2.5e6; // Veden hoyrystymislampo (J / kg)
const double RW_div_L = RW / L;

dewpoint::dewpoint() : itsUseCuda(false)
{
    itsClearTextFormula = "Td = T / (1 - (T * ln(RH)*(Rw/L)))";

    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("dewpoint"));

}

void dewpoint::Process(shared_ptr<const plugin_configuration> conf)
{

    shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

    if (c->HaveCuda())
    {
        string msg = "I possess the powers of CUDA";

        if (!conf->UseCuda())
        {
            msg += ", but I won't use them";
        }
        else
        {
            msg += ", and I'm not afraid to use them";
            itsUseCuda = true;
        }

        itsLogger->Info(msg);

    }

    // Get number of threads to use

    unsigned short threadCount = ThreadCount(conf->ThreadCount());

    boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();
	
    /*
     * Get producer information from neons
     */

    if (conf->FileWriteOption() == kNeons)
    {
        shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

        map<string,string> prodInfo = n->ProducerInfo(targetInfo->Producer().Id());

        if (prodInfo.size())
        {
            producer prod(targetInfo->Producer().Id());

            prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
            prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
            prod.Name(prodInfo["name"]);

            targetInfo->Producer(prod);
        }

    }

    /*
     * Set target parameter to potential temperature
     *
     * We need to specify grib and querydata parameter information
     * since we don't know which one will be the output format.
     *
     */

    vector<param> params;

    param requestedParam ("TD-C", 10);

    requestedParam.GribDiscipline(0);
    requestedParam.GribCategory(0);
    requestedParam.GribParameter(6);

    params.push_back(requestedParam);

    targetInfo->Params(params);

    /*
     * Create data structures.
     */

    targetInfo->Create();

    /*
     * Initialize parent class functions for dimension handling
     */

    Dimension(conf->LeadingDimension());
    FeederInfo(shared_ptr<info> (new info(*targetInfo)));
    FeederInfo()->Param(requestedParam);

    /*
     * Each thread will have a copy of the target info.
     */

    vector<shared_ptr<info> > targetInfos;

    targetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
    {

        itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

        targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

        boost::thread* t = new boost::thread(&dewpoint::Run,
                                             this,
                                             targetInfos[i],
                                             conf,
                                             i + 1);

        g.add_thread(t);

    }

    g.join_all();

	itsLogger->Info("Threads finished");

    if (conf->FileWriteOption() == kSingleFile)
    {
        shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

        string theOutputFile = conf->ConfigurationFile();

        theWriter->ToFile(targetInfo, conf->OutputFileType(), conf->FileWriteOption(), theOutputFile);

    }
}

void dewpoint::Run(shared_ptr<info> myTargetInfo,
               const shared_ptr<const configuration>& conf,
               unsigned short threadIndex)
{
    while (AdjustLeadingDimension(myTargetInfo))
    {
        Calculate(myTargetInfo, conf, threadIndex);
    }

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void dewpoint::Calculate(shared_ptr<info> myTargetInfo,
                     const shared_ptr<const configuration>& conf,
                     unsigned short threadIndex)
{

	
    shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

    // Required source parameters

    param TParam("T-K");
    param RHParam("RH-PRCNT");

    unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("dewpointThread #" + boost::lexical_cast<string> (threadIndex)));
	
    ResetNonLeadingDimension(myTargetInfo);

    myTargetInfo->FirstParam();

    while (AdjustNonLeadingDimension(myTargetInfo))
    {

        myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
                                " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

        double TBase = 0;

        shared_ptr<info> TInfo;
        shared_ptr<info> RHInfo;

        try
        {
        	TInfo = f->Fetch(conf,
        						myTargetInfo->Time(),
        	                    myTargetInfo->Level(),
        	                    TParam);

        	RHInfo = f->Fetch(conf,
        						myTargetInfo->Time(),
        	        			myTargetInfo->Level(),
        	        			RHParam);


        }
        catch (HPExceptionType e)
        {
        	switch (e)
        	{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value
					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
        }

        assert(RHInfo->Param().Unit() == kPrcnt);

    	if (TInfo->Param().Unit() == kC)
    	{
    		TBase = 273.15;
    	}

        shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> RHGrid(RHInfo->Grid()->ToNewbaseGrid());

        int missingCount = 0;
        int count = 0;

        bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && *myTargetInfo->Grid() == *RHInfo->Grid());

        if (true)
        {
            assert(targetGrid->Size() == myTargetInfo->Data()->Size());

            myTargetInfo->ResetLocation();

            targetGrid->Reset();

#ifdef DEBUG
            unique_ptr<timer> t = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
            t->Start();
#endif

            while (myTargetInfo->NextLocation() && targetGrid->Next())
            {
                count++;

                double T = kFloatMissing;
                double RH = kFloatMissing;

                InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
                InterpolateToPoint(targetGrid, RHGrid, equalGrids, RH);

                if (T == kFloatMissing || RH == kFloatMissing)
                {
                    missingCount++;

                    myTargetInfo->Value(kFloatMissing);
                    continue;
                }

                double TD = ((T+TBase) / (1 - ((T+TBase) * log(RH) * (RW_div_L)))) - 273.15 + TBase;

                if (!myTargetInfo->Value(TD))
                {
                    throw runtime_error(ClassName() + ": Failed to set value to matrix");
                }

            }
#ifdef DEBUG
            t->Stop();
            itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (t->GetTime()) + " microseconds on CPU");
#endif
        }

        /*
         * Now we are done for this level
         *
         * Clone info-instance to writer since it might change our descriptor places
         */

        myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

        if (conf->FileWriteOption() == kMultipleFiles || conf->FileWriteOption() == kNeons)
        {
            shared_ptr<writer> w = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

            w->ToFile(shared_ptr<info>(new info(*myTargetInfo)), conf->OutputFileType(), conf->FileWriteOption());
        }
    }
}
