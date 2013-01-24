/*
 * tk2tc.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "tk2tc.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

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

#ifdef HAVE_CUDA
namespace himan
{
namespace plugin
{
namespace tk2tc_cuda
{
void doCuda(const float* Tin, float* Tout, size_t N, unsigned short deviceIndex);
}
}
}
#endif

tk2tc::tk2tc() : itsUseCuda(false)
{
    itsClearTextFormula = "Tc = Tk - 273.15";

    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("tk2tc"));
}

void tk2tc::Process(std::shared_ptr<configuration> conf)
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

    /*
     * The target information is parsed from the configuration file.
     */

    shared_ptr<info> theTargetInfo = conf->Info();

    /*
     * Get producer information from neons if whole_file_write is false.
     */

    if (!conf->WholeFileWrite())
    {
        shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

        map<string,string> prodInfo = n->ProducerInfo(theTargetInfo->Producer().Id());

        if (prodInfo.size())
        {
            producer prod(theTargetInfo->Producer().Id());

            prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
            prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
            prod.Name(prodInfo["name"]);

            theTargetInfo->Producer(prod);
        }

    }

    /*
     * Set target parameter to potential temperature
     * - name T-C
     * - univ_id 4
     * - grib2 descriptor 0'00'000
     *
     * We need to specify grib and querydata parameter information
     * since we don't know which one will be the output format.
     *
     */

    vector<param> theParams;

    param theRequestedParam("T-C", 4);

    theRequestedParam.GribDiscipline(0);
    theRequestedParam.GribCategory(0);
    theRequestedParam.GribParameter(0);

    theParams.push_back(theRequestedParam);

    theTargetInfo->Params(theParams);

    /*
     * Create data structures.
     */

    theTargetInfo->Create(conf->ScanningMode(), false);

    /*
     * Initialize parent class functions for dimension handling
     */

    Dimension(conf->LeadingDimension());
    FeederInfo(theTargetInfo->Clone());
    FeederInfo()->Param(theRequestedParam);

    /*
     * Each thread will have a copy of the target info.
     */

    vector<shared_ptr<info> > theTargetInfos;

    theTargetInfos.resize(threadCount);

    for (size_t i = 0; i < threadCount; i++)
    {

        itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

        theTargetInfos[i] = theTargetInfo->Clone();

        boost::thread* t = new boost::thread(&tk2tc::Run,
                                             this,
                                             theTargetInfos[i],
                                             conf,
                                             i + 1);

        g.add_thread(t);

    }

    g.join_all();

    if (conf->WholeFileWrite())
    {

        shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

        theTargetInfo->FirstTime();
        string theOutputFile = "himan_" + theTargetInfo->Time().OriginDateTime()->String("%Y%m%d%H");
        theWriter->ToFile(theTargetInfo, conf->OutputFileType(), false, theOutputFile);

    }

}

void tk2tc::Run(shared_ptr<info> myTargetInfo,
                shared_ptr<const configuration> conf,
                unsigned short theThreadIndex)
{

    while (AdjustLeadingDimension(myTargetInfo))
    {
        Calculate(myTargetInfo, conf, theThreadIndex);
    }
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void tk2tc::Calculate(shared_ptr<info> myTargetInfo,
                      shared_ptr<const configuration> conf,
                      unsigned short theThreadIndex)
{


    shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

    // Required source parameters

    param TParam("T-K");

    unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (theThreadIndex)));

    ResetNonLeadingDimension(myTargetInfo);

    myTargetInfo->FirstParam();

    while (AdjustNonLeadingDimension(myTargetInfo))
    {

        myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
                                " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

        myTargetInfo->Data()->Resize(conf->Ni(), conf->Nj());

        // Source info for T

        shared_ptr<info> TInfo;

        try
        {
        	TInfo = theFetcher->Fetch(conf,
                                 myTargetInfo->Time(),
                                 myTargetInfo->Level(),
                                 TParam);

        	assert(TInfo->Param().Unit() == kK);

        }
        catch (HPExceptionType e)
        {
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing);
					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
        }

        int missingCount = 0;
        int count = 0;

    	shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
    	shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());

        bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid());

#ifdef HAVE_CUDA

        //if (itsUseCuda && equalGrids)
        if (itsUseCuda && equalGrids && theThreadIndex == 1)
        {
            size_t N = TGrid->Size();

            float* TOut = new float[N];

            tk2tc_cuda::doCuda(TGrid->DataPool()->Data(), TOut, N, theThreadIndex-1);

            double *data = new double[N];

            for (size_t i = 0; i < N; i++)
            {
                data[i] = static_cast<float> (TOut[i]);

                if (data[i] == kFloatMissing)
                {
                    missingCount++;
                }

                count++;
            }

            myTargetInfo->Data()->Set(data, N);

            delete [] data;
            delete [] TOut;

        }
        else
        {

#else
        if (true)
        {
#endif

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

                InterpolateToPoint(targetGrid, TGrid, equalGrids, T);

                if (T == kFloatMissing)
                {
                    missingCount++;

                    myTargetInfo->Value(kFloatMissing);
                    continue;
                }

                double TC = T - 273.15;

                if (!myTargetInfo->Value(TC))
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

        if (!conf->WholeFileWrite())
        {
            shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

            theWriter->ToFile(myTargetInfo->Clone(), conf->OutputFileType(), true);
        }
    }
}
