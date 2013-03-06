/**
 * @file tk2tc.cpp
 *
 * @dateNov 20, 2012
 * @author partio
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

#include "cuda_extern.h"

tk2tc::tk2tc() : itsUseCuda(false), itsCudaDeviceCount(0)
{
    itsClearTextFormula = "Tc = Tk - 273.15";

    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("tk2tc"));
}

void tk2tc::Process(std::shared_ptr<const plugin_configuration> conf)
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

        itsCudaDeviceCount = c->DeviceCount();
		
    }

    // Get number of threads to use

    unsigned short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->Statistics()->Enabled())
	{
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedCudaCount(itsCudaDeviceCount);
	}
	
    boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

    /*
     * Get producer information from neons if whole_file_write is false.
     */

    if (!conf->WholeFileWrite())
    {
        shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

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

    targetInfo->Params(theParams);

    /*
     * Create data structures.
     */

    targetInfo->Create();

    /*
     * Initialize parent class functions for dimension handling
     */

    Dimension(conf->LeadingDimension());
    FeederInfo(shared_ptr<info> (new info(*targetInfo)));
    FeederInfo()->Param(theRequestedParam);

    /*
     * Each thread will have a copy of the target info.
     */

    vector<shared_ptr<info> > targetInfos;

    targetInfos.resize(threadCount);

    for (size_t i = 0; i < threadCount; i++)
    {

        itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

        targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

        boost::thread* t = new boost::thread(&tk2tc::Run,
                                             this,
                                             targetInfos[i],
                                             conf,
                                             i + 1);

        g.add_thread(t);

    }

    g.join_all();

    if (conf->WholeFileWrite())
    {

        shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

        targetInfo->FirstTime();
        string theOutputFile = "himan_" + targetInfo->Time().OriginDateTime()->String("%Y%m%d%H");
        theWriter->ToFile(targetInfo, conf->OutputFileType(), false, theOutputFile);

    }

}

void tk2tc::Run(shared_ptr<info> myTargetInfo,
                shared_ptr<const plugin_configuration> conf,
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

void tk2tc::Calculate(shared_ptr<info> myTargetInfo,
                      shared_ptr<const plugin_configuration> conf,
                      unsigned short threadIndex)
{
    shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

    // Required source parameters

    param TParam("T-K");

    unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (threadIndex)));

    ResetNonLeadingDimension(myTargetInfo);

    myTargetInfo->FirstParam();

    while (AdjustNonLeadingDimension(myTargetInfo))
    {

        myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
                                " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

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

#ifdef DEBUG
        unique_ptr<timer> t = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
        t->Start();
#endif
        string deviceType;

        if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
        {
	
	    deviceType = "GPU";

            size_t N = TGrid->Size();

            float* TOut = new float[N]; // array that cuda devices will store data
            double* infoData = new double[N]; // array that's stored to info instance

            tk2tc_cuda::DoCuda(TGrid->DataPool()->Data(), TOut, N, threadIndex-1);

            for (size_t i = 0; i < N; i++)
            {
                infoData[i] = static_cast<float> (TOut[i]);

                if (infoData[i] == kFloatMissing)
                {
                    missingCount++;
                }

                count++;
            }

            myTargetInfo->Data()->Set(infoData, N);

            delete [] infoData;
            delete [] TOut;

        }
        else
        {

			deviceType = "CPU";

            assert(targetGrid->Size() == myTargetInfo->Data()->Size());

            myTargetInfo->ResetLocation();

            targetGrid->Reset();

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

	}
	
#ifdef DEBUG
        t->Stop();
        itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (t->GetTime()) + " microseconds on " + deviceType);
#endif


        /*
         * Now we are done for this level
         *
         * Clone info-instance to writer since it might change our descriptor places
         */

        myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (true)
		{
			conf->Statistics()->AddToMissingCount(missingCount);
			conf->Statistics()->AddToValueCount(count);
		}

        if (!conf->WholeFileWrite())
        {
            shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

            theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf->OutputFileType(), true);
        }
    }
}
