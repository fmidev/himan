/*
 * vvms.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "vvms.h"
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

#ifdef HAVE_CUDA
namespace himan
{
namespace plugin
{
namespace vvms_cuda
{
void doCuda(const float* Tin, float TBase, const float* Pin, float PScale, const float* VVin, float* VVout, size_t N, float PConst, unsigned short deviceIndex);
}
}
}
#endif

vvms::vvms() : itsUseCuda(false)
{
    itsClearTextFormula = "w = -(ver) * 287 * T * (9.81*p)";

    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvms"));

}

void vvms::Process(std::shared_ptr<const configuration> conf,
		std::shared_ptr<info> targetInfo)
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
     * Get producer information from neons if whole_file_write is false.
     */

    if (!conf->WholeFileWrite())
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
     * (todo: we could check from conf but why bother?)
     *
     */

    vector<param> theParams;

    param theRequestedParam ("VV-MS", 143);

    theRequestedParam.GribDiscipline(0);
    theRequestedParam.GribCategory(2);
    theRequestedParam.GribParameter(9);

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

        boost::thread* t = new boost::thread(&vvms::Run,
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

        string theOutputFile = "himan_" + targetInfo->Param().Name() + "_" + targetInfo->Time().OriginDateTime()->String("%Y%m%d%H");
        theWriter->ToFile(targetInfo, conf->OutputFileType(), false, theOutputFile);

    }
}

void vvms::Run(shared_ptr<info> myTargetInfo,
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

void vvms::Calculate(shared_ptr<info> myTargetInfo,
                     shared_ptr<const configuration> conf,
                     unsigned short theThreadIndex)
{


    shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

    // Required source parameters

    param TParam("T-K");
    param PParam("P-PA");
    param VVParam("VV-PAS");

    unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvmsThread #" + boost::lexical_cast<string> (theThreadIndex)));

    ResetNonLeadingDimension(myTargetInfo);

    myTargetInfo->FirstParam();

    while (AdjustNonLeadingDimension(myTargetInfo))
    {

        myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
                                " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

        //myTargetInfo->Data()->Resize(conf->Ni(), conf->Nj());

        double PScale = 1;
        double TBase = 0;

        /*
         * If vvms is calculated for pressure levels, the P value
         * equals to level value. Otherwise we have to fetch P
         * separately.
         */

        shared_ptr<info> PInfo;
        shared_ptr<info> VVInfo;
        shared_ptr<info> TInfo;

        shared_ptr<NFmiGrid> PGrid;

        bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

        try
        {
        	VVInfo = theFetcher->Fetch(conf,
        	                                  myTargetInfo->Time(),
        	                                  myTargetInfo->Level(),
        	                                  VVParam);

        	TInfo = theFetcher->Fetch(conf,
        	                                 myTargetInfo->Time(),
        	                                 myTargetInfo->Level(),
        	                                 TParam);

        	if (!isPressureLevel)
			{
				// Source info for P
				PInfo = theFetcher->Fetch(conf,
										  myTargetInfo->Time(),
										  myTargetInfo->Level(),
										  PParam);

				if (PInfo->Param().Unit() == kHPa)
				{
					PScale = 100;
				}

				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}
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

    	if (TInfo->Param().Unit() == kC)
    	{
    		TBase = 273.15;
    	}

        shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> VVGrid(VVInfo->Grid()->ToNewbaseGrid());

        int missingCount = 0;
        int count = 0;

        bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() &&
        					*myTargetInfo->Grid() == *VVInfo->Grid() &&
                           (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

#ifdef HAVE_CUDA

        if (itsUseCuda && equalGrids)
        {
            size_t N = TGrid->Size();

            float* VVout = new float[N];

            if (!isPressureLevel)
            {
                vvms_cuda::doCuda(TGrid->DataPool()->Data(), TBase, PGrid->DataPool()->Data(), PScale, VVGrid->DataPool()->Data(), VVout, N, 0, theThreadIndex-1);
            }
            else
            {
                vvms_cuda::doCuda(TGrid->DataPool()->Data(), TBase, 0, 0, VVGrid->DataPool()->Data(), VVout, N, 100 * myTargetInfo->Level().Value(), theThreadIndex-1);
            }

            double *data = new double[N];

            for (size_t i = 0; i < N; i++)
            {
                data[i] = static_cast<float> (VVout[i]);

                if (data[i] == kFloatMissing)
                {
                    missingCount++;
                }

                count++;
            }

            myTargetInfo->Data()->Set(data, N);

            delete [] data;
            delete [] VVout;

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
                double P = kFloatMissing;
                double VV = kFloatMissing;

                InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
                InterpolateToPoint(targetGrid, VVGrid, equalGrids, VV);

                if (isPressureLevel)
                {
                    P = 100 * myTargetInfo->Level().Value();
                }
                else
                {
                 	InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
                }

                if (T == kFloatMissing || P == kFloatMissing || VV == kFloatMissing)
                {
                    missingCount++;

                    myTargetInfo->Value(kFloatMissing);
                    continue;
                }

                double VVms = 287 * -VV * (T + TBase) / (9.81 * P * PScale);

                if (!myTargetInfo->Value(VVms))
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

            theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf->OutputFileType(), true);
        }
    }
}
