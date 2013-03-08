/**
 * @file tpot.cpp
 *
 * @brief Plugin to calculate potential temperature
 *
 * Created on: Nov 20, 2012
 * @author partio
 */

#include "tpot.h"
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

#include "cuda_extern.h"

tpot::tpot() : itsUseCuda(false), itsCudaDeviceCount(0)
{
    itsClearTextFormula = "Tp = Tk * pow((1000/P), 0.286)"; // Poissons equation

    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpot"));

}

void tpot::Process(std::shared_ptr<const plugin_configuration> conf)
{

    shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

    if (c->HaveCuda())
    {
        string msg = "I possess the powers of CUDA ";

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

    boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

    /*
     * Get producer information from neons
     */

    if (conf->FileWriteOption() == kNeons)
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
     * - name TP-K
     * - univ_id 8
     * - grib2 descriptor 0'00'002
     *
     * We need to specify grib and querydata parameter information
     * since we don't know which one will be the output format.
     * (todo: we could check from conf but why bother?)
     *
     */

    vector<param> theParams;

    param theRequestedParam ("TP-K", 8);

    theRequestedParam.GribDiscipline(0);
    theRequestedParam.GribCategory(0);
    theRequestedParam.GribParameter(2);

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

        boost::thread* t = new boost::thread(&tpot::Run,
                                             this,
                                             targetInfos[i],
                                             conf,
                                             i + 1);

        g.add_thread(t);

    }

    g.join_all();

    if (conf->FileWriteOption() == kSingleFile)
    {

        shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

        string theOutputFile = conf->ConfigurationFile();

        theWriter->ToFile(targetInfo, conf->OutputFileType(), conf->FileWriteOption(), theOutputFile);

    }
}

void tpot::Run(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> conf, unsigned short threadIndex)
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

void tpot::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> conf, unsigned short threadIndex)
{

    shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

    // Required source parameters

    param TParam ("T-K");
    param PParam ("P-PA");

    unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (threadIndex)));

    ResetNonLeadingDimension(myTargetInfo);

    myTargetInfo->FirstParam();

    while (AdjustNonLeadingDimension(myTargetInfo))
    {

        myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
                                " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

        double PScale = 1;
        double TBase = 0;

        // Source infos
        shared_ptr<info> TInfo;
        shared_ptr<info> PInfo;

        shared_ptr<NFmiGrid> PGrid;

		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		try
		{

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

				if (PInfo->Param().Unit() == kPa)
				{
					PScale = 0.01;
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

        int missingCount = 0;
        int count = 0;

        assert(targetGrid->Size() == myTargetInfo->Data()->Size());

        bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

#ifdef DEBUG
        unique_ptr<timer> t = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
        t->Start();
#endif
        string deviceType;

        if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
        {
        	deviceType = "GPU";
	    
            size_t N = TGrid->Size();

            float* TPOut = new float[N];
            double* infoData = new double[N];

            if (!isPressureLevel)
            {
                tpot_cuda::DoCuda(TGrid->DataPool()->Data(), TBase, PGrid->DataPool()->Data(), PScale, TPOut, N, 0, threadIndex-1);
            }
            else
            {
                tpot_cuda::DoCuda(TGrid->DataPool()->Data(), TBase, 0, 0, TPOut, N, myTargetInfo->Level().Value(), threadIndex-1);
            }


            for (size_t i = 0; i < N; i++)
            {
                infoData[i] = static_cast<float> (TPOut[i]);

                if (infoData[i] == kFloatMissing)
                {
                    missingCount++;
                }

                count++;
            }

            myTargetInfo->Data()->Set(infoData, N);

            delete [] infoData;
            delete [] TPOut;

        }
        else
        {

        	deviceType = "CPU";
	    
            myTargetInfo->ResetLocation();

            targetGrid->Reset();

            while (myTargetInfo->NextLocation() && targetGrid->Next())
            {
                count++;

                double T = kFloatMissing;
                double P = kFloatMissing;

                InterpolateToPoint(targetGrid, TGrid, equalGrids, T);

                if (isPressureLevel)
                {
                    P = myTargetInfo->Level().Value();
                }
                else
                {
                    InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
                }

                if (T == kFloatMissing || P == kFloatMissing)
                {
                    missingCount++;

                    myTargetInfo->Value(kFloatMissing);
                    continue;
                }

                double Tp = (T + TBase) * pow((1000 / (P * PScale)), 0.286);

                if (!myTargetInfo->Value(Tp))
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
         * */

        myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

        if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
        {
            shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

            theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf->OutputFileType(), conf->FileWriteOption());
        }
    }
}
