/**
 * @file dewpoint.cpp
 *
 * @date Jan 21, 2012
 * @author partio
 */

#include "dewpoint.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"
#include "metutil.h"

using namespace std;
using namespace himan::plugin;

dewpoint::dewpoint()
{
	itsClearTextFormula = "Td = T / (1 - (T * ln(RH)*(Rw/L)))";
	itsCudaEnabledCalculation = true;

	itsLogger = logger_factory::Instance()->GetLog("dewpoint");

}

void dewpoint::Process(shared_ptr<const plugin_configuration> conf)
{

	Init(conf);
	
	/*
	 * Set target parameter to dewpoint.
	 *
	 */

	param requestedParam ("TD-C", 10, 0, 0, 6);
	requestedParam.Unit(kK);

	SetParams({requestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void dewpoint::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	const param TParam("T-K");
	const param RHParam("RH-PRCNT");

	auto myThreadedLogger = logger_factory::Instance()->GetLog("dewpointThread #" + boost::lexical_cast<string> (threadIndex));
	
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));
	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(threadIndex);
	
	double TBase = 0;
	double RHScale = 1;
		
	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
	info_t RHInfo = Fetch(forecastTime, forecastLevel, RHParam, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

	if (!TInfo || !RHInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	assert(TInfo->Grid()->AB() == RHInfo->Grid()->AB());
		
	SetAB(myTargetInfo, TInfo);

    // Special case for harmonie
    if (itsConfiguration->SourceProducer().Id() == 199)
    { 
		RHScale = 100;
	} 

	if (RHInfo->Param().Unit() != kPrcnt)
	{
		itsLogger->Warning("Unable to determine RH unit, assuming percent");
	}

	// Formula assumes T == Celsius

	if (TInfo->Param().Unit() == kC)
	{

		TBase = himan::constants::kKelvin;
	}

	string deviceType;

#ifdef HAVE_CUDA

	if (useCudaInThisThread)
	{

		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, TInfo, RHInfo); 

		dewpoint_cuda::Process(*opts);

	}
	else
#endif
	{
		deviceType = "CPU";
			
		LOCKSTEP(myTargetInfo, TInfo, RHInfo)
		{		

			double T = TInfo->Value();
			double RH = RHInfo->Value();


			if (T == kFloatMissing || RH == kFloatMissing)
			{
				continue;
			}

			T += TBase;
			RH *= RHScale;

			double TD = kFloatMissing;

			if (RH > 50)
			{
				TD = metutil::DewPointFromHighRH_(T, RH);
			}
			else
			{
				TD = metutil::DewPointFromLowRH_(T, RH);
			}

			myTargetInfo->Value(TD);

		}
	}
	
	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}

#ifdef HAVE_CUDA

unique_ptr<dewpoint_cuda::options> dewpoint::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> RHInfo)
{
	unique_ptr<dewpoint_cuda::options> opts(new dewpoint_cuda::options);

	opts->t = TInfo->ToSimple();
	opts->rh = RHInfo->ToSimple();
	opts->td = myTargetInfo->ToSimple();

	opts->N = opts->td->size_x * opts->td->size_y;

	if (TInfo->Param().Unit() == kC)
	{
		opts->t_base = himan::constants::kKelvin;
	}

	if (RHInfo->Param().Unit() != kPrcnt || itsConfiguration->SourceProducer().Id() == 199)
	{
		// If unit cannot be detected, assume the values are from 0 .. 1
		opts->rh_scale = 100;
	}

	return opts;
}

#endif
