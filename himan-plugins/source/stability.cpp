/**
 * @file stability.cpp
 *
 *  @date: Jan 23, 2013
 *  @author aaltom, revised by partio
 */

#include "stability.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "metutil.h"
#include <algorithm> // for std::transform
#include <functional> // for std::plus
#include "NFmiGrid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "hitool.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

// Required source and target parameters and levels

const param TParam("T-K");
const param TDParam("TD-C");
const param HParam("Z-M2S2");
const params PParam({param("P-HPA"), param("P-PA")});
const param KIParam("KINDEX-N");
const param VTIParam("VTI-N");
const param CTIParam("CTI-N");
const param TTIParam("TTI-N");
const param SIParam("SI-N");
const param LIParam("LI-N");

const level P850Level(himan::kPressure, 850, "PRESSURE");
const level P700Level(himan::kPressure, 700, "PRESSURE");
const level P500Level(himan::kPressure, 500, "PRESSURE");
level groundLevel(himan::kHeight, 0, "HEIGHT");

void T500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const vector<double>& H0mVector, const vector<double>& H500mVector, vector<double>& result);
void TD500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const vector<double>& H0mVector, const vector<double>& H500mVector, vector<double>& result);

stability::stability() : itsLICalculation(true)
{
	itsClearTextFormula = "<multiple algorithms>";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("stability"));

}

void stability::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	vector<param> theParams;

	// Kindex

	param ki("KINDEX-N", 80, 0, 7, 2);
	theParams.push_back(ki);

	// Cross totals index
	param cti("CTI-N", 4751);
	theParams.push_back(cti);

	// Vertical Totals index
	param vti("VTI-N", 4754);
	theParams.push_back(vti);

	// Total Totals index
	param tti("TTI-N", 4755, 0, 7, 4);
	theParams.push_back(tti);

	// Showalter Index
	param si("SI-N", 4750, 0, 7, 13);
	theParams.push_back(si);

	if (itsConfiguration->Exists("li") && itsConfiguration->GetValue("li") == "false")
	{
		itsLICalculation = false;
	}
	else
	{
		// Lifted index
		param li("LI-N", 4751, 0, 7, 192);
		theParams.push_back(li);
	}

	/*
	if (itsConfiguration->Exists("srh") && itsConfiguration->GetValue("srh") == "true")
	{
		// Storm relative helicity 0 .. 1 km
		param hlcy("HLCY-1-M2S2", 4773);
		theParams.push_back(hlcy);

		// Storm relative helicity 0 .. 3 km
		hlcy = param("HLCY-M2S2", 4772, 0, 7, 8);
		theParams.push_back(hlcy);
	}
	*/
	
	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void stability::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	vector<double> T500mVector, TD500mVector, P500mVector;
	
	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("stabilityThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	//bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(theThreadIndex);
	bool useCudaInThisThread = false;
	
	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		shared_ptr<info> T850Info;
		shared_ptr<info> T700Info;
		shared_ptr<info> T500Info;
		shared_ptr<info> TD850Info;
		shared_ptr<info> TD700Info;
		
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		bool LICalculation = itsLICalculation;

		if (!GetSourceData(T850Info, T700Info, T500Info, TD850Info, TD700Info, myTargetInfo, useCudaInThisThread))
		{
			itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()) + " param KI");

			for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
			{
				myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value
			}

			if (itsConfiguration->StatisticsEnabled())
			{
				itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
				itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
			}

			continue;
		}
		
		if (LICalculation)
		{
			if (!GetLISourceData(myTargetInfo, T500mVector, TD500mVector, P500mVector, useCudaInThisThread))
			{
				itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()) + " param LI");
				myTargetInfo->Param(param("LI-N"));
				myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

				if (itsConfiguration->StatisticsEnabled())
				{
					itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
					itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
				}

				LICalculation = false;
			}
		}
		
		bool equalGrids = CompareGrids({myTargetInfo->Grid(), T850Info->Grid(), T700Info->Grid(), T500Info->Grid(), TD850Info->Grid(), TD700Info->Grid()});

		size_t missingCount = 0;
		size_t count = 0;

		string deviceType = "CPU";

#ifdef HAVE_CUDA

		if (!equalGrids)
		{
			myThreadedLogger->Debug("Unpacking for CPU calculation");
			Unpack({T500Info, T700Info, T850Info, TD700Info, TD850Info});
		}

		else if (useCudaInThisThread)
		{
			deviceType = "GPU";

			unique_ptr<stability_cuda::options> opts(new stability_cuda::options);

			opts->t500 = T500Info->ToSimple();
			opts->t700 = T700Info->ToSimple();
			opts->t850 = T850Info->ToSimple();
			opts->td700 = TD700Info->ToSimple();
			opts->td850 = TD850Info->ToSimple();

			myTargetInfo->Param(param("KINDEX-N"));
			opts->ki = myTargetInfo->ToSimple();

			myTargetInfo->Param(param("VTI-N"));
			opts->vti = myTargetInfo->ToSimple();

			myTargetInfo->Param(param("CTI-N"));
			opts->cti = myTargetInfo->ToSimple();

			myTargetInfo->Param(param("TTI-N"));
			opts->tti = myTargetInfo->ToSimple();

			myTargetInfo->Param(param("SI-N"));
			opts->si = myTargetInfo->ToSimple();

			opts->t500m = &T500mVector[0];
			opts->td500m = &TD500mVector[0];
			opts->p500m = &P500mVector[0];

			if (LICalculation)
			{
				myTargetInfo->Param(param("LI-N"));
				opts->li = myTargetInfo->ToSimple();
			}
	
			opts->N = opts->t500->size_x * opts->t500->size_y;

			stability_cuda::Process(*opts);

			count = opts->N;
			missingCount = opts->missing;

			CudaFinish(move(opts), myTargetInfo, T500Info, T700Info, T850Info, TD700Info, TD850Info);

		}
		else
#endif
		{
			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> T700Grid(T700Info->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> T500Grid(T500Info->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> TD850Grid(TD850Info->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> TD700Grid(TD700Info->Grid()->ToNewbaseGrid());

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T850 = kFloatMissing;
				double T700 = kFloatMissing;
				double T500 = kFloatMissing;
				double TD850 = kFloatMissing;
				double TD700 = kFloatMissing;

				InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
				assert(T850 > 0);

				InterpolateToPoint(targetGrid, T700Grid, equalGrids, T700);
				assert(T700 > 0);

				InterpolateToPoint(targetGrid, T500Grid, equalGrids, T500);
				assert(T500 > 0);

				InterpolateToPoint(targetGrid, TD850Grid, equalGrids, TD850);
				assert(TD850 > 0);

				InterpolateToPoint(targetGrid, TD700Grid, equalGrids, TD700);
				assert(TD700 > 0);
				
				double value = kFloatMissing;

				if (T850 == kFloatMissing || T700 == kFloatMissing || T500 == kFloatMissing || TD850 == kFloatMissing || TD700 == kFloatMissing)
				{
					missingCount++;

					for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
					{
						myTargetInfo->Value(kFloatMissing);
					}
				}
				else
				{

					value = KI(T850, T700, T500, TD850, TD700) - constants::kKelvin;
					myTargetInfo->Param(KIParam);
					myTargetInfo->Value(value);

					value = CTI(T500, TD850);
					myTargetInfo->Param(CTIParam);
					myTargetInfo->Value(value);

					value = VTI(T850, T500);
					myTargetInfo->Param(VTIParam);
					myTargetInfo->Value(value);

					value = TTI(T850, T500, TD850);
					myTargetInfo->Param(TTIParam);
					myTargetInfo->Value(value);
				
					value = SI(T850, T500, TD850);
					myTargetInfo->Param(SIParam);
					myTargetInfo->Value(value);

					if (LICalculation)
					{
						size_t locationIndex = myTargetInfo->LocationIndex();

						double T500m = T500mVector[locationIndex];
						double TD500m = TD500mVector[locationIndex];
						double P500m = P500mVector[locationIndex];

						assert(T500m != kFloatMissing);
						assert(TD500m != kFloatMissing);
						assert(P500m != kFloatMissing);

						if (T500m == kFloatMissing || TD500m == kFloatMissing || P500m == kFloatMissing)
						{
							missingCount++;
							value = kFloatMissing;
						}
						else
						{
							value = LI(T500, T500m, TD500m, P500m);
						}

						myTargetInfo->Param(LIParam);
						myTargetInfo->Value(value);
					}
				}
			}

			if (itsConfiguration->StatisticsEnabled())
			{
				itsConfiguration->Statistics()->AddToMissingCount(missingCount);
				itsConfiguration->Statistics()->AddToValueCount(count);
			}

			/*
			 * Now we are done for this level
			 *
			 * Clone info-instance to writer since it might change our descriptor places
			 */

			myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));
		}

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			SwapTo(myTargetInfo, kBottomLeft);
		}

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}


inline
double stability::CTI(double TD850, double T500) const
{
	return TD850 - T500;
}

inline
double stability::VTI(double T850, double T500) const
{
	return T850 - T500;
}

inline
double stability::TTI(double T850, double T500, double TD850) const
{
	return CTI(TD850, T500) + VTI(T850, T500);
}

inline
double stability::KI(double T850, double T700, double T500, double TD850, double TD700) const
{
	return T850 - T500 + TD850 - (T700 - TD700);
}

inline
double stability::LI(double T500, double T500m, double TD500m, double P500m) const
{
	lcl_t LCL = metutil::LCL_(50000, T500m, TD500m);

	double li = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL.P == kFloatMissing)
	{
		return li;
	}

	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = metutil::DryLift_(P500m, T500m, TARGET_PRESSURE);

		if (dryT != kFloatMissing)
		{
			li = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud

		double wetT = metutil::MoistLift_(P500m, T500m, TD500m, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			li = T500 - wetT;
		}
	}

	return li;
}

inline
double stability::SI(double T850, double T500, double TD850) const
{
	lcl_t LCL = metutil::LCL_(85000, T850, TD850);

	double si = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL.P == kFloatMissing)
	{
		return si;
	}
	
	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = metutil::DryLift_(85000, T850, TARGET_PRESSURE);
		
		if (dryT != kFloatMissing)
		{
			si = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud
		
		double wetT = metutil::MoistLift_(85000, T850, TD850, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			si = T500 - wetT;
		}
	}

	return si;
}

void T500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const vector<double>& H0mVector, const vector<double>& H500mVector, vector<double>& result)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	h->Time(ftime);

#ifdef DEBUG
	assert(H0mVector.size() == H500mVector.size());
	
	for (size_t i = 0; i < result.size(); i++)
	{
		assert(H0mVector[i] != kFloatMissing);
		assert(H500mVector[i] != kFloatMissing);
	}
#endif

	result = h->VerticalAverage(param("T-K"), H0mVector, H500mVector);

#ifdef DEBUG
	for (size_t i = 0; i < result.size(); i++)
	{
		assert(result[i] != kFloatMissing);
	}
#endif
}

void TD500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const vector<double>& H0mVector, const vector<double>& H500mVector, vector<double>& result)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	h->Time(ftime);

	result = h->VerticalAverage(param("TD-C"), H0mVector, H500mVector);

}

/*
inline
double si::StormRelativeHelicity(double UID, double VID, double U_lower, double U_higher, double V_lower, double V_higher)
{
	return ((UID - U_lower) * (V_lower - V_higher)) - ((VID - V_lower) * (U_lower - U_higher));
}
*/

#ifdef HAVE_CUDA

void stability::CudaFinish(unique_ptr<stability_cuda::options> opts, shared_ptr<info>& myTargetInfo,
	shared_ptr<info>& T500Info, shared_ptr<info>& T700Info, shared_ptr<info>& T850Info, shared_ptr<info>& TD700Info, shared_ptr<info>& TD850Info)
{
	// Copy data back to infos

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
	{
		string parmName = myTargetInfo->Param().Name();

		if (parmName == "KINDEX-N")
		{
			CopyDataFromSimpleInfo(myTargetInfo, opts->ki, false);
		}
		else if (parmName == "SI-N")
		{
			CopyDataFromSimpleInfo(myTargetInfo, opts->si, false);
		}
		else if (parmName == "LI-N")
		{
			CopyDataFromSimpleInfo(myTargetInfo, opts->li, false);
		}
		else if (parmName == "CTI-N")
		{
			CopyDataFromSimpleInfo(myTargetInfo, opts->cti, false);
		}
		else if (parmName == "VTI-N")
		{
			CopyDataFromSimpleInfo(myTargetInfo, opts->vti, false);
		}
		else if (parmName == "TTI-N")
		{
			CopyDataFromSimpleInfo(myTargetInfo, opts->tti, false);
		}

		SwapTo(myTargetInfo, T500Info->Grid()->ScanningMode());
	}

	if (T500Info->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(T500Info, opts->t500, true);
	}

	if (T700Info->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(T700Info, opts->t700, true);
	}

	if (T850Info->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(T850Info, opts->t850, true);
	}

	if (TD700Info->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(TD700Info, opts->td700, true);
	}

	if (TD850Info->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(TD850Info, opts->td850, true);
	}

	// opts is destroyed after leaving this function
}

#endif

bool stability::GetSourceData(shared_ptr<info>& T850Info, shared_ptr<info>& T700Info, shared_ptr<info>& T500Info, shared_ptr<info>& TD850Info, shared_ptr<info>& TD700Info, const shared_ptr<info>& myTargetInfo, bool useCudaInThisThread)
{
	bool ret = true;

	try
	{
		auto theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

		if (!T850Info)
		{
			T850Info = theFetcher->Fetch(itsConfiguration,
						myTargetInfo->Time(),
						P850Level,
						TParam,
						itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		}

		if (!T700Info)
		{
			T700Info = theFetcher->Fetch(itsConfiguration,
						myTargetInfo->Time(),
						P700Level,
						TParam,
						itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		}

		if (!T500Info)
		{
			T500Info = theFetcher->Fetch(itsConfiguration,
						myTargetInfo->Time(),
						P500Level,
						TParam,
						itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		}

		if (!TD850Info)
		{
			TD850Info = theFetcher->Fetch(itsConfiguration,
						myTargetInfo->Time(),
						P850Level,
						TDParam,
						itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		}

		if (!TD700Info)
		{
			TD700Info = theFetcher->Fetch(itsConfiguration,
						myTargetInfo->Time(),
						P700Level,
						TDParam,
						itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		}

		assert(T850Info);
		assert(T700Info);
		assert(T500Info);
		assert(TD850Info);
		assert(TD700Info);

	}
	catch (HPExceptionType& e)
	{
		switch (e)
		{
			case kFileDataNotFound:

				ret = false;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
		}
	}

	return ret;
}

bool stability::GetLISourceData(const shared_ptr<info>& myTargetInfo, vector<double>& T500mVector, vector<double>& TD500mVector, vector<double>& P500mVector, bool useCudaInThisThread)
{
	bool ret = true;
		
	try
	{
		auto theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

		// Fetch Z uncompressed since it is not transferred to cuda

		auto HInfo = theFetcher->Fetch(itsConfiguration,
				myTargetInfo->Time(),
				groundLevel,
				HParam,
				false);


		auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

		h->Configuration(itsConfiguration);

		vector<double> H0mVector = HInfo->Grid()->Data()->Values();
		vector<double> H500mVector(HInfo->SizeLocations());

		for (size_t i = 0; i < H500mVector.size(); i++)
		{
			// H0mVector contains the height of ground (compared to MSL). Height can be negative
			// (maybe even in real life (Netherlands?)), but in our case we use 0 as smallest height.
			// TODO: check how it is in smarttools

			H0mVector[i] *= constants::kIg;

			if (H0mVector[i] < 0)
			{
				H0mVector[i] = 0;
			}
			
			H500mVector[i] = H0mVector[i] + 500.;
		}

		h->Time(myTargetInfo->Time());

		// Fetch average values of T, TD and P over vertical height range 0 ... 500m OVER GROUND

		boost::thread t1(&T500mSearch, itsConfiguration, myTargetInfo->Time(), H0mVector, H500mVector, boost::ref(T500mVector));
		boost::thread t2(&TD500mSearch, itsConfiguration, myTargetInfo->Time(), H0mVector, H500mVector, boost::ref(TD500mVector));

		P500mVector = h->VerticalAverage(PParam, H0mVector, H500mVector);

		assert(P500mVector[0] != kFloatMissing);

		if (P500mVector[0] < 1500)
		{
			transform(P500mVector.begin(), P500mVector.end(), P500mVector.begin(), bind1st(multiplies<double>(), 100)); // hPa to Pa
		}
	
		t1.join(); t2.join();

	}
	catch (HPExceptionType& e)
	{
		switch (e)
		{
			case kFileDataNotFound:

				ret = false;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
		}
	}

	return ret;
}
