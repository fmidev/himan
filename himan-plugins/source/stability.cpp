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
#include <boost/foreach.hpp>
#include "metutil.h"
#include <algorithm> // for std::transform
#include <functional> // for std::plus
#include "level.h"
#include "forecast_time.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"
#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

pair<vector<double>, vector<double>> Shear(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const param& wantedParam, double lowerHeight, double upperHeight);

#ifdef DEBUG
void DumpVector(const vector<double>& vec);
#endif

// Required source and target parameters and levels

const param TParam("T-K");
const param TDParam("TD-C");
const param HParam("Z-M2S2");
const params PParam({param("P-HPA"), param("P-PA")});
const param KIParam("KINDEX-N", 80, 0, 7, 2);
const param VTIParam("VTI-N", 4754);
const param CTIParam("CTI-N", 4751);
const param TTIParam("TTI-N", 4755, 0, 7, 4);
const param SIParam("SI-N", 4750, 0, 7, 13);
const param LIParam("LI-N", 4751, 0, 7, 192);
const param BS01Param("WSH-1-KT", 4771); // knots!
const param BS06Param("WSH-KT", 4770); // knots!
const param SRH01Param("HLCY-1-M2S2", 4773);
const param SRH03Param("HLCY-M2S2",  4772, 0, 7, 8);

const level P850Level(himan::kPressure, 850, "PRESSURE");
const level P700Level(himan::kPressure, 700, "PRESSURE");
const level P500Level(himan::kPressure, 500, "PRESSURE");
level groundLevel(himan::kHeight, 0, "HEIGHT");

void T500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const vector<double>& H0mVector, const vector<double>& H500mVector, vector<double>& result);
void TD500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const vector<double>& H0mVector, const vector<double>& H500mVector, vector<double>& result);

stability::stability() : itsLICalculation(false), itsBSCalculation(false)
{
	itsClearTextFormula = "<multiple algorithms>";

	itsLogger = logger_factory::Instance()->GetLog("stability");

}

void stability::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	vector<param> theParams;

	// Kindex
	theParams.push_back(KIParam);

	// Cross totals index
	theParams.push_back(CTIParam);

	// Vertical Totals index
	theParams.push_back(VTIParam);

	// Total Totals index
	theParams.push_back(TTIParam);

	// Showalter Index
	theParams.push_back(SIParam);

	if (itsConfiguration->Exists("li") && itsConfiguration->GetValue("li") == "true")
	{
		// Lifted index

		itsLICalculation = true;
		theParams.push_back(LIParam);
	}

	if (itsConfiguration->Exists("bs") && itsConfiguration->GetValue("bs") == "true")
	{

		itsBSCalculation = true;

		// Bulk shear 0 .. 1 km
		theParams.push_back(BS01Param);

		// Bulk shear 0 .. 6 km

		theParams.push_back(BS06Param);
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

	vector<double> T500mVector, TD500mVector, P500mVector, U01Vector, V01Vector, U06Vector, V06Vector;
	
	auto myThreadedLogger = logger_factory::Instance()->GetLog("stabilityThread #" + boost::lexical_cast<string> (theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	//bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(theThreadIndex);
	bool useCudaInThisThread = false;
	
	info_t T850Info, T700Info, T500Info, TD850Info, TD700Info;
		
	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	bool LICalculation = itsLICalculation;
	bool BSCalculation = itsBSCalculation;

	if (!GetSourceData(T850Info, T700Info, T500Info, TD850Info, TD700Info, myTargetInfo, useCudaInThisThread))
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}
		
	if (LICalculation)
	{
		if (!GetLISourceData(myTargetInfo, T500mVector, TD500mVector, P500mVector))
		{
			myThreadedLogger->Warning("Source data not found for param LI");
			LICalculation = false;
		}
	}

	if (BSCalculation)
	{
		if (!GetWindShearSourceData(myTargetInfo, U01Vector, V01Vector, U06Vector, V06Vector))
		{
			myThreadedLogger->Warning("Source data not found for param BulkShear");
			BSCalculation = false;
		}
	}

	string deviceType = "CPU";

#ifdef HAVE_CUDA

	if (useCudaInThisThread)
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

		if (LICalculation)
		{
			opts->t500m = &T500mVector[0];
			opts->td500m = &TD500mVector[0];
			opts->p500m = &P500mVector[0];

			myTargetInfo->Param(param("LI-N"));
			opts->li = myTargetInfo->ToSimple();
		}

		opts->N = opts->t500->size_x * opts->t500->size_y;

		stability_cuda::Process(*opts);

		CudaFinish(move(opts), myTargetInfo, T500Info, T700Info, T850Info, TD700Info, TD850Info);

	}
	else
#endif
	{

		LOCKSTEP(myTargetInfo, T850Info, T700Info, T500Info, TD850Info, TD700Info)
		{

			double T850 = T850Info->Value();
			double T700 = T700Info->Value();
			double T500 = T500Info->Value();
			double TD850 = TD850Info->Value();
			double TD700 = TD700Info->Value();

			assert(T850 > 0);
			assert(T700 > 0);
			assert(T500 > 0);
			assert(TD850 > 0);
			assert(TD700 > 0);

			double value = kFloatMissing;

			if (T850 == kFloatMissing || T700 == kFloatMissing || T500 == kFloatMissing || TD850 == kFloatMissing || TD700 == kFloatMissing)
			{
				continue;
			}

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

				if (T500m != kFloatMissing && TD500m != kFloatMissing && P500m != kFloatMissing)
				{
					value = LI(T500, T500m, TD500m, P500m);

					myTargetInfo->Param(LIParam);
					myTargetInfo->Value(value);
				}
			}

			if (BSCalculation)
			{
				size_t locationIndex = myTargetInfo->LocationIndex();

				double U01 = U01Vector[locationIndex];
				double V01 = V01Vector[locationIndex];
				double U06 = U06Vector[locationIndex];
				double V06 = V06Vector[locationIndex];
				
				assert(U01 != kFloatMissing);
				assert(V01 != kFloatMissing);
				
				assert(U06 != kFloatMissing);
				assert(V06 != kFloatMissing);

				if (U01 != kFloatMissing && V01 != kFloatMissing)
				{
					value = BulkShear(U01, V01);

					myTargetInfo->Param(BS01Param);
					myTargetInfo->Value(value);
				}

				if (U06 != kFloatMissing && V01 != kFloatMissing)
				{
					value = BulkShear(U06, V06);

					myTargetInfo->Param(BS06Param);
					myTargetInfo->Value(value);
				}
			}
		}
	}
	
	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));
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

inline
double stability::BulkShear(double U, double V)
{
	return sqrt(U*U + V*V) * 1.943844492; // converting to knots
	
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
#if 0
inline
double stability::StormRelativeHelicity(double UID, double VID, double U_lower, double U_higher, double V_lower, double V_higher)
{
	return ((UID - U_lower) * (V_lower - V_higher)) - ((VID - V_lower) * (U_lower - U_higher));
}
#endif

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

	if (!T850Info)
	{
		T850Info = Fetch(myTargetInfo->Time(), P850Level, TParam, useCudaInThisThread);
	}

	if (!T700Info)
	{
		T700Info = Fetch(myTargetInfo->Time(), P700Level, TParam, useCudaInThisThread);
	}

	if (!T500Info)
	{
		T500Info = Fetch(myTargetInfo->Time(), P500Level, TParam, useCudaInThisThread);
	}

	if (!TD850Info)
	{
		TD850Info = Fetch(myTargetInfo->Time(), P850Level, TDParam, useCudaInThisThread);
	}

	if (!TD700Info)
	{
		TD700Info = Fetch(myTargetInfo->Time(),	P700Level, TDParam, useCudaInThisThread);
	}

	if (!T850Info || !T700Info || !T500Info || !TD850Info || !TD700Info)
	{
		ret = false;
	}

	return ret;
}

bool stability::GetLISourceData(const shared_ptr<info>& myTargetInfo, vector<double>& T500mVector, vector<double>& TD500mVector, vector<double>& P500mVector)
{
		
	// Fetch Z uncompressed since it is not transferred to cuda

	auto HInfo = Fetch(myTargetInfo->Time(), groundLevel, HParam, false);

	if (!HInfo)
	{
		return false;
	}

	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);

	vector<double> H0mVector = HInfo->Grid()->Data()->Values();
	vector<double> H500mVector(HInfo->SizeLocations());

	for (size_t i = 0; i < H500mVector.size(); i++)
	{
		// H0mVector contains the height of ground (compared to MSL). Height can be negative
		// (maybe even in real life (Netherlands?)), but in our case we use 0 as smallest height.
		// TODO: check how it is in smarttools

		if (H0mVector[i] == kFloatMissing)
		{
			continue;
		}
		
		H0mVector[i] *= constants::kIg;
		H0mVector[i] = fmax(0, H0mVector[i]);

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

	return true;
}

bool stability::GetWindShearSourceData(const shared_ptr<info>& myTargetInfo, vector<double>& U01Vector, vector<double>& V01Vector, vector<double>& U06Vector, vector<double>& V06Vector)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	// BS 0-6
	auto Us = Shear(itsConfiguration, myTargetInfo->Time(), param("U-MS"), 0, 6000);
	
	auto lowerU = Us.first;
	auto upperU = Us.second;


	auto Vs = Shear(itsConfiguration, myTargetInfo->Time(), param("V-MS"), 0, 6000);

	auto lowerV = Vs.first;
	auto upperV = Vs.second;

#ifdef YES_WE_HAVE_GCC_WHICH_SUPPORTS_LAMBDAS
	transform(lowerU.begin(), lowerU.end(), upperU.begin(), back_inserter(U06Vector), [](double l, double u) { return (u == kFloatMissing || l == kFloatMissing) ? kFloatMissing : u - l; });
	transform(lowerV.begin(), lowerV.end(), upperV.begin(), back_inserter(V06Vector), [](double l, double u) { return (u == kFloatMissing || l == kFloatMissing) ? kFloatMissing : u - l; });
#else
	U06Vector.resize(lowerU.size(), kFloatMissing);
	V06Vector.resize(lowerU.size(), kFloatMissing);

	for (size_t i = 0; i < lowerU.size(); i++)
	{
		double l = lowerU[i];
		double u = lowerV[i];

		if (u == kFloatMissing || l == kFloatMissing)
		{
			continue;
		}

		U06Vector[i] = u - l;

		l = lowerV[i];
		u = upperV[i];

		if (u == kFloatMissing || l == kFloatMissing)
		{
			continue;
		}

		V06Vector[i] = u - l;
	}
#endif

#ifdef DEBUG
	DumpVector(U06Vector);
	DumpVector(V06Vector);
#endif

	// BS 0-1

	Us = Shear(itsConfiguration, myTargetInfo->Time(), param("U-MS"), 0, 1000);
	upperU = Us.second;

	Vs = Shear(itsConfiguration, myTargetInfo->Time(), param("V-MS"), 0, 1000);
	upperV = Vs.second;

#ifdef YES_WE_HAVE_GCC_WHICH_SUPPORTS_LAMBDAS
	transform(lowerU.begin(), lowerU.end(), upperU.begin(), back_inserter(U01Vector), [](double l, double u) { return (u == kFloatMissing || l == kFloatMissing) ? kFloatMissing : u - l; });
	transform(lowerV.begin(), lowerV.end(), upperV.begin(), back_inserter(V01Vector), [](double l, double u) { return (u == kFloatMissing || l == kFloatMissing) ? kFloatMissing : u - l; });
#else
	U01Vector.resize(lowerU.size(), kFloatMissing);
	V01Vector.resize(lowerU.size(), kFloatMissing);

	for (size_t i = 0; i < lowerU.size(); i++)
	{
		double l = lowerU[i];
		double u = lowerV[i];

		if (u == kFloatMissing || l == kFloatMissing)
		{
			continue;
		}

		U01Vector[i] = u - l;

		l = lowerV[i];
		u = upperV[i];

		if (u == kFloatMissing || l == kFloatMissing)
		{
			continue;
		}

		V01Vector[i] = u - l;
	}
#endif

#ifdef DEBUG
	DumpVector(U01Vector);
	DumpVector(V01Vector);
#endif

	return true;
}


pair<vector<double>, vector<double>> Shear(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const param& wantedParam, double lowerHeight, double upperHeight)
{

	auto f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	auto height = f->Fetch(conf, ftime, level(kHeight, 0), param("Z-M2S2"), false);

	h->Configuration(conf);
	h->Time(ftime);

	vector<double> lowerHeights (height->Data()->Size(), lowerHeight);
	vector<double> upperHeights (height->Data()->Size(), upperHeight);

	size_t i;

	for (i = 0, height->ResetLocation(); height->NextLocation(); i++)
	{
		
		if (height->Value() == kFloatMissing)
		{
			lowerHeights[i] = kFloatMissing;
			upperHeights[i] = kFloatMissing;
			continue;
		}

		lowerHeights[i] = fmax(0, height->Value() * constants::kIg);
		upperHeights[i] += lowerHeights[i];

#ifdef DEBUG
		assert(lowerHeights[i] != kFloatMissing);
		assert(upperHeights[i] != kFloatMissing);
#endif

	}

	auto lowerValues = h->VerticalValue(wantedParam, lowerHeights);
	auto upperValues = h->VerticalValue(wantedParam, upperHeights);

	return make_pair(lowerValues, upperValues);
}

void StormRelativeHelicity(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime)
{
/*
	const param SRH06("HLCY-M2S2");
	const param SRH01("HLCY-1-M2S2");

	vector<param> params = { SRH06, SRH01 };
	vector<forecast_time> times = { itsTime };
	vector<level> levels = { level(kHeight, 0, "HEIGHT") };

	auto ret = make_shared<info> (*itsConfiguration->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create();

	auto f = dynamic_pointer_cast <plugin::fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	param wantedParam ("Z-M2S2");

	auto height = f->Fetch(itsConfiguration, itsTime, level(kHeight, 0), wantedParam);
*/
	// NOTES COPIED FROM SMARTTOOLS-LIBRARY

	/* // **********  SRH calculation help from Pieter Groenemeijer ******************

	Some tips here on how tyo calculate storm-relative helciity

	How to calculate storm-relative helicity

	Integrate the following from p = p_surface to p = p_top (or in case of height coordinates from h_surface to h_top):

	storm_rel_helicity -= ((u_ID-u[p])*(v[p]-v[p+1]))-((v_ID - v[p])*(u[p]-u[p+1]));

	Here, u_ID and v_ID are the forecast storm motion vectors calculated with the so-called ID-method. These can be calculated as follows:

	where

	/average wind
	u0_6 = average 0_6 kilometer u-wind component
	v0_6 = average 0_6 kilometer v-wind component
	(you should use a pressure-weighted average in case you work with height coordinates)

	/shear
	shr_0_6_u = u_6km - u_surface;
	shr_0_6_v = v_6km - v_surface;

	/ shear unit vector
	shr_0_6_u_n = shr_0_6_u / ((shr_0_6_u^2 + shr_0_6_v^2)**0.5);
	shr_0_6_v_n = shr_0_6_v / ((shr_0_6_u^2 + shr_0_6_v^2)** 0.5);

	/id-vector components
	u_ID = u0_6 + shr_0_6_v_n * 7.5;
	v_ID = v0_6 - shr_0_6_u_n * 7.5;

	(7.5 are meters per second... watch out when you work with knots instead)

	*/ // **********  SRH calculation help from Pieter Groenemeijer ******************

}

#ifdef DEBUG
void DumpVector(const vector<double>& vec)
{

	double min = 1e38, max = -1e38, sum = 0;
	size_t count = 0, missing = 0;

	BOOST_FOREACH(double val, vec)
	{
		if (val == kFloatMissing)
		{
			missing++;
			continue;
		}

		min = (val < min) ? val : min;
		max = (val > max) ? val : max;
		count++;
		sum += val;
	}

	double mean = numeric_limits<double>::quiet_NaN();

	if (count > 0)
	{
		mean = sum / static_cast<double> (count);
	}

	cout << "min " << min << " max " << max << " mean " << mean << " count " << count << " missing " << missing << endl;

	
}

#endif