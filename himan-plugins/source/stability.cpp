/**
 * @file stability.cpp
 *
 */

#include "stability.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "plugin_factory.h"
#include <algorithm>  // for std::transform
#include <boost/thread.hpp>
#include <functional>  // for std::plus

#include "fetcher.h"
#include "hitool.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

vector<double> Shear(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const param& wantedParam,
                     double lowerHeight, double upperHeight);

#ifdef DEBUG
void DumpVector(const vector<double>& vec);
#endif

// Required source and target parameters and levels

const param TParam("T-K");
const param TDParam("TD-K");
const param HParam("Z-M2S2");
const params PParam({param("P-HPA"), param("P-PA")});
const param KIParam("KINDEX-N", 80, 0, 7, 2);
const param VTIParam("VTI-N", 4754);
const param CTIParam("CTI-N", 4751);
const param TTIParam("TTI-N", 4755, 0, 7, 4);
const param SIParam("SI-N", 4750, 0, 7, 13);
const param LIParam("LI-N", 4751, 0, 7, 192);
const param BS01Param("WSH-1-KT", 4771);  // knots!
const param BS06Param("WSH-KT", 4770);    // knots!
const param SRH01Param("HLCY-1-M2S2", 4773);
const param SRH03Param("HLCY-M2S2", 4772, 0, 7, 8);

const level P850Level(himan::kPressure, 850, "PRESSURE");
const level P700Level(himan::kPressure, 700, "PRESSURE");
const level P500Level(himan::kPressure, 500, "PRESSURE");
level groundLevel(himan::kHeight, 0, "HEIGHT");

void T500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, vector<double>& result);
void TD500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, vector<double>& result);

stability::stability() : itsLICalculation(false), itsBSCalculation(false), itsSRHCalculation(false)
{
	itsLogger = logger("stability");
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

	if (itsConfiguration->Exists("li") && itsConfiguration->GetValue("li") == "true")
	{
		// Lifted index

		itsLICalculation = true;
		theParams.push_back(LIParam);

		// Showalter Index
		theParams.push_back(SIParam);
	}

	if (itsConfiguration->Exists("bs") && itsConfiguration->GetValue("bs") == "true")
	{
		itsBSCalculation = true;

		// Bulk shear 0 .. 1 km
		theParams.push_back(BS01Param);

		// Bulk shear 0 .. 6 km

		theParams.push_back(BS06Param);
	}

	if (itsConfiguration->Exists("srh") && itsConfiguration->GetValue("srh") == "true")
	{
		// Storm relative helicity 0 .. 1 km
		theParams.push_back(SRH01Param);

		// Storm relative helicity 0 .. 3 km
		theParams.push_back(SRH03Param);
	}

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
	vector<double> T500mVector, TD500mVector, P500mVector, U01Vector, V01Vector, U06Vector, V06Vector, UidVector,
	    VidVector;

	auto myThreadedLogger = logger("stabilityThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	info_t T850Info, T700Info, T500Info, TD850Info, TD700Info;

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	bool LICalculation = itsLICalculation;
	bool BSCalculation = itsBSCalculation;
	bool SRHCalculation = itsSRHCalculation;

	if (!GetSourceData(T850Info, T700Info, T500Info, TD850Info, TD700Info, myTargetInfo,
	                   itsConfiguration->UseCudaForPacking()))
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	if (LICalculation)
	{
		if (!GetLISourceData(myTargetInfo, T500mVector, TD500mVector, P500mVector))
		{
			myThreadedLogger.Warning("Source data not found for param LI");
			LICalculation = false;
		}
	}

	if (BSCalculation)
	{
		if (!GetWindShearSourceData(myTargetInfo, U01Vector, V01Vector, U06Vector, V06Vector))
		{
			myThreadedLogger.Warning("Source data not found for param BulkShear");
			BSCalculation = false;
		}
	}

	if (SRHCalculation)
	{
		if (!GetSRHSourceData(myTargetInfo, UidVector, VidVector))
		{
			myThreadedLogger.Warning("Source data not found for param SRH");
			SRHCalculation = false;
		}
	}

	string deviceType = "CPU";

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
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

		if (LICalculation)
		{
			opts->t500m = &T500mVector[0];
			opts->td500m = &TD500mVector[0];
			opts->p500m = &P500mVector[0];

			myTargetInfo->Param(param("LI-N"));
			opts->li = myTargetInfo->ToSimple();

			myTargetInfo->Param(param("SI-N"));
			opts->si = myTargetInfo->ToSimple();
		}

		if (BSCalculation)
		{
			opts->u01 = &U01Vector[0];
			opts->v01 = &V01Vector[0];
			opts->u06 = &U06Vector[0];
			opts->v06 = &V06Vector[0];

			myTargetInfo->Param(BS01Param);
			opts->bs01 = myTargetInfo->ToSimple();
			myTargetInfo->Param(BS06Param);
			opts->bs06 = myTargetInfo->ToSimple();
		}

		opts->N = opts->t500->size_x * opts->t500->size_y;

		stability_cuda::Process(*opts);
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

			double value = MissingDouble();

			value = metutil::KI_(T850, T700, T500, TD850, TD700);
			myTargetInfo->Param(KIParam);
			myTargetInfo->Value(value);

			value = metutil::CTI_(T500, TD850);
			myTargetInfo->Param(CTIParam);
			myTargetInfo->Value(value);

			value = metutil::VTI_(T850, T500);
			myTargetInfo->Param(VTIParam);
			myTargetInfo->Value(value);

			value = metutil::TTI_(T850, T500, TD850);
			myTargetInfo->Param(TTIParam);
			myTargetInfo->Value(value);

			if (LICalculation)
			{
				size_t locationIndex = myTargetInfo->LocationIndex();

				double T500m = T500mVector[locationIndex];
				double TD500m = TD500mVector[locationIndex];
				double P500m = P500mVector[locationIndex];

				assert(!IsMissing(T500m));
				assert(!IsMissing(TD500m));
				assert(!IsMissing(P500m));

				value = metutil::LI_(T500, T500m, TD500m, P500m);

				myTargetInfo->Param(LIParam);
				myTargetInfo->Value(value);

				value = metutil::SI_(T850, T500, TD850);
				myTargetInfo->Param(SIParam);
				myTargetInfo->Value(value);
			}

			if (BSCalculation)
			{
				size_t locationIndex = myTargetInfo->LocationIndex();

				double U01 = U01Vector[locationIndex];
				double V01 = V01Vector[locationIndex];
				double U06 = U06Vector[locationIndex];
				double V06 = V06Vector[locationIndex];

				assert(!IsMissing(U01));
				assert(!IsMissing(V01));
				assert(!IsMissing(U06));
				assert(!IsMissing(V06));

				value = metutil::BulkShear_(U01, V01);

				myTargetInfo->Param(BS01Param);
				myTargetInfo->Value(value);

				value = metutil::BulkShear_(U06, V06);

				myTargetInfo->Param(BS06Param);
				myTargetInfo->Value(value);
			}
#if 0
			if (SRHCalculation)
			{
				size_t locationIndex = myTargetInfo->LocationIndex();

				double Uid = UidVector[locationIndex];
				double Vid = VidVector[locationIndex];

				assert(!IsMissing(Uid));
				assert(!IsMissing(Vid));

				if (!IsMissing(Uid) && !IsMissing(Vid))
				{
				}
			}
#endif
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void T500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, vector<double>& result)
{
	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	h->Time(ftime);

	result = h->VerticalAverage(param("T-K"), 0, 500);

#ifdef DEBUG
	for (size_t i = 0; i < result.size(); i++)
	{
		assert(!IsMissing(result[i]));
	}
#endif
}

void TD500mSearch(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, vector<double>& result)
{
	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	h->Time(ftime);

	result = h->VerticalAverage(param("TD-K"), 0, 500);
}
#if 0
inline
double stability::StormRelativeHelicity(double UID, double VID, double U_lower, double U_higher, double V_lower, double V_higher)
{
	return ((UID - U_lower) * (V_lower - V_higher)) - ((VID - V_lower) * (U_lower - U_higher));
}
#endif

bool stability::GetSourceData(shared_ptr<info>& T850Info, shared_ptr<info>& T700Info, shared_ptr<info>& T500Info,
                              shared_ptr<info>& TD850Info, shared_ptr<info>& TD700Info,
                              const shared_ptr<info>& myTargetInfo, bool useCudaInThisThread)
{
	bool ret = true;

	if (!T850Info)
	{
		T850Info = Fetch(myTargetInfo->Time(), P850Level, TParam, myTargetInfo->ForecastType(), useCudaInThisThread);
	}

	if (!T700Info)
	{
		T700Info = Fetch(myTargetInfo->Time(), P700Level, TParam, myTargetInfo->ForecastType(), useCudaInThisThread);
	}

	if (!T500Info)
	{
		T500Info = Fetch(myTargetInfo->Time(), P500Level, TParam, myTargetInfo->ForecastType(), useCudaInThisThread);
	}

	if (!TD850Info)
	{
		TD850Info = Fetch(myTargetInfo->Time(), P850Level, TDParam, myTargetInfo->ForecastType(), useCudaInThisThread);
	}

	if (!TD700Info)
	{
		TD700Info = Fetch(myTargetInfo->Time(), P700Level, TDParam, myTargetInfo->ForecastType(), useCudaInThisThread);
	}

	if (!T850Info || !T700Info || !T500Info || !TD850Info || !TD700Info)
	{
		ret = false;
	}

	return ret;
}

bool stability::GetLISourceData(const shared_ptr<info>& myTargetInfo, vector<double>& T500mVector,
                                vector<double>& TD500mVector, vector<double>& P500mVector)
{
	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

// Fetch Z uncompressed since it is not transferred to cuda
#if 0
	auto HInfo = Fetch(myTargetInfo->Time(), groundLevel, HParam, false);

	if (!HInfo)
	{
		return false;
	}


	vector<double> H0mVector = HInfo->Grid()->Data().Values();
	vector<double> H500mVector(HInfo->SizeLocations());

	for (size_t i = 0; i < H500mVector.size(); i++)
	{
		// H0mVector contains the height of ground (compared to MSL). Height can be negative
		// (maybe even in real life (Netherlands?)), but in our case we use 0 as smallest height.
		// TODO: check how it is in smarttools

		if (IsMissing(H0mVector[i]))
		{
			continue;
		}
		
		H0mVector[i] *= constants::kIg;
		H0mVector[i] = fmax(0, H0mVector[i]);

		H500mVector[i] = H0mVector[i] + 500.;
	}

#endif

	// Fetch average values of T, TD and P over vertical height range 0 ... 500m OVER GROUND

	boost::thread t1(&T500mSearch, itsConfiguration, myTargetInfo->Time(), boost::ref(T500mVector));
	boost::thread t2(&TD500mSearch, itsConfiguration, myTargetInfo->Time(), boost::ref(TD500mVector));

	P500mVector = h->VerticalAverage(PParam, 0., 500.);

	assert(!IsMissing(P500mVector[0]));

	if (P500mVector[0] < 1500)
	{
		transform(P500mVector.begin(), P500mVector.end(), P500mVector.begin(),
		          bind1st(multiplies<double>(), 100));  // hPa to Pa
	}

	t1.join();
	t2.join();

	return true;
}

bool stability::GetWindShearSourceData(const shared_ptr<info>& myTargetInfo, vector<double>& U01Vector,
                                       vector<double>& V01Vector, vector<double>& U06Vector, vector<double>& V06Vector)
{
	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	// BS 0-6
	U06Vector = Shear(itsConfiguration, myTargetInfo->Time(), param("U-MS"), 0, 6000);
	V06Vector = Shear(itsConfiguration, myTargetInfo->Time(), param("V-MS"), 0, 6000);

#ifdef DEBUG
	DumpVector(U06Vector);
	DumpVector(V06Vector);
#endif

	// BS 0-1

	U01Vector = Shear(itsConfiguration, myTargetInfo->Time(), param("U-MS"), 0, 1000);
	V01Vector = Shear(itsConfiguration, myTargetInfo->Time(), param("V-MS"), 0, 1000);

#ifdef DEBUG
	DumpVector(U01Vector);
	DumpVector(V01Vector);
#endif

	return true;
}

vector<double> Shear(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, const param& wantedParam,
                     double lowerHeight, double upperHeight)
{
	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));

	auto lowerValues = h->VerticalValue(wantedParam, lowerHeight);
	auto upperValues = h->VerticalValue(wantedParam, upperHeight);

	vector<double> ret(lowerValues.size(), MissingDouble());

	for (size_t i = 0; i < lowerValues.size(); i++)
	{
		ret[i] = upperValues[i] - lowerValues[i];
	}

	return ret;
}

bool stability::GetSRHSourceData(const shared_ptr<info>& myTargetInfo, vector<double>& Uid, vector<double>& Vid)
{
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

	*/  // **********  SRH calculation help from Pieter Groenemeijer ******************

	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	// average wind
	auto Uavg = h->VerticalAverage(param("U-MS"), 0, 6000);
	auto Vavg = h->VerticalAverage(param("V-MS"), 0, 6000);

	// shear
	auto Ushear = Shear(itsConfiguration, myTargetInfo->Time(), param("U-MS"), 0, 6000);
	auto Vshear = Shear(itsConfiguration, myTargetInfo->Time(), param("V-MS"), 0, 6000);

	// shear unit vectors
	Uid.resize(Ushear.size(), MissingDouble());
	Vid.resize(Vshear.size(), MissingDouble());

	assert(Uid.size() == Vid.size());
	assert(Uid.size() == Uavg.size());

	for (size_t i = 0; i < Ushear.size(); i++)
	{
		double u = Ushear[i];
		double v = Vshear[i];

		double Uunit = u / sqrt(u * u + v * v);
		double Vunit = v / sqrt(u * u + v * v);

		Uid[i] = Uavg[i] - Vunit * 7.5;
		Vid[i] = Vavg[i] - Uunit * 7.5;
	}

	return true;
}

#ifdef DEBUG
void DumpVector(const vector<double>& vec)
{
	double min = 1e38, max = -1e38, sum = 0;
	size_t count = 0, missing = 0;

	for (double val : vec)
	{
		if (IsMissing(val))
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
		mean = sum / static_cast<double>(count);
	}

	cout << "min " << min << " max " << max << " mean " << mean << " count " << count << " missing " << missing << endl;
}

#endif
