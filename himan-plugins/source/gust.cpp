/**
 * @file gust.cpp
 *
 * Computes wind gusts
 */

#include "gust.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include <thread>
#include "util.h"

#include "hitool.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

struct deltaT
{
	vector<double> deltaT_100;
	vector<double> deltaT_200;
	vector<double> deltaT_300;
	vector<double> deltaT_400;
	vector<double> deltaT_500;
	vector<double> deltaT_600;
	vector<double> deltaT_700;

	deltaT() = default;
};

struct deltaTot
{
	vector<double> deltaTot_100;
	vector<double> deltaTot_200;
	vector<double> deltaTot_300;
	vector<double> deltaTot_400;
	vector<double> deltaTot_500;
	vector<double> deltaTot_600;
	vector<double> deltaTot_700;

	deltaTot() = default;
};

struct intT
{
	vector<double> intT_100;
	vector<double> intT_200;
	vector<double> intT_300;
	vector<double> intT_400;
	vector<double> intT_500;
	vector<double> intT_600;
	vector<double> intT_700;

	intT() = default;
};

struct intTot
{
	vector<double> intTot_100;
	vector<double> intTot_200;
	vector<double> intTot_300;
	vector<double> intTot_400;
	vector<double> intTot_500;
	vector<double> intTot_600;
	vector<double> intTot_700;

	intTot() = default;
};

void DeltaT(const shared_ptr<const plugin_configuration>& conf, shared_ptr<info<double>> T_lowestLevel,
            const forecast_time& ftime, const forecast_type& ftype, size_t gridSize, deltaT& dT);
void DeltaTot(deltaTot& dTot, const shared_ptr<info<double>>& T_lowestLevel, size_t gridSize);
void IntT(intT& iT, const deltaT& dT, size_t gridSize);
void IntTot(intTot& iTot, const deltaTot& dTot, size_t gridSize);
void LowAndMiddleClouds(vector<double>& lowAndMiddleClouds, const shared_ptr<info<double>>& lowClouds,
                        const shared_ptr<info<double>>& middleClouds, const shared_ptr<info<double>>& highClouds,
                        const shared_ptr<info<double>>& totalClouds);

gust::gust()
{
	itsLogger = logger("gust");
}
void gust::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param theRequestedParam("FFG2-MS", aggregation(kMaximum, conf->ForecastStep()), processing_type());

	theRequestedParam.Unit(kMs);

	SetParams({theRequestedParam});

	Start();
}

/*
 * Calculate()

 * This function does the actual calculation.
 */

void gust::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("gust_pluginThread #" + to_string(threadIndex));

	/*
	 * Required source parameters
	 *
	 */
	const size_t gridSize = myTargetInfo->Grid()->Size();

	const param BLHParam("MIXHGT-M");  // boundary layer height
	const param WSParam("FF-MS");      // wind speed
	const param GustParam("FFG-MS", aggregation(kMaximum, itsConfiguration->ForecastStep()),
	                      processing_type());                             // wind gust
	const param TParam("T-K");                                            // temperature
	const param TopoParam("Z-M2S2");                                      // geopotential height
	const params LowCloudParam = {param("NL-0TO1"), param("NL-PRCNT")};   // low cloud cover
	const params MidCloudParam = {param("NM-0TO1"), param("NM-PRCNT")};   // middle cloud cover
	const params HighCloudParam = {param("NH-0TO1"), param("NH-PRCNT")};  // high cloud cover
	const params TotalCloudParam = {param("N-PRCNT"), param("N-0TO1")};   // total cloud cover

	level H0, H10;

	level lowestHybridLevel = util::CreateHybridLevel(itsConfiguration->SourceProducer(0), "last");

	if (myTargetInfo->Producer().Id() == 240 || myTargetInfo->Producer().Id() == 243)
	{
		H0 = level(kGround, 0);
		H10 = H0;
	}
	else
	{
		H0 = level(kHeight, 0);
		H10 = level(kHeight, 10);
	}

	shared_ptr<info<double>> GustInfo, T_LowestLevelInfo, BLHInfo, TopoInfo, LCloudInfo, MCloudInfo, HCloudInfo,
	    TCloudInfo;

	// Current time and level

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	GustInfo = Fetch(forecastTime, H10, GustParam, forecastType, false);
	T_LowestLevelInfo = Fetch(forecastTime, lowestHybridLevel, TParam, forecastType, false);
	BLHInfo = Fetch(forecastTime, H0, BLHParam, forecastType, false);
	TopoInfo = Fetch(forecastTime, H0, TopoParam, forecastType, false);
	LCloudInfo = Fetch(forecastTime, H0, LowCloudParam, forecastType, false);
	MCloudInfo = Fetch(forecastTime, H0, MidCloudParam, forecastType, false);
	HCloudInfo = Fetch(forecastTime, H0, HighCloudParam, forecastType, false);
	TCloudInfo = Fetch(forecastTime, H0, TotalCloudParam, forecastType, false);

	if (!BLHInfo || !TopoInfo || !GustInfo || !T_LowestLevelInfo || !LCloudInfo || !MCloudInfo || !HCloudInfo ||
	    !TCloudInfo)
	{
		itsLogger.Error("Unable to find all source data");
		return;
	}

	deltaT dT;
	vector<double> lowAndMiddleClouds(gridSize, himan::MissingDouble());

	thread t(&DeltaT, itsConfiguration, T_LowestLevelInfo, forecastTime, forecastType, gridSize, std::ref(dT));
	thread t2(&LowAndMiddleClouds, std::ref(lowAndMiddleClouds), LCloudInfo, MCloudInfo, HCloudInfo, TCloudInfo);

	// calc boundary layer height
	vector<double> z_boundaryl(gridSize, himan::MissingDouble());
	vector<double> z_one_third_boundaryl(gridSize, himan::MissingDouble());
	vector<double> z_two_third_boundaryl(gridSize, himan::MissingDouble());
	vector<double> z_zero(gridSize, 0);
	for (size_t i = 0; i < gridSize; ++i)
	{
		if (IsMissing(BLHInfo->Data()[i]))
		{
			continue;
		}

		z_boundaryl[i] = BLHInfo->Data()[i];

		if (BLHInfo->Data()[i] >= 200)
		{
			z_boundaryl[i] = 0.5 * BLHInfo->Data()[i] + 100;

			if (BLHInfo->Data()[i] > 1200)
			{
				z_boundaryl[i] = 700;
			}
		}

		z_one_third_boundaryl[i] = z_boundaryl[i] / 3;
		z_two_third_boundaryl[i] = 2 * z_one_third_boundaryl[i];
	}

	// maybe need adjusting
	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(forecastTime);
	h->ForecastType(forecastType);

	vector<double> maxEstimate;
	try
	{
		maxEstimate = h->VerticalAverage<double>(WSParam, z_two_third_boundaryl, z_boundaryl);
	}

	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + to_string(e));
		}
		else
		{
			myThreadedLogger.Error("hitool was unable to find data");
			t.join();
			t2.join();
			return;
		}
	}

	vector<double> BLtop_ws;
	try
	{
		BLtop_ws = h->VerticalAverage<double>(WSParam, z_one_third_boundaryl, z_boundaryl);
	}

	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + to_string(e));
		}
		else
		{
			myThreadedLogger.Error("hitool was unable to find data");
			t.join();
			t2.join();
			return;
		}
	}

	vector<double> meanWind;
	try
	{
		meanWind = h->VerticalAverage<double>(WSParam, z_zero, z_boundaryl);
	}

	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + to_string(e));
		}
		else
		{
			myThreadedLogger.Error("hitool was unable to find data");
			t.join();
			t2.join();
			return;
		}
	}

	vector<double> t_diff;
	try
	{
		t_diff = h->VerticalMaximum<double>(TParam, 0, 200);
		for (size_t i = 0; i < gridSize; ++i)
		{
			t_diff[i] = t_diff[i] - T_LowestLevelInfo->Data()[i];
		}
	}

	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + to_string(e));
		}
		else
		{
			myThreadedLogger.Error("hitool was unable to find data");
			t.join();
			t2.join();
			return;
		}
	}

	vector<double> BLbottom_ws;
	try
	{
		BLbottom_ws = h->VerticalAverage<double>(WSParam, 0, 200);
	}

	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + to_string(e));
		}
		else
		{
			myThreadedLogger.Error("hitool was unable to find data");
			t.join();
			t2.join();
			return;
		}
	}

	t.join();
	t2.join();

	deltaTot dTot;
	intT iT;
	intTot iTot;

	DeltaTot(dTot, T_LowestLevelInfo, gridSize);
	IntT(iT, dT, gridSize);
	IntTot(iTot, dTot, gridSize);

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, GustInfo, T_LowestLevelInfo, TopoInfo, BLHInfo, LCloudInfo, MCloudInfo)
	{
		size_t i = myTargetInfo->LocationIndex();

		double topo = TopoInfo->Value();
		double esto = himan::MissingDouble();
		double esto_tot = himan::MissingDouble();
		double result = himan::MissingDouble();
		double turb_lisa = 0;
		double turb_kerroin = 0;
		double pilvikerroin = 0;
		double cloudCover = (LCloudInfo->Value() + MCloudInfo->Value()) * 100;
		topo *= himan::constants::kIg;

		double elevationAngle = metutil::ElevationAngle_(myTargetInfo->LatLon(), forecastTime.ValidDateTime());

		/* Calculations go here */

		if (z_boundaryl[i] >= 200)
		{
			esto = iT.intT_100[i] + iT.intT_200[i];
			esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i];
		}

		if (z_boundaryl[i] >= 300)
		{
			esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i];
			esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i];
		}

		if (z_boundaryl[i] >= 400)
		{
			esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i];
			esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i];
		}

		if (z_boundaryl[i] >= 500)
		{
			esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i] + iT.intT_500[i];
			esto_tot =
			    iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i] + iTot.intTot_500[i];
		}

		if (z_boundaryl[i] >= 600)
		{
			esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i] + iT.intT_500[i] + iT.intT_600[i];
			esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i] +
			           iTot.intTot_500[i] + iTot.intTot_600[i];
		}

		if (z_boundaryl[i] == 700)
		{
			esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i] + iT.intT_500[i] + iT.intT_600[i] +
			       iT.intT_700[i];
			esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i] +
			           iTot.intTot_500[i] + iTot.intTot_600[i] + iTot.intTot_700[i];
		}

		if (z_boundaryl[i] >= 200 && esto >= 0 && esto <= esto_tot)
		{
			result = ((BLbottom_ws[i] - BLtop_ws[i]) / esto_tot) * esto + BLtop_ws[i];
		}

		if (z_boundaryl[i] >= 200 && esto <= 0 && esto >= -200)
		{
			result = ((BLtop_ws[i] - maxEstimate[i]) / 200) * esto + BLtop_ws[i];
		}

		if (z_boundaryl[i] >= 200 && esto < -200)
		{
			result = maxEstimate[i];
		}

		if (z_boundaryl[i] >= 200 && esto > esto_tot)
		{
			result = BLbottom_ws[i];
		}

		if (z_boundaryl[i] < 200)
		{
			result = meanWind[i];
		}

		if (IsMissing(result) || result < 1)
		{
			result = 1;
		}

		if (topo > 15 && topo < 400 && t_diff[i] > 0 && t_diff[i] <= 4 && BLbottom_ws[i] < 7 && elevationAngle < 15)
		{
			result = ((1 - result) / 4) * t_diff[i] + result;
		}

		if (topo > 15 && topo < 400 && t_diff[i] > 4 && BLbottom_ws[i] < 7)
		{
			result = 1;
		}

		if (cloudCover >= 30 && cloudCover <= 70)
		{
			pilvikerroin = -0.025 * cloudCover + 1.75;
		}

		if (cloudCover < 30)
		{
			pilvikerroin = 1;
		}

		if (elevationAngle >= 20 && topo > 10 && z_boundaryl[i] >= 200 && esto < 0)
		{
			turb_lisa = 0.133333333 * elevationAngle - 2.666666667;

			if (elevationAngle > 50)
			{
				turb_lisa = 4;
			}

			if (result > 4 && result <= 14)
			{
				turb_kerroin = -0.1 * result + 1.4;
			}

			if (result <= 4)
			{
				turb_kerroin = 1;
			}
		}

		result = result + turb_lisa * turb_kerroin * pilvikerroin;

		if (topo > 400 || IsMissing(T_LowestLevelInfo->Value()) || IsMissing(BLHInfo->Value()))
		{
			result = GustInfo->Value() * 0.95;
		}

		myTargetInfo->Value(result);
	}

	himan::matrix<double> filter_kernel(3, 3, 1, himan::MissingDouble());
	filter_kernel.Fill(1.0 / 9.0);
	himan::matrix<double> gust_filtered =
	    numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	auto b = myTargetInfo->Base();
	b->data = move(gust_filtered);

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void DeltaT(const shared_ptr<const plugin_configuration>& conf, shared_ptr<info<double>> T_lowestLevel,
            const forecast_time& ftime, const forecast_type& ftype, size_t gridSize, deltaT& dT)
{
	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(ftime);
	h->ForecastType(ftype);

	const param TParam("T-K");

	try
	{
		// Potential temperature differences
		dT.deltaT_100 = h->VerticalValue<double>(TParam, 100);
		dT.deltaT_200 = h->VerticalValue<double>(TParam, 200);
		dT.deltaT_300 = h->VerticalValue<double>(TParam, 300);
		dT.deltaT_400 = h->VerticalValue<double>(TParam, 400);
		dT.deltaT_500 = h->VerticalValue<double>(TParam, 500);
		dT.deltaT_600 = h->VerticalValue<double>(TParam, 600);
		dT.deltaT_700 = h->VerticalValue<double>(TParam, 700);

		for (size_t i = 0; i < gridSize; ++i)
		{
			if (IsMissing(T_lowestLevel->Data()[i]))
			{
				dT.deltaT_100[i] = himan::MissingDouble();
				dT.deltaT_200[i] = himan::MissingDouble();
				dT.deltaT_300[i] = himan::MissingDouble();
				dT.deltaT_400[i] = himan::MissingDouble();
				dT.deltaT_500[i] = himan::MissingDouble();
				dT.deltaT_600[i] = himan::MissingDouble();
				dT.deltaT_700[i] = himan::MissingDouble();
				continue;
			}
			if (!IsMissing(dT.deltaT_100[i]))
				dT.deltaT_100[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (100.0 / 1000.0));
			if (!IsMissing(dT.deltaT_200[i]))
				dT.deltaT_200[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (200.0 / 1000.0));
			if (!IsMissing(dT.deltaT_300[i]))
				dT.deltaT_300[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (300.0 / 1000.0));
			if (!IsMissing(dT.deltaT_400[i]))
				dT.deltaT_400[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (400.0 / 1000.0));
			if (!IsMissing(dT.deltaT_500[i]))
				dT.deltaT_500[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (500.0 / 1000.0));
			if (!IsMissing(dT.deltaT_600[i]))
				dT.deltaT_600[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (600.0 / 1000.0));
			if (!IsMissing(dT.deltaT_700[i]))
				dT.deltaT_700[i] -= T_lowestLevel->Data()[i] + 9.8 * (0.010 - (700.0 / 1000.0));
		}
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("DeltaT() caught exception " + to_string(e));
		}
	}
}

void DeltaTot(deltaTot& dTot, const shared_ptr<info<double>>& T_lowestLevel, size_t gridSize)
{
	dTot.deltaTot_100 = vector<double>(gridSize, 0);
	dTot.deltaTot_200 = vector<double>(gridSize, 0);
	dTot.deltaTot_300 = vector<double>(gridSize, 0);
	dTot.deltaTot_400 = vector<double>(gridSize, 0);
	dTot.deltaTot_500 = vector<double>(gridSize, 0);
	dTot.deltaTot_600 = vector<double>(gridSize, 0);
	dTot.deltaTot_700 = vector<double>(gridSize, 0);

	for (size_t i = 0; i < gridSize; ++i)
	{
		dTot.deltaTot_100[i] =
		    (T_lowestLevel->Data()[i] + 6 * (0.010 - (100.0 / 1000.0))) -
		    (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (100.0 / 1000.0)));  // => -3.8*(0.010-(100/1000)) ???
		dTot.deltaTot_200[i] =
		    (T_lowestLevel->Data()[i] + 6 * (0.010 - (200.0 / 1000.0))) -
		    (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (200.0 / 1000.0)));  // => -3.8*(0.010-(200/1000))
		dTot.deltaTot_300[i] = (T_lowestLevel->Data()[i] + 6 * (0.010 - (300.0 / 1000.0))) -
		                       (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (300.0 / 1000.0)));  // ... etc.
		dTot.deltaTot_400[i] = (T_lowestLevel->Data()[i] + 6 * (0.010 - (400.0 / 1000.0))) -
		                       (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (400.0 / 1000.0)));
		dTot.deltaTot_500[i] = (T_lowestLevel->Data()[i] + 6 * (0.010 - (500.0 / 1000.0))) -
		                       (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (500.0 / 1000.0)));
		dTot.deltaTot_600[i] = (T_lowestLevel->Data()[i] + 6 * (0.010 - (600.0 / 1000.0))) -
		                       (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (600.0 / 1000.0)));
		dTot.deltaTot_700[i] = (T_lowestLevel->Data()[i] + 6 * (0.010 - (700.0 / 1000.0))) -
		                       (T_lowestLevel->Data()[i] + 9.8 * (0.010 - (700.0 / 1000.0)));
	}
}

void IntT(intT& iT, const deltaT& dT, size_t gridSize)
{
	iT.intT_100 = vector<double>(gridSize, himan::MissingDouble());
	iT.intT_200 = vector<double>(gridSize, himan::MissingDouble());
	iT.intT_300 = vector<double>(gridSize, himan::MissingDouble());
	iT.intT_400 = vector<double>(gridSize, himan::MissingDouble());
	iT.intT_500 = vector<double>(gridSize, himan::MissingDouble());
	iT.intT_600 = vector<double>(gridSize, himan::MissingDouble());
	iT.intT_700 = vector<double>(gridSize, himan::MissingDouble());

	for (size_t i = 0; i < gridSize; ++i)
	{
		if (!IsMissing(dT.deltaT_100[i]))
			iT.intT_100[i] = 0.5 * dT.deltaT_100[i] * 100;  // why 0.5 * here? Would make sense in case of
			                                                // (dT.deltaT_100[i] + dT.deltaT_0[i]) if this is
			                                                // integration using trapezoidal rule
		if (!IsMissing(dT.deltaT_100[i]) && !IsMissing(dT.deltaT_200[i]))
			iT.intT_200[i] = 0.5 * (dT.deltaT_200[i] + dT.deltaT_100[i]) * 100;
		if (!IsMissing(dT.deltaT_200[i]) && !IsMissing(dT.deltaT_300[i]))
			iT.intT_300[i] = 0.5 * (dT.deltaT_300[i] + dT.deltaT_200[i]) * 100;
		if (!IsMissing(dT.deltaT_300[i]) && !IsMissing(dT.deltaT_400[i]))
			iT.intT_400[i] = 0.5 * (dT.deltaT_400[i] + dT.deltaT_300[i]) * 100;
		if (!IsMissing(dT.deltaT_400[i]) && !IsMissing(dT.deltaT_500[i]))
			iT.intT_500[i] = 0.5 * (dT.deltaT_500[i] + dT.deltaT_400[i]) * 100;
		if (!IsMissing(dT.deltaT_500[i]) && !IsMissing(dT.deltaT_600[i]))
			iT.intT_600[i] = 0.5 * (dT.deltaT_600[i] + dT.deltaT_500[i]) * 100;
		if (!IsMissing(dT.deltaT_600[i]) && !IsMissing(dT.deltaT_700[i]))
			iT.intT_700[i] = 0.5 * (dT.deltaT_700[i] + dT.deltaT_600[i]) * 100;
	}
}

void IntTot(intTot& iTot, const deltaTot& dTot, size_t gridSize)
{
	iTot.intTot_100 = vector<double>(gridSize, 0);
	iTot.intTot_200 = vector<double>(gridSize, 0);
	iTot.intTot_300 = vector<double>(gridSize, 0);
	iTot.intTot_400 = vector<double>(gridSize, 0);
	iTot.intTot_500 = vector<double>(gridSize, 0);
	iTot.intTot_600 = vector<double>(gridSize, 0);
	iTot.intTot_700 = vector<double>(gridSize, 0);

	for (size_t i = 0; i < gridSize; ++i)
	{
		iTot.intTot_100[i] = 0.5 * dTot.deltaTot_100[i] * 100;  // 0.5 * ???
		iTot.intTot_200[i] = 0.5 * (dTot.deltaTot_200[i] + dTot.deltaTot_100[i]) * 100;
		iTot.intTot_300[i] = 0.5 * (dTot.deltaTot_300[i] + dTot.deltaTot_200[i]) * 100;
		iTot.intTot_400[i] = 0.5 * (dTot.deltaTot_400[i] + dTot.deltaTot_300[i]) * 100;
		iTot.intTot_500[i] = 0.5 * (dTot.deltaTot_500[i] + dTot.deltaTot_400[i]) * 100;
		iTot.intTot_600[i] = 0.5 * (dTot.deltaTot_600[i] + dTot.deltaTot_500[i]) * 100;
		iTot.intTot_700[i] = 0.5 * (dTot.deltaTot_700[i] + dTot.deltaTot_600[i]) * 100;
	}
}

void LowAndMiddleClouds(vector<double>& lowAndMiddleClouds, const shared_ptr<info<double>>& lowClouds,
                        const shared_ptr<info<double>>& middleClouds, const shared_ptr<info<double>>& highClouds,
                        const shared_ptr<info<double>>& totalClouds)
{
	for (size_t i = 0; i < lowAndMiddleClouds.size(); ++i)
	{
		if (IsMissing(highClouds->Data()[i]) || highClouds->Data()[i] == 0.0)
		{
			if (IsMissing(totalClouds->Data()[i]))
				lowAndMiddleClouds[i] = himan::MissingDouble();
			else
				lowAndMiddleClouds[i] = totalClouds->Data()[i] * 100;
		}
		else
		{
			if (!IsMissing(lowClouds->Data()[i]))
			{
				if (!IsMissing(middleClouds->Data()[i]))
					lowAndMiddleClouds[i] = max(lowClouds->Data()[i], middleClouds->Data()[i]) * 100;
				else
					lowAndMiddleClouds[i] = lowClouds->Data()[i] * 100;
			}
			else
			{
				if (IsMissing(middleClouds->Data()[i]))
					lowAndMiddleClouds[i] = himan::MissingDouble();
				else
					lowAndMiddleClouds[i] = middleClouds->Data()[i] * 100;
			}
		}
	}
}
