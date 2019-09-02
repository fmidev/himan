/**
 * @file pop.cpp
 *
 */
#include "pop.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"

#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

pop::pop()
    : itsECEPSGeom("ECEUR0200"),
      itsECGeom("ECGLO0100"),
      itsPEPSGeom("PEPSSCAND"),
      itsHirlamGeom("RCR068"),
      itsMEPSGeom("MEPSSCAN2500G2"),
      itsGFSGeom("GFS0250")
{
	itsLogger = logger("pop");
}

void pop::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param theRequestedParam("POP-PRCNT", 259);

	theRequestedParam.Unit(kPrcnt);

	SetParams({theRequestedParam});

	if (itsConfiguration->Exists("eceps_geom"))
	{
		itsECEPSGeom = itsConfiguration->GetValue("eceps_geom");
	}

	if (itsConfiguration->Exists("ec_geom"))
	{
		itsECGeom = itsConfiguration->GetValue("ec_geom");
	}

	if (itsConfiguration->Exists("peps_geom"))
	{
		itsPEPSGeom = itsConfiguration->GetValue("peps_geom");
	}

	if (itsConfiguration->Exists("hirlam_geom"))
	{
		itsHirlamGeom = itsConfiguration->GetValue("hirlam_geom");
	}

	if (itsConfiguration->Exists("meps_geom"))
	{
		itsMEPSGeom = itsConfiguration->GetValue("meps_geom");
	}
	else if (itsConfiguration->Exists("harmonie_geom"))
	{
		itsMEPSGeom = itsConfiguration->GetValue("harmonie_geom");
	}

	if (itsConfiguration->Exists("gfs_geom"))
	{
		itsGFSGeom = itsConfiguration->GetValue("gfs_geom");
	}

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void pop::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	// Macro comments in Finnish, my addition in English

	// The mathematical definition of Probability of Precipitation is defined as: PoP = C * A
	// C = the confidence that precipitation will occur somewhere in the forecast area
	// A = the percent of the area that will receive measurable precipitation, if it occurs at all

	// Weights for different parameters

	const double K1 = 0.25;  // EC:n Fraktiili 50
	const double K2 = 0.25;  // EC:n Fraktiili 75
	const double K3 = 1;     // EC:n edellinen malliajo
	const double K4 = 1;     // PEPS rr>0.2 mm
	const double K5 = 2;     // EC:n viimeisin malliajo
	const double K6 = 1;     // Hirlamin viimeisin malliajo
	const double K7 = 1;     // GFS:n viimeisin malliajo
	const double K8 = 1;     // MEPS:n viimeisin malliajo

	// Current time and level as given to this thread

	const forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	const level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("popThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	vector<double> PEPS, Hirlam, MEPS, GFS, EC, ECprev, ECprob1, ECprob2, ECfract50, ECfract75;

	/*
	 * Required source parameters
	 */

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
	auto f = GET_PLUGIN(fetcher);

	try
	{
		// ECMWF deterministic
		cnf->SourceGeomNames({itsECGeom});

		// Current forecast
		auto ECInfo = f->Fetch(cnf, forecastTime, level(kHeight, 0), param("RRR-KGM2"), forecastType, false);
		EC = VEC(ECInfo);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			myThreadedLogger.Error("ECMWF deterministic precipitation data not found");
		}

		return;
	}

	/*
	 * Optional source parameters
	 */

	auto prevTime = forecastTime;
	prevTime.OriginDateTime().Adjust(kHourResolution, -12);

	try
	{
		// ECMWF deterministic
		cnf->SourceGeomNames({itsECGeom});

		// Previous forecast
		auto ECprevInfo = f->Fetch(cnf, prevTime, level(kHeight, 0), param("RRR-KGM2"), forecastType, false);
		ECprev = VEC(ECprevInfo);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			ECprev.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	// ECMWF probabilities from EPS

	try
	{
		cnf->SourceProducers({producer(242, 0, 0, "ECM_PROB")});
		cnf->SourceGeomNames({itsECEPSGeom});

		// PROB-RR-1 = "RR>= 1mm 6h"
		auto ECprob1Info = f->Fetch(cnf, prevTime, level(kGround, 0), param("PROB-RR-1"), forecastType, false);
		ECprob1 = VEC(ECprob1Info);

		// PROB-RR3-1 = "RR>= 0.3mm 3h"
		auto ECProb2Info = f->Fetch(cnf, prevTime, level(kGround, 0), param("PROB-RR3-1"), forecastType, false);
		ECprob2 = VEC(ECProb2Info);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			ECprob1.resize(myTargetInfo->Data().Size(), MissingDouble());
			ECprob2.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	// ECMWF fractiles from EPS

	try
	{
		cnf->SourceProducers({producer(242, 0, 0, "ECM_PROB")});

		// 50th fractile (median)
		auto ECfract50Info = f->Fetch(cnf, prevTime, level(kHeight, 0), param("F50-RR-6-MM"), forecastType, false);
		ECfract50 = VEC(ECfract50Info);

		// 75th fractile
		auto ECfract75Info = f->Fetch(cnf, prevTime, level(kHeight, 0), param("F75-RR-6-MM"), forecastType, false);
		ECfract75 = VEC(ECfract75Info);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			ECfract50.resize(myTargetInfo->Data().Size(), MissingDouble());
			ECfract75.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	// PEPS

	try
	{
		// Peps uses grib2, neons support for this is patchy and we have to give grib centre & ident
		// so that correct name is found

		cnf->SourceGeomNames({itsPEPSGeom});
		cnf->SourceProducers({producer(121, 86, 121, "PEPSSCAN")});

		// PROB-RR-1 = "RR>= 0.2mm 1h"
		// Yes, the level really is height/2!
		auto PEPSInfo = f->Fetch(cnf, forecastTime, level(kHeight, 2), param("PROB-RR-1"), forecastType, false);
		PEPS = VEC(PEPSInfo);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			PEPS.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	// Hirlam

	try
	{
		cnf->SourceGeomNames({itsHirlamGeom});
		cnf->SourceProducers({producer(230, 0, 0, "HL2MTA")});

		auto HirlamInfo = f->Fetch(cnf, forecastTime, forecastLevel, param("RRR-KGM2"), forecastType, false);

		Hirlam = VEC(HirlamInfo);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			Hirlam.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	// MEPS control

	try
	{
		cnf->SourceGeomNames({itsMEPSGeom});
		cnf->SourceProducers({producer(260, 0, 0, "MEPSMTA")});

		auto MEPSInfo =
		    f->Fetch(cnf, forecastTime, forecastLevel, param("RRR-KGM2"), forecast_type(kEpsControl, 0), false);

		MEPS = VEC(MEPSInfo);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			MEPS.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	// GFS

	try
	{
		cnf->SourceGeomNames({itsGFSGeom});
		cnf->SourceProducers({producer(250, 0, 0, "GFSMTA")});
		auto GFSInfo = f->Fetch(cnf, forecastTime, forecastLevel, param("RRR-KGM2"), forecastType, false);

		GFS = VEC(GFSInfo);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			GFS.resize(myTargetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
	}

	const string deviceType = "CPU";

	matrix<double> area(myTargetInfo->Data().SizeX(), myTargetInfo->Data().SizeY(), 1, MissingDouble(), 0);  // "A"
	matrix<double> confidence(area.SizeX(), area.SizeY(), 1, MissingDouble(), MissingDouble());              // "C"

	// 1. Calculate initial area and confidence of precipitation

	for (auto&& tup :
	     zip_range(confidence.Values(), area.Values(), ECfract50, ECfract75, EC, ECprev, PEPS, Hirlam, MEPS, GFS))
	{
		double& out_confidence = tup.get<0>();
		double& out_area = tup.get<1>();
		double rr_f50 = tup.get<2>();
		double rr_f75 = tup.get<3>();
		double rr_ec = tup.get<4>();
		double rr_ecprev = tup.get<5>();
		double rr_peps = tup.get<6>();
		double rr_hirlam = tup.get<7>();
		double rr_meps = tup.get<8>();
		double rr_gfs = tup.get<9>();

		if (IsMissing(rr_ec))
		{
			continue;
		}

		// Coefficients are duplicated because they are evaluated separately
		// for each grid point.

		double _K1 = K1;
		double _K2 = K2;
		double _K3 = K3;
		double _K4 = K4;
		double _K6 = K6;
		double _K7 = K7;
		double _K8 = K8;

		double ecf50 = 0;
		double ecf75 = 0;
		double ec = 0;
		double ecprev = 0;
		double peps = 0;
		double hirlam = 0;
		double meps = 0;
		double gfs = 0;

		if (IsMissing(rr_f50))
		{
			_K1 = 0;
		}
		else if (rr_f50 > 0.1)
		{
			ecf50 = 1;
		}

		if (IsMissing(rr_f75))
		{
			_K2 = 0;
		}
		else if (rr_f75 > 0.1)
		{
			ecf75 = 1;
		}

		if (IsMissing(rr_ecprev))
		{
			_K3 = 0;
		}
		else if (rr_ecprev > 0.1)
		{
			ecprev = 1;
		}

		if (IsMissing(rr_peps))
		{
			_K4 = 0;
		}
		else if (rr_peps > 30)
		{
			peps = 1;
		}

		if (IsMissing(rr_hirlam))
		{
			_K6 = 0;
		}
		else if (rr_hirlam > 0.05)
		{
			hirlam = 1;
		}

		if (IsMissing(rr_gfs))
		{
			_K7 = 0;
		}
		else if (rr_gfs > 0.05)
		{
			gfs = 1;
		}

		if (IsMissing(rr_meps))
		{
			_K8 = 0;
		}
		else if (rr_meps > 0.05)
		{
			meps = 1;
		}

		if (rr_ec > 0.05)
		{
			ec = 1;
		}

		out_confidence =
		    (_K1 * ecf50 + _K2 * ecf75 + _K3 * ecprev + _K4 * peps + K5 * ec + _K6 * hirlam + _K7 * gfs + _K8 * meps) /
		    (_K1 + _K2 + _K3 + _K4 + K5 + _K6 + _K7 + _K8);

		ASSERT(out_confidence <= 1.01);

		if (out_confidence > 0)
		{
			out_area = 1.;
		}
	}

	// 2. Smoothen area coverage

	/* Macro averages over 4 grid points in all directions
	 * Edit data resolution = 7.5km --> 7.5km * 4 = 30km
	 * ECMWF data resolution = 12.5km --> 12.5km * 3 = 37.5km
	 *
	 * Graphical presentation (here 4 grid points are used):
	 *
	 * x x x x x x x x x
	 * x x x x x x x x x
	 * x x x x x x x x x
	 * x x x x x x x x x
	 * x x x x o x x x x
	 * x x x x x x x x x
	 * x x x x x x x x x
	 * x x x x x x x x x
	 * x x x x x x x x x
	 *
	 * o = center grid point that is under scrutiny now
	 * x = grid point that is used in averaging
	 */

	himan::matrix<double> filter_kernel(9, 9, 1, MissingDouble(), 1 / 81.);

	area = numerical_functions::Filter2D<double>(area, filter_kernel, itsConfiguration->UseCuda());

	// 2. Calculate the probability of precipitation

	auto& result = VEC(myTargetInfo);

	for (auto&& tup : zip_range(result, confidence.Values(), area.Values(), ECprob1, ECprob2))
	{
		double& out_result = tup.get<0>();
		double out_confidence = tup.get<1>();
		double out_area = tup.get<2>();
		double rr_ecprob1 = tup.get<3>();
		double rr_ecprob01 = tup.get<4>();

		if (IsMissing(out_confidence) || IsMissing(out_area))
		{
			continue;
		}

		ASSERT(out_confidence <= 1.01);
		ASSERT(out_area <= 1.01);

		double PoP = out_confidence * out_area * 100;

		if (!IsMissing(rr_ecprob1) && !IsMissing(rr_ecprob01))
		{
			PoP = (3 * PoP + 0.5 * rr_ecprob1 + 0.5 * rr_ecprob01) * 0.25;
		}

		ASSERT(PoP <= 100.01);

		long step = forecastTime.Step().Hours();

		// AJALLISESTI KAUKAISTEN SUURTEN POPPIEN PIENENT�MIST� - Kohta 1

		if (step >= 24 && step < 72 && PoP >= 85)
		{
			PoP = 85;
		}

		// AJALLISESTI KAUKAISTEN SUURTEN POPPIEN PIENENT�MIST�  - Kohta 2

		if (step >= 48 && step < 72 && PoP >= 80)
		{
			PoP = 80;
		}

		// AJALLISESTI KAUKAISTEN SUURTEN POPPIEN PIENENT�MIST�  - Kohta 3

		if (step >= 72 && PoP >= 65)
		{
			PoP = 65;
		}

		out_result = PoP;
	}

	// 3. Find the maximum PoP over nearby grid points

	/* Macro gets the maximum over 15 grid points in all directions
	 * Edit data resolution = 7.5km --> 7.5km * 15 = 111km
	 * ECMWF data resolution = 12.5km --> 12.5km * 9 = 112.5km
	 *
	 */

	filter_kernel = himan::matrix<double>(7, 7, 1, MissingDouble(), 1);

	auto max_result =
	    numerical_functions::Max2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	// 4. Fill the possible holes in the PoP coverage

	for (auto&& tup : zip_range(result, max_result.Values()))
	{
		double& out_result = tup.get<0>();
		double _max_result = tup.get<1>();

		if (IsMissing(out_result) || IsMissing(_max_result))
		{
			continue;
		}

		if (_max_result > 20 && out_result < 10)
		{
			out_result = 10 + rand() % 6;
		}

		// This seems silly ?
		if (out_result < 5)
		{
			out_result = 1 + rand() % 5;
		}
	}

	// 5. Smoothen the final result

	/* Macro averages over 3 grid points in all directions
	 * We need to smooth a lot more to get similar look.
	 */

	filter_kernel = himan::matrix<double>(5, 5, 1, MissingDouble(), 1 / 25.);

	auto smoothenedResult =
	    numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	auto b = myTargetInfo->Base();
	b->data = move(smoothenedResult);

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
