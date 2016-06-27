/**
 * @file pop.cpp
 *
 * @date May 26, 2016
 * @author partio
 */

#include <boost/lexical_cast.hpp>

#include "pop.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "numerical_functions.h"
#include "util.h"

#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

pop::pop()
	: itsECEPSGeom("ECFESC250")
	, itsECGeom("ECGLO0100")
	, itsPEPSGeom("PEPSSCAN")
	, itsHirlamGeom("RCR068")
	, itsHarmonieGeom("HARMONIE022")
	, itsGFSGeom("GFS0250")
{
	itsLogger = logger_factory::Instance()->GetLog("pop");
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
		itsPEPSGeom = itsConfiguration->GetValue("hirlam_geom");
	}
	
	if (itsConfiguration->Exists("harmonie_geom"))
	{
		itsPEPSGeom = itsConfiguration->GetValue("harmonie_geom");
	}
	
	if (itsConfiguration->Exists("gfs_geom"))
	{
		itsPEPSGeom = itsConfiguration->GetValue("gfs_geom");
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

	const double K1 = 0.25;   //EC:n Fraktiili 50
	const double K2 = 0.25;   //EC:n Fraktiili 75
	const double K3 = 1;   //EC:n edellinen malliajo
	const double K4 = 1;   //PEPS rr>0.2 mm
	const double K5 = 2;   //EC:n viimeisin malliajo
	const double K6 = 1;   //Hirlamin viimeisin malliajo
	const double K7 = 1;   //GFS:n viimeisin malliajo
	const double K8 = 1;   //Harmonien viimeisin malliajo

	// Current time and level as given to this thread

	const forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	const level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger_factory::Instance()->GetLog("popThread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Debug("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	/*
	 * Required source parameters
	 */

	info_t ECprob1, ECprob01, ECfract50, ECfract75, EC, ECprev, PEPS, Hirlam, Harmonie, GFS;
	
	try 
	{
		auto f = GET_PLUGIN(fetcher);
		auto cnf = make_shared<plugin_configuration> (*itsConfiguration);

		auto prevTime = forecastTime;
		prevTime.OriginDateTime().Adjust(kHourResolution, -12);

		// ECMWF probabilities from EPS
		cnf->SourceProducers({producer(242, 0, 0, "ECM_PROB")});
		cnf->SourceGeomNames({itsECEPSGeom});
		
		// PROB-RR-1 = "RR>= 1mm 6h"
		ECprob1 = f->Fetch(cnf, prevTime, forecastLevel, param("PROB-RR-1"), forecastType, false);
		
		// PROB-RR-01 = "RR>= 0.1mm 6h"
		ECprob01 = f->Fetch(cnf, prevTime, forecastLevel, param("PROB-RR-01"), forecastType, false);

		// ECMWF fractiles from EPS
		cnf->SourceProducers({producer(240, 0, 0, "ECGMTA")});

		// 50th fractile (median)
		ECfract50 = f->Fetch(cnf, prevTime, level(kGround, 0), param("F50-RR-6"), forecastType, false);
		
		// 75th fractile
		ECfract75 = f->Fetch(cnf, prevTime, level(kGround, 0), param("F75-RR-6"), forecastType, false);

		// ECMWF deterministic
		cnf->SourceGeomNames({itsECGeom});
		
		// Current forecast
		EC = f->Fetch(cnf, forecastTime, level(kHeight, 0), param("RRR-KGM2"), forecastType, false);

		// Previous forecast
		ECprev = f->Fetch(cnf, prevTime, level(kHeight, 0), param("RRR-KGM2"), forecastType, false);

		// PEPS
		
		// Peps uses grib2, neons support for this is patchy and we have to give grib centre & ident
		// so that correct name is found

		cnf->SourceGeomNames({itsPEPSGeom});
		cnf->SourceProducers({producer(121,86,121,"PEPSSCAN")});

		// PROB-RR-1 = "RR>= 0.2mm 1h"
		PEPS = f->Fetch(cnf, forecastTime, level(kHeight, 2), param("PROB-RR-1"), forecastType, false);

		// Hirlam
		cnf->SourceGeomNames({itsHirlamGeom});
		cnf->SourceProducers({producer(230, 0, 0, "HL2MTA")});
		Hirlam = f->Fetch(cnf, forecastTime, forecastLevel, param("RRR-KGM2"), forecastType, false);

		// Harmonie
		cnf->SourceGeomNames({itsHarmonieGeom});
		cnf->SourceProducers({producer(210, 0, 0, "AROMTA")});

		forecastTime.StepResolution(kMinuteResolution);
		Harmonie = f->Fetch(cnf, forecastTime, forecastLevel, param("RRR-KGM2"), forecastType, false);
		forecastTime.StepResolution(kHourResolution);

		// GFS
		cnf->SourceGeomNames({itsGFSGeom});
		cnf->SourceProducers({producer(250,0, 0, "GFSMTA")});
		GFS = f->Fetch(cnf, forecastTime, forecastLevel, param("RRR-KGM2"), forecastType, false);
	}
	catch (HPExceptionType e)
	{
		if (e == kFileDataNotFound)
		{
			myThreadedLogger->Error("Some data not found");
		}

		return;

	}	

	const string deviceType = "CPU";

	matrix<double> area(myTargetInfo->Data().SizeX(), myTargetInfo->Data().SizeY(), 1, kFloatMissing, 0); // "A"
	matrix<double> confidence(area.SizeX(), area.SizeY(), 1, kFloatMissing, kFloatMissing); // "C"

#ifdef POINTDEBUG
	int i = -1;
#endif
	// 1. Calculate initial area and confidence of precipitation
	
	for (auto&& tup : zip_range(confidence.Values(), area.Values(), VEC(ECfract50), VEC(ECfract75), VEC(EC), VEC(ECprev), VEC(PEPS), VEC(Hirlam), VEC(Harmonie), VEC(GFS)))
	{
#ifdef POINTDEBUG
		bool print = false;
		myTargetInfo->LocationIndex(++i);

		if (fabs(myTargetInfo->LatLon().X() - 34.0) < 0.1 && fabs(myTargetInfo->LatLon().Y() - 68.81) < 0.1)
		{
			print = true;
		}
#endif
		double& out_confidence		= tup.get<0>();
		double& out_area			= tup.get<1>();
		double rr_f50			= tup.get<2>();
		double rr_f75			= tup.get<3>();
		double rr_ec			= tup.get<4>();
		double rr_ecprev		= tup.get<5>();
		double rr_peps			= tup.get<6>();
		double rr_hirlam		= tup.get<7>();
		double rr_harmonie		= tup.get<8>();
		double rr_gfs			= tup.get<9>();

#ifdef POINTDEBUG	
		if (print)
		{
			std::cout << myTargetInfo->LatLon().X() << "," << myTargetInfo->LatLon().Y() << std::endl
					<< "rr_f50\t" << rr_f50 << std::endl
					<< "rr_f75\t" << rr_f75 << std::endl
					<< "rr_ec\t" << rr_ec << std::endl
					<< "rr_ecprev\t" << rr_ecprev << std::endl
					<< "rr_peps\t" << rr_peps << std::endl
					<< "rr_hirlam\t" << rr_hirlam << std::endl
					<< "rr_harm\t" << rr_harmonie << std::endl
					<< "rr_gfs\t" << rr_gfs << std::endl;
		}
#endif

		if (rr_ec == kFloatMissing || rr_ecprev == kFloatMissing || rr_f50 == kFloatMissing || rr_f75 == kFloatMissing)
		{
			continue;
		}

		double _K4 = K4;
		double _K6 = K6;
		double _K7 = K7;
		double _K8 = K8;

		if (rr_peps == kFloatMissing)
		{
			_K4 = 0;
		}
		
		if (rr_hirlam == kFloatMissing)
		{
			_K6 = 0;
		}
		
		if (rr_gfs == kFloatMissing)
		{
			_K7 = 0;
		}
		
		if (rr_harmonie == kFloatMissing)
		{
			_K8 = 0;
		}

		double f50 = 0;
		double f75 = 0;
		double ec = 0;
		double ecprev = 0;
		double peps = 0;
		double hirlam = 0;
		double harmonie = 0;
		double gfs = 0;
		
		if (rr_f50 > 0.1)
		{
			f50 = 1;
		}
		
		if (rr_f75 > 0.1)
		{
			f75 = 1;
		}
		
		if (rr_ec > 0.05)
		{
			ec = 1;
		}
		
		if (rr_ecprev > 0.1)
		{
			ecprev = 1;
		}
		
		if (rr_peps > 30)
		{
			peps = 1;
		}
		
		if (rr_hirlam > 0.05)
		{
			hirlam = 1;
		}
		
		if (rr_harmonie > 0.05)
		{
			harmonie = 1;
		}
		
		if (rr_gfs > 0.05)
		{
			gfs = 1;
		}

		out_confidence = (K1*f50 + K2*f75 + K3*ecprev + _K4*peps + K5*ec + _K6*hirlam + _K7*gfs + _K8*harmonie) / (K1+K2+K3+_K4+K5+_K6+_K7+_K8); 
		
		assert(out_confidence <= 1.01);

		if (out_confidence > 0)
		{
			out_area = 1.;
		}

#ifdef POINTDEBUG
		if (print)
		{
			std::cout << "(" <<K1<<"*" <<f50 << " + "<< K2 << "*" << f75 << "+" << K3 <<"*" << ecprev <<"+" <<_K4 <<"*"<<peps <<"+" <<K5<<"*"<<ec <<"+" <<_K6<<"*"<<hirlam <<"+" <<_K7<<"*"<<gfs <<" + "<<_K8<<"*"<<harmonie<<") = "
					<< (K1*f50 + K2*f75 + K3*ecprev + _K4*peps + K5*ec + _K6*hirlam + _K7*gfs + _K8*harmonie) << std::endl
					<< "("<<K1<<"+"<<K2<<"+"<<K3<<"+"<<_K4<<"+"<<K5<<"+"<<_K6<<"+"<<_K7<<"+"<<_K8<<")" << " = " 
					<< (K1+K2+K3+_K4+K5+_K6+_K7+_K8) << std::endl
					<< "confidence " << _confidence << " area " << _area << std::endl;
		}
#endif
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
	

	himan::matrix<double> filter_kernel(9,9,1,kFloatMissing,1/81.);

	area = numerical_functions::Filter2D(area, filter_kernel);

	// 2. Calculate the probability of precipitation

	auto& result = VEC(myTargetInfo);
#ifdef POINTDEBUG
	i = -1;
#endif
	
	for (auto&& tup : zip_range(result, confidence.Values(), area.Values(), VEC(ECprob1), VEC(ECprob01)))
	{

#ifdef POINTDEBUG
		myTargetInfo->LocationIndex(++i);
		bool print = false;
                if (fabs(myTargetInfo->LatLon().X() - 34.0) < 0.1 && fabs(myTargetInfo->LatLon().Y() - 68.81) < 0.1)
                {
                        print = true;
                }
#endif

		double& out_result		= tup.get<0>();
		double out_confidence	= tup.get<1>();
		double out_area		= tup.get<2>();
		double rr_ecprob1	= tup.get<3>();
		double rr_ecprob01	= tup.get<4>();

		if (out_confidence == kFloatMissing || out_area == kFloatMissing)
		{
			continue;
		}
		
		assert(out_confidence <= 1.01);
		assert(out_area <= 1.01);
		
		double PoP = out_confidence * out_area * 100;

		if (rr_ecprob1 != kFloatMissing && rr_ecprob01 != kFloatMissing)
		{
			PoP = (3 * PoP + 0.5 * rr_ecprob1 + 0.5 * rr_ecprob01) * 0.25;
#ifdef POINTDEBUG
			if (print)
			{
				std::cout << "(3 * " <<PoP<<" + 0.5 * "<<rr_ecprob1<<" + 0.5 * "<<rr_ecprob01<<") * 0.25 = " << PoP << std::endl;
			}
#endif
		}

		assert(PoP <= 100.01);

		long step = forecastTime.Step();
		assert(forecastTime.StepResolution() == kHourResolution);

		//AJALLISESTI KAUKAISTEN SUURTEN POPPIEN PIENENT�MIST� - Kohta 1
		
		if (step >= 24 && step < 72 && PoP >= 85)
		{
			PoP = 85;
		}
		
		//AJALLISESTI KAUKAISTEN SUURTEN POPPIEN PIENENT�MIST�  - Kohta 2
		
		if (step >= 48 && step < 72 && PoP >= 80)
		{
			PoP = 80;
		}

		//AJALLISESTI KAUKAISTEN SUURTEN POPPIEN PIENENT�MIST�  - Kohta 3
		
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

	filter_kernel = himan::matrix<double> (7,7,1,kFloatMissing,1);

	auto max_result = numerical_functions::Max2D(myTargetInfo->Data(), filter_kernel);

	// 4. Fill the possible holes in the PoP coverage
	
	for (auto&& tup : zip_range(result, max_result.Values()))
	{
		double& out_result		= tup.get<0>();
		double _max_result	= tup.get<1>();

		if (out_result == kFloatMissing || _max_result == kFloatMissing)
		{
			continue;
		}

		if (_max_result > 20 && out_result < 10)
		{
			out_result = 10 + rand() % 6 ;
		}
		
		// This seems silly ?
		if (out_result < 5)
		{
			out_result = 1 + rand() % 5 ;
		}
		
	}
	
	// 5. Smoothen the final result

	/* Macro averages over 3 grid points in all directions
	 * We need to smooth a lot more to get similar look.
	*/
	
	filter_kernel = himan::matrix<double> (5,5,1,kFloatMissing,1/25.);
		
	auto smoothenedResult = numerical_functions::Filter2D(myTargetInfo->Data(), filter_kernel);

	myTargetInfo->Grid()->Data(smoothenedResult);

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
