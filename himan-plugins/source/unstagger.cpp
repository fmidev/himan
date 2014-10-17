/**
 * @file unstagger.cpp
 *
 * Calculate the co-located velocity field for U and V
 *
 * @date Oct 16, 2014
 * @author Tack
 */

#include <boost/lexical_cast.hpp>

#include "util.h"
#include "matrix.h"
#include "unstagger.h"
#include "logger_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "logger_factory.h"

using namespace std;
using namespace himan::plugin;

unstagger::unstagger()
{
	itsClearTextFormula = "U(i) = (U(i-0.5) + U(i+0.5)) / 2";

	itsLogger = logger_factory::Instance()->GetLog("unstagger");
}

void unstagger::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter properties
	 * - name PARM_NAME, this name is found from neons. For example: T-K
	 * - univ_id UNIV_ID, newbase-id, ie code table 204
	 * - grib1 id must be in database
	 * - grib2 descriptor X'Y'Z, http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-2.shtml
	 *
	 */

	param theUVelocityParam("U-MS", 23, 0, 2, 2);
	param theVVelocityParam("V-MS", 24, 0, 2, 3);

	// If this param is also used as a source param for other calculations
	// (like for example dewpoint, relative humidity), unit should also be
	// specified

	theUVelocityParam.Unit(kMs);
	theVVelocityParam.Unit(kMs);

	SetParams({theUVelocityParam, theVVelocityParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void unstagger::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	/*
	 * Required source parameters
	 *
	 * eg. param PParam("P-Pa"); for pressure in pascals
	 *
	 */

	const param UParam("U-MS");
	const param VParam("V-MS");

	// ----	
	
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	
	auto myThreadedLogger = logger_factory::Instance()->GetLog("de_stagger Thread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Debug("Calculating time " + static_cast<string> (*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	info_t UInfo = Fetch(forecastTime, forecastLevel, UParam, false);
	info_t VInfo = Fetch(forecastTime, forecastLevel, VParam, false);

	if (!UInfo || !VInfo)
	{
		myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));

		if (itsConfiguration->StatisticsEnabled())
		{
			// When time or level is skipped, all values are missing
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data()->Size());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data()->Size());
		}

		return;

	}

	// If calculating for hybrid levels, A/B vertical coordinates must be set
	// (copied from source)
	
	SetAB(myTargetInfo, UInfo);

	string deviceType = "CPU";
	// calculate for U
    himan::matrix<double> filter_kernel_U(2,1,1);
    filter_kernel_U.Fill(0.5);

	himan::matrix<double> unstaggered_U = util::Filter2D(*UInfo->Data(), filter_kernel_U);	

    auto unstaggered_U_ptr = make_shared<himan::matrix<double>> (unstaggered_U);
	myTargetInfo->ParamIndex(0);
	myTargetInfo->Grid()->Data(unstaggered_U_ptr);

	// calculate for V
    himan::matrix<double> filter_kernel_V(1,2,1);
    filter_kernel_V.Fill(0.5);

	himan::matrix<double> unstaggered_V = util::Filter2D(*VInfo->Data(), filter_kernel_V);	

    auto unstaggered_V_ptr = make_shared<himan::matrix<double>> (unstaggered_V);
	myTargetInfo->ParamIndex(1);
	myTargetInfo->Grid()->Data(unstaggered_V_ptr);



	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}
