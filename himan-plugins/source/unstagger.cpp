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
#include "plugin_factory.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "cache.h"
#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

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
	
	auto myThreadedLogger = logger_factory::Instance()->GetLog("unstagger Thread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Debug("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	auto f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	info_t UInfo, VInfo;

	try
	{
		f->DoInterpolation(false);
		f->UseCache(false);

		UInfo = f->Fetch(itsConfiguration, forecastTime, forecastLevel, UParam, itsConfiguration->UseCudaForPacking());
		VInfo = f->Fetch(itsConfiguration, forecastTime, forecastLevel, VParam, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (UInfo->Grid()->IsPackedData())
		{
			assert(UInfo->Grid()->PackedData().ClassName() == "simple_packed");

			util::Unpack({UInfo->Grid(), VInfo->Grid()});
		}
#endif
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
		}
		myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	// If calculating for hybrid levels, A/B vertical coordinates must be set
	// (copied from source)
	
	SetAB(myTargetInfo, UInfo);

	string deviceType = "CPU";
	// calculate for U
	himan::matrix<double> filter_kernel_U(2,1,1);
	filter_kernel_U.Fill(0.5);

	himan::matrix<double> unstaggered_U = util::Filter2D(UInfo->Data(), filter_kernel_U);	

	myTargetInfo->ParamIndex(0);
	myTargetInfo->Grid()->Data(unstaggered_U);

	// calculate for V
	himan::matrix<double> filter_kernel_V(1,2,1);
	filter_kernel_V.Fill(0.5);

	himan::matrix<double> unstaggered_V = util::Filter2D(VInfo->Data(), filter_kernel_V);	

	myTargetInfo->ParamIndex(1);
	myTargetInfo->Grid()->Data(unstaggered_V);

	// Re-calculate grid coordinates

	point bl = UInfo->Grid()->BottomLeft(), tr = UInfo->Grid()->TopRight();

	UInfo->Grid()->BottomLeft(point(bl.X() - (UInfo->Grid()->Di() * 0.5), bl.Y()));
	UInfo->Grid()->TopRight(point(tr.X() - (UInfo->Grid()->Di() * 0.5), tr.Y()));

	bl = VInfo->Grid()->BottomLeft(), tr = VInfo->Grid()->TopRight();
	VInfo->Grid()->BottomLeft(point(bl.X(), bl.Y() - (VInfo->Grid()->Dj() * 0.5)));
	VInfo->Grid()->TopRight(point(tr.X(), tr.Y() - (VInfo->Grid()->Dj() * 0.5)));

	auto c = dynamic_pointer_cast <cache> (plugin_factory::Instance()->Plugin("cache"));

	c->Insert(*UInfo);
	c->Insert(*VInfo);

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + 
		boost::lexical_cast<string> (UInfo->Data().MissingCount() + VInfo->Data().MissingCount()) +
		"/" + boost::lexical_cast<string> (UInfo->Data().Size() + VInfo->Data().Size()));


}
