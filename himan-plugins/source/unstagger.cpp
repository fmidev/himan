/**
 * @file unstagger.cpp
 *
 * Calculate the co-located velocity field for U and V
 *
 */
#include "forecast_time.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "logger.h"
#include "matrix.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "stereographic_grid.h"
#include "unstagger.cuh"
#include "unstagger.h"
#include "util.h"

#include "cache.h"
#include "fetcher.h"
#include "radon.h"

using namespace std;
using namespace himan::plugin;

unstagger::unstagger()
{
	itsLogger = logger("unstagger");
}

void unstagger::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

#ifdef HAVE_CUDA
	// Initialise sparse matrix for interpolation grid
	if (itsConfiguration->UseCuda())
	{
		auto r = GET_PLUGIN(radon);

		string query = "SELECT ni,nj FROM geom WHERE name = '" + itsConfiguration->TargetGeomName() + "' ";
		r->RadonDB().Query(query);
		vector<string> gridSize = r->RadonDB().FetchRow();

		unstagger_cuda::Init(stoi(gridSize[0]), stoi(gridSize[1]));
	}
#endif

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
	auto myThreadedLogger = logger("unstagger Thread #" + to_string(threadIndex));

	if (myTargetInfo->Grid()->Class() != kRegularGrid)
	{
		itsLogger.Error("Unable to stagger irregular grids");
		return;
	}

	const param UParam("U-MS");
	const param VParam("V-MS");

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	auto f = GET_PLUGIN(fetcher);

	info_t UInfo, VInfo;

	try
	{
		f->DoInterpolation(false);
		f->UseCache(false);

		UInfo = f->Fetch(itsConfiguration, forecastTime, forecastLevel, UParam, forecastType,
		                 itsConfiguration->UseCudaForPacking());
		VInfo = f->Fetch(itsConfiguration, forecastTime, forecastLevel, VParam, forecastType,
		                 itsConfiguration->UseCudaForPacking());

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
		myThreadedLogger.Info("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                      static_cast<string>(forecastLevel));
		return;
	}

	// If calculating for hybrid levels, A/B vertical coordinates must be set
	// (copied from source)

	myTargetInfo->ParamIndex(0);
	SetAB(myTargetInfo, UInfo);

	myTargetInfo->ParamIndex(1);
	SetAB(myTargetInfo, VInfo);

	string deviceType;

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";
		std::pair<std::vector<double>, std::vector<double>> unstaggered_UV;
		unstaggered_UV = unstagger_cuda::Process(UInfo->Data().Values(), VInfo->Data().Values());

		myTargetInfo->ParamIndex(0);
		myTargetInfo->Grid()->Data().Set(unstaggered_UV.first);

		myTargetInfo->ParamIndex(1);
		myTargetInfo->Grid()->Data().Set(unstaggered_UV.second);
	}
	else
#endif
	{
		deviceType = "CPU";
		// calculate for U
		himan::matrix<double> filter_kernel_U(2, 1, 1, MissingDouble());
		filter_kernel_U.Fill(0.5);

		himan::matrix<double> unstaggered_U = numerical_functions::Filter2D(UInfo->Data(), filter_kernel_U);

		myTargetInfo->ParamIndex(0);
		myTargetInfo->Grid()->Data(unstaggered_U);

		// calculate for V
		himan::matrix<double> filter_kernel_V(1, 2, 1, MissingDouble());
		filter_kernel_V.Fill(0.5);

		himan::matrix<double> unstaggered_V = numerical_functions::Filter2D(VInfo->Data(), filter_kernel_V);

		myTargetInfo->ParamIndex(1);
		myTargetInfo->Grid()->Data(unstaggered_V);
	}

	// Re-calculate grid coordinates

	point u_bl = UInfo->Grid()->BottomLeft(), u_tr = UInfo->Grid()->TopRight();
	point v_bl = VInfo->Grid()->BottomLeft(), v_tr = VInfo->Grid()->TopRight();
	double u_di = UInfo->Grid()->Di();
	double v_dj = VInfo->Grid()->Dj();

	// this is ugly, there has to be a better solution
	switch (UInfo->Grid()->Type())
	{
		case kLatitudeLongitude:
			dynamic_cast<latitude_longitude_grid*>(UInfo->Grid())->BottomLeft(point(u_bl.X() - (u_di * 0.5), u_bl.Y()));
			dynamic_cast<latitude_longitude_grid*>(UInfo->Grid())->TopRight(point(u_tr.X() - (u_di * 0.5), u_tr.Y()));
			dynamic_cast<latitude_longitude_grid*>(VInfo->Grid())->BottomLeft(point(v_bl.X(), v_bl.Y() - (v_dj * 0.5)));
			dynamic_cast<latitude_longitude_grid*>(VInfo->Grid())->TopRight(point(v_tr.X(), v_tr.Y() - (v_dj * 0.5)));
			break;
		case kRotatedLatitudeLongitude:
			dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())
			    ->BottomLeft(point(u_bl.X() - (u_di * 0.5), u_bl.Y()));
			dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())
			    ->TopRight(point(u_tr.X() - (u_di * 0.5), u_tr.Y()));
			dynamic_cast<rotated_latitude_longitude_grid*>(VInfo->Grid())
			    ->BottomLeft(point(v_bl.X(), v_bl.Y() - (v_dj * 0.5)));
			dynamic_cast<rotated_latitude_longitude_grid*>(VInfo->Grid())
			    ->TopRight(point(v_tr.X(), v_tr.Y() - (v_dj * 0.5)));
			break;
		case kStereographic:
			dynamic_cast<stereographic_grid*>(UInfo->Grid())->BottomLeft(point(u_bl.X() - (u_di * 0.5), u_bl.Y()));
			dynamic_cast<stereographic_grid*>(UInfo->Grid())->TopRight(point(u_tr.X() - (u_di * 0.5), u_tr.Y()));
			dynamic_cast<stereographic_grid*>(VInfo->Grid())->BottomLeft(point(v_bl.X(), v_bl.Y() - (v_dj * 0.5)));
			dynamic_cast<stereographic_grid*>(VInfo->Grid())->TopRight(point(v_tr.X(), v_tr.Y() - (v_dj * 0.5)));
			break;
		default:
			throw runtime_error("Unsupported grid type: " + HPGridTypeToString.at(UInfo->Grid()->Type()));
	}

	auto c = GET_PLUGIN(cache);

	c->Insert(*UInfo);
	c->Insert(*VInfo);

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " +
	                      to_string(UInfo->Data().MissingCount() + VInfo->Data().MissingCount()) + "/" +
	                      to_string(UInfo->Data().Size() + VInfo->Data().Size()));
}
