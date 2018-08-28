#include <math.h>

#include "pot.h"

#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "level.h"
#include "logger.h"
#include "matrix.h"
#include "numerical_functions.h"
#include "plugin_factory.h"

using namespace std;
using namespace himan::plugin;

/*
 *
 * class definitions for time_series
 *
 * */

time_series::time_series(param theParam, size_t expectedSize) : itsParam(theParam)
{
	itsInfos.reserve(expectedSize);
}
void time_series::Fetch(std::shared_ptr<const plugin_configuration> config, forecast_time startTime,
                        const HPTimeResolution& timeSpan, int stepSize, int numSteps, const level& forecastLevel,
                        const forecast_type& requestedType = forecast_type(kDeterministic), bool readPackedData = false)
{
	auto f = GET_PLUGIN(fetcher);

	itsInfos.clear();

	for (int i = 0; i < numSteps; ++i)
	{
		try
		{
			auto info = f->Fetch(config, startTime, forecastLevel, itsParam, requestedType);

			startTime.ValidDateTime().Adjust(timeSpan, stepSize);

			itsInfos.push_back(info);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				himan::Abort();
			}
			else
			{
				startTime.ValidDateTime().Adjust(timeSpan, stepSize);
			}
		}
	}
}

void time_series::Param(param theParam)
{
	itsParam = theParam;
}
/*
 *
 * function definitions for "modifier" functions
 *
 * These functions are contained in the POT plugin preliminarily until time_series/generator functionality
 * is implemented to himan-lib. Thus these functions are now written in a more generic way then required
 * for this particular case. Functions expect input_iterators as arguments with iterators pointing to the
 * half-closed interval [begin, end), i.e. end is not included in the interval.
 *
 * */

template <class InputIt>
himan::info_t Max(InputIt begin, InputIt end)
{
	// Empty series
	if (begin == end)
		return nullptr;

	// Find first field that contains data
	while (*begin == nullptr)
	{
		++begin;
		if (begin == end)
			return nullptr;
	}

	// Set first field as first set of maximum values
	auto maxInfo = make_shared<himan::info>((*begin)->Clone());
	++begin;

	for (; begin != end; ++begin)
	{
		// Empty info instance, skip
		if (*begin == nullptr)
			continue;

		// An explicit way to write the zip_range, avoiding the tuples
		auto input = VEC((*begin)).begin();
		auto maximum = VEC(maxInfo).begin();

		auto inputEnd = VEC((*begin)).end();
		auto maximumEnd = VEC(maxInfo).end();

		for (; input != inputEnd, maximum != maximumEnd; ++input, ++maximum)
		{
			*maximum = std::max(*input, *maximum);
		}
	}

	return maxInfo;
}

template <class InputIt>
himan::info_t Mean(InputIt begin, InputIt end)
{
	if (begin == end)
		return nullptr;

	// Find first field that contains data
	while (*begin == nullptr)
	{
		++begin;
		if (begin == end)
			return nullptr;
	}

	// Set first field as first set of mean values
	auto meanInfo = make_shared<himan::info>((*begin)->Clone());
	++begin;

	size_t count = 1;

	for (; begin != end; ++begin)
	{
		// Empty info instance, skip
		if (*begin == nullptr)
			continue;

		// An explicit way to write the zip_range, avoiding the tuples
		auto input = VEC((*begin)).begin();
		auto sum = VEC(meanInfo).begin();

		auto inputEnd = VEC((*begin)).end();
		auto sumEnd = VEC(meanInfo).end();

		for (; input != inputEnd, sum != sumEnd; ++input, ++sum)
		{
			*sum += *input;
		}
		++count;
	}

	// Calculate actual mean values
	double countInv = 1 / static_cast<double>(count);

	for (auto&& val : VEC(meanInfo))
	{
		val *= countInv;
	}

	return meanInfo;
}

himan::matrix<double> area_prob(const himan::matrix<double>& A, const himan::matrix<double>& B,
                                std::function<bool(double)> f)
{
	using himan::MissingDouble;

	// find center position of kernel (half of kernel size)
	himan::matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double probability;  // probability value of the kernel

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// calculate for inner field
	// the weights are used as given on input
	// ASSERT (sum(B) == 1)

	ASSERT(B.MissingCount() == 0);

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			probability = 0.0;
			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const double a = A.At(ii, jj, 0);
					const double b = B.At(mm, nn, 0);

					if (!himan::IsMissingDouble(a) && f(a))
					{
						probability += b;
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = probability;
		}
	}

	// treat boundaries separately
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			probability = 0.0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary

					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (!himan::IsMissingDouble(a) && f(a))
						{
							probability += b;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = probability;
		}
	}

	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			probability = 0.0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (!himan::IsMissingDouble(a) && f(a))
						{
							probability += b;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = probability;
		}
	}

	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			probability = 0.0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (!himan::IsMissingDouble(a) && f(a))
						{
							probability += b;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = probability;
		}
	}

	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			probability = 0.0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (!himan::IsMissingDouble(a) && f(a))
						{
							probability += b;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = probability;
		}
	}

	return ret;
}

/*
 *  plug-in definitions
 *
 * */

pot::pot() : itsStrictMode(false)
{
	itsLogger = logger("pot");
}
void pot::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param POT("POT-PRCNT");

	POT.Unit(kPrcnt);

	if (itsConfiguration->GetValue("strict") == "true")
	{
		itsStrictMode = true;
	}

	SetParams({POT});

	Start();
}

void pot::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 */

	const param CapeParamEC("CAPE-JKG");
	const param CapeParamHiman("CAPE1040-JKG");
	const level MU(kMaximumThetaE, 0);
	const param RainParam("RRR-KGM2");
	const param ELHeight("EL-M");
	const param LCLHeight("LCL-M");
	const param LCLTemp("LCL-K");
	const param LFCHeight("LFC-M");

	forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("pot_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	info_t CAPEMaxInfo, CbTopMaxInfo, LfcMinInfo, RRInfo, LclInfo, LclTempInfo;

	// Fetch params
	CAPEMaxInfo = Fetch(forecastTime, MU, CapeParamHiman, forecastType, false);
	if (!CAPEMaxInfo)
		CAPEMaxInfo = Fetch(forecastTime, forecastLevel, CapeParamEC, forecastType, false);
	CbTopMaxInfo = Fetch(forecastTime, MU, ELHeight, forecastType, false);
	LfcMinInfo = Fetch(forecastTime, MU, LFCHeight, forecastType, false);
	LclInfo = Fetch(forecastTime, MU, LCLHeight, forecastType, false);
	LclTempInfo = Fetch(forecastTime, MU, LCLTemp, forecastType, false);
	RRInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);

	if (!CAPEMaxInfo || !CbTopMaxInfo || !LfcMinInfo || !LclInfo || !LclTempInfo || !RRInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	const double smallRadius = 35;
	const double largeRadius = 62;

	int smallFilterSizeX = 3;
	int smallFilterSizeY = 3;
	int largeFilterSizeX = 5;
	int largeFilterSizeY = 5;

	switch (myTargetInfo->Grid()->Type())
	{
		case kLatitudeLongitude:
		case kRotatedLatitudeLongitude:
			smallFilterSizeX =
			    static_cast<int>((smallRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Di() / 111.0));
			smallFilterSizeY =
			    static_cast<int>((smallRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Dj() / 111.0));
			largeFilterSizeX =
			    static_cast<int>((largeRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Di() / 111.0));
			largeFilterSizeY =
			    static_cast<int>((largeRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Dj() / 111.0));
			break;
		case kStereographic:
		case kLambertConformalConic:
			smallFilterSizeX =
			    static_cast<int>(round(smallRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Di() * 1000.0));
			smallFilterSizeY =
			    static_cast<int>(round(smallRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Dj() * 1000.0));
			largeFilterSizeX =
			    static_cast<int>(round(largeRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Di() * 1000.0));
			largeFilterSizeY =
			    static_cast<int>(round(largeRadius / dynamic_cast<regular_grid*>(myTargetInfo->Grid())->Dj() * 1000.0));
			break;
		default:
			break;
	}

	// filters
	himan::matrix<double> small_filter_kernel(smallFilterSizeX, smallFilterSizeY, 1, MissingDouble(), 1.0);
	himan::matrix<double> large_filter_kernel(largeFilterSizeX, largeFilterSizeY, 1, MissingDouble(),
	                                          1.0 / (largeFilterSizeX * largeFilterSizeY));

	// Cape filtering
	himan::matrix<double> filtered_CAPE = numerical_functions::Max2D(CAPEMaxInfo->Data(), small_filter_kernel);
	CAPEMaxInfo->Grid()->Data(filtered_CAPE);

	// Cb_top filtering
	himan::matrix<double> filtered_CbTop = numerical_functions::Max2D(CbTopMaxInfo->Data(), small_filter_kernel);

	// LFC filtering
	himan::matrix<double> filtered_LFC = numerical_functions::Min2D(LfcMinInfo->Data(), small_filter_kernel);

	// Lift filtering
	himan::matrix<double> filtered_PoLift =
	    area_prob(RRInfo->Data(), large_filter_kernel, [](double d) { return d >= 0.1; });

	// hitool to find Cb/LCL Top temps
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	vector<double> CbTopTemp;

	try
	{
		CbTopTemp = h->VerticalValue(param("T-K"), filtered_CbTop.Values());
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return;
		}

		throw e;
	}

	string deviceType = "CPU";

	// starting point of the algorithm POT v2.5
	for (auto&& tup : zip_range(VEC(myTargetInfo), filtered_CAPE.Values(), filtered_PoLift.Values(),
	                            filtered_CbTop.Values(), CbTopTemp, filtered_LFC.Values(), VEC(LclTempInfo)))
	{
		double& POT = tup.get<0>();
		const double& CAPE = tup.get<1>();
		const double& PoLift = tup.get<2>();
		const double& Cb_top = tup.get<3>();
		const double& Cb_top_temp = tup.get<4>() - himan::constants::kKelvin;
		const double& LFC = tup.get<5>();
		const double& LCL_temp = tup.get<6>() - himan::constants::kKelvin;

		double PoThermoDyn = 0;  // Probability of ThermoDynamics = todennäköisyys ukkosta suosivalle termodynamiikalle
		double PoColdTop = 0;    // Probability of Cold Top, riittävän kylmä pilven toppi
		double PoMixedPhase = 0;  // Probability of Mixed Phase, Todennäköisyys sekafaasikerrokseen
		double PoDepth = 0;       // Probability of Depth, konvektiota tulee tapahtua riittävän paksussa kerroksessa

		const double verticalVelocity = sqrt(2 * CAPE);

		// Relaatio pystynopeuden ja todennäköisyyden välillä:
		// Todennäköisyys kasvaa 0->1, kun pystynopeus kasvaa 5->30 m/s
		if (verticalVelocity >= 5 && verticalVelocity <= 30)
		{
			PoThermoDyn = 0.04 * verticalVelocity - 0.2;
		}

		if (verticalVelocity > 30)
		{
			PoThermoDyn = 1;
		}

		// Salamoinnin kannalta tarpeeksi kylmän pilven topin todennäköisyys kasvaa
		// 0 --> 1, kun pilven topin lämpötila laskee -15C --> -30C
		if (Cb_top_temp <= -15 && Cb_top_temp >= -30)
		{
			PoColdTop = -0.06666667 * Cb_top_temp - 1;
		}

		if (Cb_top_temp < -30)
		{
			PoColdTop = 1;
		}

		// Probability of Mixed Phase
		// Konvektiopilvessä tulee olla tarpeeksi paksu sekafaasikerros, jotta sähköistyminen voi tapahtua.
		// Näin ollen pilven pohjan korkeudella (LCL-tasolla) lämpötila ei saa olla kylmempi kuin ~ -12C
		if (LCL_temp >= -12 && LCL_temp <= 0)
		{
			PoMixedPhase = 0.0833333 * LCL_temp + 1;
		}

		if (LCL_temp > 0)
		{
			PoMixedPhase = 1;
		}

		// Probability of Depth
		// Konvektion tulee tapahtua riittävän paksussa kerroksessa LFC-->EL

		double depth = Cb_top - LFC;

		if (depth >= 2000.0 && depth <= 4000.0)
		{
			PoDepth = 0.0005 * depth - 1;
		}

		if (depth > 4000.0)
		{
			PoDepth = 1;
		}

		POT = PoLift * PoThermoDyn * PoColdTop * PoMixedPhase * PoDepth * 100;
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
