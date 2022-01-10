/**
 * @file windvector.cpp
 */

#include "windvector.h"
#include "forecast_time.h"
#include "interpolate.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <iostream>
#include <math.h>
#include <ogr_spatialref.h>

#include "cache.h"
#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

// Would have been easier to just define U-MS,V-MS,DD-D,FF-MS on different levels ...

const map<HPWindVectorTargetType, pair<vector<string>, vector<string>>> inout = {
    make_pair(kWind, make_pair(vector<string>({"FF-MS", "DD-D"}), vector<string>({"U-MS", "V-MS"}))),
    make_pair(kIce, make_pair(vector<string>({"IFF-MS", "IDD-D"}), vector<string>({"IVELU-MS", "IVELV-MS"}))),
    make_pair(kSea, make_pair(vector<string>({"SFF-MS", "SDD-D"}), vector<string>({"WVELU-MS", "WVELV-MS"}))),
    make_pair(kGust, make_pair(vector<string>({"FFG-MS"}), vector<string>({"WGU-MS", "WGV-MS"})))};

windvector::windvector()
    : itsCalculationTarget(kUnknownElement), itsVectorCalculation(false), itsReverseCalculation(false)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("windvector");
}

vector<himan::param> GetParams(HPWindVectorTargetType ttype, bool reverse)
{
	if (inout.find(ttype) == inout.end())
	{
		throw runtime_error(fmt::format("Invalid target type: {}", ttype));
	}

	const auto& pars = inout.at(ttype);

	std::vector<himan::param> ret;
	std::vector<std::string> tpars;

	if (!reverse)
	{
		tpars = pars.first;
	}
	else
	{
		tpars = pars.second;
	}

	for (const auto& name : tpars)
	{
		ret.emplace_back(name);
	}

	return ret;
}

void windvector::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to windvector
	 */

	vector<param> theParams;

	if (itsConfiguration->Exists("vector") && util::ParseBoolean(itsConfiguration->GetValue("vector")))
	{
		itsVectorCalculation = true;
	}

	if (itsConfiguration->Exists("reverse") && util::ParseBoolean(itsConfiguration->GetValue("reverse")))
	{
		itsReverseCalculation = true;
	}

	if (itsConfiguration->Exists("for_ice") && util::ParseBoolean(itsConfiguration->GetValue("for_ice")))
	{
		itsCalculationTarget = kIce;
	}
	else if (itsConfiguration->Exists("for_sea") && util::ParseBoolean(itsConfiguration->GetValue("for_sea")))
	{
		itsCalculationTarget = kSea;
	}
	else if (itsConfiguration->Exists("for_gust") && util::ParseBoolean(itsConfiguration->GetValue("for_gust")))
	{
		itsCalculationTarget = kGust;
	}
	else
	{
		itsCalculationTarget = kWind;
	}

	auto pars = GetParams(itsCalculationTarget, itsReverseCalculation);
	if (itsVectorCalculation && itsCalculationTarget == kWind)
	{
		pars.emplace_back("DF-MS");
	}

	SetParams(pars);
	Start<float>();

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		windvector_cuda::FreeLongitudeCache();
	}
#endif
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void DoCalculation(vector<float>& A, vector<float>& B, const vector<shared_ptr<himan::info<float>>>& sources,
                   bool reverse, float offset)
{
	std::function<void(const float& U, const float& V, float& speed, float& direction)> SpeedAndDirection =
	    [&offset](const float& U, const float& V, float& speed, float& direction)
	{
		speed = sqrtf(U * U + V * V);
		direction = static_cast<float>(himan::constants::kRad) * atan2(U, V) + offset;
		direction = round(fmodf((direction + 360.f), 360.f));
	};

	std::function<void(const float& speed, const float& direction, float& U, float& V)> UV =
	    [&offset](const float& speed, const float& direction, float& U, float& V)
	{
		float sinv, cosv;
		sincosf(fmodf(direction + offset, 360.f) * static_cast<float>(himan::constants::kDeg), &sinv, &cosv);
		U = speed * sinv;
		V = speed * cosv;
	};

	std::function<void(const float& a, const float& b, float& c, float& d)> call = (reverse) ? UV : SpeedAndDirection;

	for (auto&& tup : zip_range(A, B, VEC(sources[0]), VEC(sources[1])))
	{
		float& ta = tup.get<0>();
		float& tb = tup.get<1>();
		float sa = tup.get<2>();
		float sb = tup.get<3>();

		if (himan::IsMissing(sa) || himan::IsMissing(sb))
		{
			continue;
		}

		call(sa, sb, ta, tb);
	}
}

void windvector::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	const auto sourceParams = GetParams(itsCalculationTarget, !itsReverseCalculation);

	float directionOffset = 180;  // For wind direction add this

	switch (itsCalculationTarget)
	{
		case kSea:
		case kIce:
			directionOffset = 0;
			break;
		default:
			break;
	}

	auto myThreadedLogger = logger("windvectorThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	string deviceType;

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda() && !itsVectorCalculation && !itsReverseCalculation)
	{
		deviceType = "GPU";
		windvector_cuda::RunCuda(itsConfiguration, myTargetInfo, sourceParams[0], sourceParams[1],
		                         itsCalculationTarget);
	}
	else
#endif
	{
		deviceType = "CPU";

		vector<shared_ptr<info<float>>> sources;

		for (const auto& sourceParam : sourceParams)
		{
			auto src =
			    FetchOne(forecastTime, forecastLevel, sourceParam, forecastType, itsConfiguration->UseCudaForPacking());

			if (!src)
			{
				myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
				                         static_cast<string>(forecastLevel));
				return;
			}

			sources.push_back(src);
		}

		for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>();)
		{
			SetAB(myTargetInfo, sources[0]);
		}

		ASSERT(sources[0]->Grid()->Type() == sources[1]->Grid()->Type());

#ifdef HAVE_CUDA
		if (sources[0]->PackedData()->HasData())
		{
			util::Unpack<float>(sources, false);
		}
#endif
		// TODO: a better way should exist to provide vector component rotation
		// than creating a dummy area

		if (!itsReverseCalculation)
		{
			latitude_longitude_grid dummy(kBottomLeft, point(), point(), 0, 0, earth_shape<double>());
			interpolate::RotateVectorComponents(sources[0]->Grid().get(), &dummy, *sources[0], *sources[1],
			                                    itsConfiguration->UseCuda());

			auto c = GET_PLUGIN(cache);

			c->Replace(sources[0]);
			c->Replace(sources[1]);
		}
		else
		{
			myTargetInfo->Grid()->UVRelativeToGrid(sources[0]->Grid()->UVRelativeToGrid());
		}

		myTargetInfo->Index<param>(0);
		auto& A = VEC(myTargetInfo);
		vector<float> B(A.size(), himan::MissingFloat());

		DoCalculation(A, B, sources, itsReverseCalculation, directionOffset);

		if (myTargetInfo->Size<param>() > 1)
		{
			myTargetInfo->Index<param>(1);
			myTargetInfo->Data().Set(B);
		}

		if (itsVectorCalculation)
		{
			myTargetInfo->Index<param>(2);
			auto& DF = VEC(myTargetInfo);

			shared_ptr<himan::info<float>> speed, direction;

			if (itsReverseCalculation)
			{
				speed = sources[0];
				direction = sources[1];
			}
			else
			{
				myTargetInfo->Index<param>(0);
				speed = make_shared<info<float>>(*myTargetInfo);
				myTargetInfo->Index<param>(1);
				direction = make_shared<info<float>>(*myTargetInfo);
			}

			for (auto&& tup : zip_range(DF, VEC(speed), VEC(direction)))
			{
				float& df = tup.get<0>();
				const float& spd = tup.get<1>();
				const float& dir = tup.get<2>();

				df = roundf(dir * 0.1f) + 100.f * roundf(spd);
			}
		}
	}

	myThreadedLogger.Info(fmt::format("[{}] Missing values: {}/{}", deviceType, myTargetInfo->Data().MissingCount(),
	                                  myTargetInfo->Data().Size()));
}

shared_ptr<himan::info<float>> windvector::FetchOne(const forecast_time& theTime, const level& theLevel,
                                                    const param& theParam, const forecast_type& theType,
                                                    bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);
	f->DoVectorComponentRotation(true);

	shared_ptr<info<float>> ret;

	try
	{
		ret = f->Fetch<float>(itsConfiguration, theTime, theLevel, theParam, theType,
		                      itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack<float>({ret}, false);

			auto c = GET_PLUGIN(cache);

			c->Insert(ret);
		}
#endif
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
		}
	}

	return ret;
}
