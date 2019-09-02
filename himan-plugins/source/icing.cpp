#include "icing.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"

#include "hitool.h"

using namespace std;
using namespace himan::plugin;

icing::icing()
{
	itsLogger = logger("icing");
}

void icing::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("ICING-N", 480, 0, 19, 7)});

	Start<float>();
}

void icing::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short theThreadIndex)
{
	// Required source parameters

	const param TParam("T-K");
	const params VvParam = {param("VV-MS"), param("VV-MMS")};
	const params NParam({himan::param("N-PRCNT"), himan::param("N-0TO1")});
	const param ClParam("CLDWAT-KGKG");
	const params PrecFormParam({himan::param("PRECFORM2-N"), himan::param("PRECFORM-N")});
	const param PrecParam("RRR-KGM2");
	const param ZeroLevelParam("H0C-M");
	const param HeightParam("HL-M");  // Height of the current hybrid level

	const level surface(himan::kHeight, 0, "HEIGHT");

	auto myThreadedLogger = logger("icingThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	auto TInfo = Fetch<float>(forecastTime, forecastLevel, TParam, forecastType, false);
	auto VvInfo = Fetch<float>(forecastTime, forecastLevel, VvParam, forecastType, false);
	auto ClInfo = Fetch<float>(forecastTime, forecastLevel, ClParam, forecastType, false);
	auto PrecFormInfo = Fetch<float>(forecastTime, surface, PrecFormParam, forecastType, false);  // fetch from surface
	auto PrecInfo = Fetch<float>(forecastTime, surface, PrecParam, forecastType, false);
	auto HeightInfo = Fetch<float>(forecastTime, forecastLevel, HeightParam, forecastType, false);

	auto ZeroLevelInfo = Fetch<float>(forecastTime, surface, ZeroLevelParam, forecastType, false);

	if (!ZeroLevelInfo)
	{
		ZeroLevelInfo = Fetch<float>(forecastTime, level(kIsothermal, 27315), param("HL-M"), forecastType, false);
	}

	level newLevel = forecastLevel;
	newLevel.Value(newLevel.Value() + 2);

	auto HeightInfo2down = Fetch<float>(forecastTime, newLevel, HeightParam, forecastType, false);

	if (!HeightInfo2down)
	{
		// Hybrid level two below not found, perhaps calculation is done for the first level?
		// First try one level below, and if that's not found then pick the current level

		newLevel.Value(newLevel.Value() - 1);
		HeightInfo2down = Fetch<float>(forecastTime, newLevel, HeightParam, forecastType, false);

		if (!HeightInfo2down)
		{
			HeightInfo2down = HeightInfo;
		}
	}
	if (!TInfo || !VvInfo || !ClInfo || !PrecFormInfo || !PrecInfo || !ZeroLevelInfo || !HeightInfo)
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	float VvScale = 1;  // Assume we'll have VV-MMS
	float ClScale = 1000;

	if (VvInfo->Param().Name() == "VV-MS")
	{
		VvScale = 1000;
	}

	SetAB(myTargetInfo, TInfo);

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	// Stratus cloud base [m] (0-300m=0-985ft, N>50%
	auto base = h->VerticalHeightGreaterThan<float>(NParam, 0, 305, 0.5);

	string deviceType = "CPU";

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(TInfo), VEC(VvInfo), VEC(ClInfo), VEC(PrecFormInfo), VEC(PrecInfo),
	                            VEC(ZeroLevelInfo), VEC(HeightInfo), base, VEC(HeightInfo2down)))
	{
		auto& result = tup.get<0>();
		auto T = tup.get<1>();
		auto Vv = tup.get<2>();
		auto Cl = tup.get<3>();
		auto Pf = tup.get<4>();
		auto Rr = tup.get<5>();
		auto Zl = tup.get<6>();
		auto Hl = tup.get<7>();
		auto StrBase = tup.get<8>();
		auto Hl2down = tup.get<9>();

		if (IsMissingValue({T, Vv, Cl}))
		{
			continue;
		}

		float Icing = 0;
		float TBase = static_cast<float>(constants::kKelvin);
		int vCor = kHPMissingInt;
		int tCor = kHPMissingInt;

		T = T - TBase;
		Vv *= VvScale;
		Cl *= ClScale;

		if (Cl > 0 && T <= 0)
		{
			// Vertical velocity correction factor

			vCor = 5;  // vv > 1000

			if (Vv < 0)
			{
				vCor = -1;
			}
			else if (Vv <= 50)
			{
				vCor = 0;
			}
			else if (Vv <= 100)
			{
				vCor = 1;
			}
			else if (Vv <= 200)
			{
				vCor = 2;
			}
			else if (Vv <= 300)
			{
				vCor = 3;
			}
			else if (Vv <= 1000)
			{
				vCor = 4;
			}

			// Temperature correction factor

			tCor = 0;  // T > 0

			if (T > -1 || T < -18)
			{
				tCor = -2;
			}
			else if (T > -2)
			{
				tCor = -1;
			}
			else if (T > -3)
			{
				tCor = 0;
			}
			else if (T > -12)
			{
				tCor = 1;
			}
			else if (T > -15)
			{
				tCor = 0;
			}
			else if (T > -18)
			{
				tCor = -1;
			}

			Icing = round(log(Cl) + 6) + static_cast<float>(vCor) + static_cast<float>(tCor);
		}

		// freezing drizzle, values applied to all model levels below two hybrid level above
		// base of the stratus cloud,
		if (Pf == 4 && !IsMissing(StrBase))
		{
			const float IzingFZDZ = 6 + Rr * 10;

			if (Hl <= StrBase)
			{
				// Below stratus base
				Icing = IzingFZDZ;
			}
			else
			{
				// Above stratus base
				// Correction should be applied *only* to two first hybrid levels above
				// stratus base.

				if (Hl2down <= StrBase)
				{
					Icing = (Icing + IzingFZDZ) / 2;
				}
			}
		}

		// freezing rain, values applied to all model levels in the surface sub-zero layer
		else if (Pf == 5 && Hl < Zl)
		{
			Icing = 7 + Rr * 1.5f;
		}

		// Maximum (15) and minimum (0) values for index

		Icing = fminf(15., Icing);
		Icing = fmaxf(0., Icing);

		result = Icing;
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
