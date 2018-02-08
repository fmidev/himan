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

	Start();
}

void icing::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
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

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, false);
	info_t VvInfo = Fetch(forecastTime, forecastLevel, VvParam, forecastType, false);
	info_t ClInfo = Fetch(forecastTime, forecastLevel, ClParam, forecastType, false);
	info_t PrecFormInfo = Fetch(forecastTime, surface, PrecFormParam, forecastType, false);  // fetch from surface
	info_t PrecInfo = Fetch(forecastTime, surface, PrecParam, forecastType, false);
	info_t ZeroLevelInfo = Fetch(forecastTime, surface, ZeroLevelParam, forecastType, false);
	info_t HeightInfo = Fetch(forecastTime, forecastLevel, HeightParam, forecastType, false);

	level newLevel = forecastLevel;
	newLevel.Value(newLevel.Value() + 2);

	info_t HeightInfo2down = Fetch(forecastTime, newLevel, HeightParam, forecastType, false);

	if (!HeightInfo2down)
	{
		// Hybrid level two below not found, perhaps calculation is done for the first level?
		// First try one level below, and if that's not found then pick the current level

		newLevel.Value(newLevel.Value() - 1);
		HeightInfo2down = Fetch(forecastTime, newLevel, HeightParam, forecastType, false);

		if (!HeightInfo2down)
		{
			HeightInfo2down = HeightInfo;
		}
	}
	if (!TInfo || !VvInfo || !ClInfo || !PrecFormInfo || !PrecInfo || !ZeroLevelInfo || !HeightInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	double VvScale = 1;  // Assume we'll have VV-MMS
	double ClScale = 1000;

	if (VvInfo->Param().Name() == "VV-MS")
	{
		VvScale = 1000;
	}

	ASSERT(TInfo->Grid()->AB() == VvInfo->Grid()->AB() && TInfo->Grid()->AB() == ClInfo->Grid()->AB());

	SetAB(myTargetInfo, TInfo);

	auto h = dynamic_pointer_cast<hitool>(plugin_factory::Instance()->Plugin("hitool"));
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	// Stratus cloud base [m] (0-300m=0-985ft, N>50%
	auto base = h->VerticalHeightGreaterThan(NParam, 0, 300, 0.5);

	string deviceType = "CPU";

	auto& target = VEC(myTargetInfo);

	// LOCKSTEP(myTargetInfo, TInfo, VvInfo, ClInfo)
	for (auto&& tup : zip_range(target, VEC(TInfo), VEC(VvInfo), VEC(ClInfo), VEC(PrecFormInfo), VEC(PrecInfo),
	                            VEC(ZeroLevelInfo), VEC(HeightInfo), base, VEC(HeightInfo2down)))
	{
		double& result = tup.get<0>();
		double T = tup.get<1>();
		double Vv = tup.get<2>();
		double Cl = tup.get<3>();
		double Pf = tup.get<4>();
		double Rr = tup.get<5>();
		double Zl = tup.get<6>();
		double Hl = tup.get<7>();
		double StrBase = tup.get<8>();
		double Hl2down = tup.get<9>();

		if (IsMissingValue({T, Vv, Cl}))
		{
			continue;
		}

		double Icing;
		double TBase = constants::kKelvin;
		int vCor = kHPMissingInt;
		int tCor = kHPMissingInt;

		T = T - TBase;
		Vv *= VvScale;
		Cl *= ClScale;

		// Vertical velocity correction factor

		if (Vv < 0)
		{
			vCor = -1;
		}
		else if ((Vv >= 0) && (Vv <= 50))
		{
			vCor = 0;
		}
		else if ((Vv >= 50) && (Vv <= 100))
		{
			vCor = 1;
		}
		else if ((Vv >= 100) && (Vv <= 200))
		{
			vCor = 2;
		}
		else if ((Vv >= 200) && (Vv <= 300))
		{
			vCor = 3;
		}
		else if ((Vv >= 300) && (Vv <= 1000))
		{
			vCor = 4;
		}
		else if (Vv > 1000)
		{
			vCor = 5;
		}

		// Temperature correction factor

		if ((T <= 0) && (T > -1))
		{
			tCor = -2;
		}
		else if ((T <= -1) && (T > -2))
		{
			tCor = -1;
		}
		else if ((T <= -2) && (T > -3))
		{
			tCor = 0;
		}
		else if ((T <= -3) && (T > -12))
		{
			tCor = 1;
		}
		else if ((T <= -12) && (T > -15))
		{
			tCor = 0;
		}
		else if ((T <= -15) && (T > -18))
		{
			tCor = -1;
		}
		else if (T < -18)
		{
			tCor = -2;
		}
		else
		{
			tCor = 0;
		}

		if ((Cl <= 0) || (T > 0))
		{
			Icing = 0;
		}
		else
		{
			Icing = round(log(Cl) + 6) + vCor + tCor;
		}

		// freezing drizzle, values applied to all model levels below two hybrid level above
		// base of the stratus cloud,
		if (Pf == 4 && !IsMissing(StrBase))
		{
			const double IzingFZDZ = 6 + Rr * 10;

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
			Icing = 7 + Rr * 1.5;
		}

		// Maximum and minimum values for index

		if (Icing > 15)
		{
			Icing = 15;
		}

		if (Icing < 0)
		{
			Icing = 0;
		}

		result = Icing;
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
