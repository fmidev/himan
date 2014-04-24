/**
 * @file stability.cpp
 *
 *  @date: Jan 23, 2013
 *  @author aaltom, revised by partio
 */

#include "stability.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

stability::stability()
{
	itsClearTextFormula = "<multiple algorithms>";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("stability"));

}

void stability::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	vector<param> theParams;

	// By default calculate only KINDEX
	
	if (itsConfiguration->Exists("kindex") && itsConfiguration->GetValue("ki") == "false")
	{
		;
	}
	else
	{
		param ki("KINDEX-N", 80, 0, 7, 2);
		theParams.push_back(ki);
	}

	if (itsConfiguration->Exists("si") && itsConfiguration->GetValue("si") != "true")
	{
		// Showalter Index
		param si("SI-N", 4750, 0, 7, 13);
		theParams.push_back(si);
	}

	if (itsConfiguration->Exists("li") && itsConfiguration->GetValue("li") != "true")
	{
		// Lifted index
		param li("LI-N", 4751, 0, 7, 192);
		theParams.push_back(li);
	}

	if (itsConfiguration->Exists("cti") && itsConfiguration->GetValue("cti") != "true")
	{
		// Cross totals index
		param cti("CTI-N", 4751);
		theParams.push_back(cti);
	}

	if (itsConfiguration->Exists("vti") && itsConfiguration->GetValue("vti") != "true")
	{
		// Vertical Totals index
		param vti("VTI-N", 4754);
		theParams.push_back(vti);
	}

	if (itsConfiguration->Exists("tti") && itsConfiguration->GetValue("tti") != "true")
	{
		// Total Totals index
		param tti("TTI-N", 4755, 0, 7, 4);
		theParams.push_back(tti);
	}

	if (itsConfiguration->Exists("srh") && itsConfiguration->GetValue("srh") != "true")
	{
		// Storm relative helicity 0 .. 1 km
		param hlcy("HLCY-1-M2S2", 4773);
		theParams.push_back(hlcy);

		// Storm relative helicity 0 .. 3 km
		hlcy = param("HLCY-M2S2", 4772, 0, 7, 8);
		theParams.push_back(hlcy);
	}
	
	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void stability::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param TdParam("TD-C");  

	level T850Level(himan::kPressure, 850, "PRESSURE");
	level T700Level(himan::kPressure, 700, "PRESSURE");
	level T500Level(himan::kPressure, 500, "PRESSURE");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("stabilityThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		shared_ptr<info> T850Info;
		shared_ptr<info> T700Info;
		shared_ptr<info> T500Info;
		shared_ptr<info> Td850Info;
		shared_ptr<info> Td700Info;

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
		{

			string parName = myTargetInfo->Param().Name();

			myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
									" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()) + " parameter " + parName);
			
			try
			{
				if (parName == "KINDEX-N")
				{
					if (!T850Info)
					{
						T850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TParam);
					}

					if (!T700Info)
					{
						T700Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T700Level,
									 TParam);
					}

					if (!T500Info)
					{
						T500Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T500Level,
									 TParam);
					}

					if (!Td850Info)
					{
						Td850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TdParam);
					}

					if (!Td700Info)
					{
						Td700Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T700Level,
									 TdParam);
					}
				}
				else if (parName == "SI-N")
				{
					if (!T850Info)
					{
						T850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TParam);
					}

					if (!T500Info)
					{
						T500Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T500Level,
									 TParam);
					}

					if (!Td850Info)
					{
						Td850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TdParam);
					}
				}
				else if (parName == "LI-N")
				{
					if (!T500Info)
					{
						T500Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T500Level,
									 TParam);
					}
				}
				else if (parName == "CTI-N")
				{
					if (!T500Info)
					{
						T500Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T500Level,
									 TParam);
					}

					if (!Td850Info)
					{
						Td850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TdParam);
					}
				}
				else if (parName == "VTI-N")
				{
					if (!T850Info)
					{
						T850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TParam);
					}

					if (!T500Info)
					{
						T500Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T500Level,
									 TParam);
					}
				}
				else if (parName == "TTI-N")
				{
					if (!T500Info)
					{
						T500Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T500Level,
									 TParam);
					}

					if (!Td850Info)
					{
						Td850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TdParam);
					}

					if (!T850Info)
					{
						T850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TParam);
					}

				}
			}
			catch (HPExceptionType& e)
			{

				switch (e)
				{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

					if (itsConfiguration->StatisticsEnabled())
					{
						itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
					}

					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
				}
			}

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> T850Grid;
			shared_ptr<NFmiGrid> T700Grid;
			shared_ptr<NFmiGrid> T500Grid;
			shared_ptr<NFmiGrid> Td850Grid(Td850Info->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> Td700Grid(Td700Info->Grid()->ToNewbaseGrid());

			if (T850Info)
			{
				T850Grid = shared_ptr<NFmiGrid> (T850Info->Grid()->ToNewbaseGrid());
			}

			if (T700Info)
			{
				T700Grid = shared_ptr<NFmiGrid> (T700Info->Grid()->ToNewbaseGrid());
			}

			if (T500Info)
			{
				T500Grid = shared_ptr<NFmiGrid> (T500Info->Grid()->ToNewbaseGrid());
			}

			if (Td850Info)
			{
				Td850Grid = shared_ptr<NFmiGrid> (Td850Info->Grid()->ToNewbaseGrid());
			}

			if (Td700Info)
			{
				Td700Grid = shared_ptr<NFmiGrid> (Td700Info->Grid()->ToNewbaseGrid());
			}

			size_t missingCount = 0;
			size_t count = 0;

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			bool equalGrids = CompareGrids({myTargetInfo->Grid(), T850Info->Grid(),
								T700Info->Grid(), T500Info->Grid(), Td850Info->Grid(),
								Td700Info->Grid()});

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			string deviceType = "CPU";

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T850 = kFloatMissing;
				double T700 = kFloatMissing;
				double T500 = kFloatMissing;
				double Td850 = kFloatMissing;
				double Td700 = kFloatMissing;

				if (T850Grid)
				{
					InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
					assert(T850 > 0);
				}
				if (T700Grid)
				{
					InterpolateToPoint(targetGrid, T700Grid, equalGrids, T700);
					assert(T700 > 0);
				}
				if (T500Grid)
				{
					InterpolateToPoint(targetGrid, T500Grid, equalGrids, T500);
					assert(T500 > 0);
				}
				if (Td850Grid)
				{
					InterpolateToPoint(targetGrid, Td850Grid, equalGrids, Td850);
					assert(Td850 > 0);
				}
				if (Td700Grid)
				{
					InterpolateToPoint(targetGrid, Td700Grid, equalGrids, Td700);
					assert(Td700 > 0);
				}

				double value = kFloatMissing;
				
				if (parName == "KINDEX-N")
				{
					if (T850 == kFloatMissing || T700 == kFloatMissing || T500 == kFloatMissing || Td850 == kFloatMissing || Td700 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = KI(T850, T700, T500, Td850, Td700);
					}

				}
				else if (parName == "SI-N")
				{
					if (T850 == kFloatMissing || T500 == kFloatMissing || Td850 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = SI(T850, T500, Td850);
					}
				}
				else if (parName == "LI-N")
				{
					if (T500 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = LI(T500);
					}
				}
				else if (parName == "CTI-N")
				{
					if (T500 == kFloatMissing || Td850 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = CTI(T500, Td850);
					}
				}
				else if (parName == "VTI-N")
				{
					if (T850 == kFloatMissing || T500 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = VTI(T850, T500);
					}
				}
				else if (parName == "TTI-N")
				{
					if (T850 == kFloatMissing || T500 == kFloatMissing || Td850 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = TTI(T850, T500, Td850);
					}
				}

				if (value != kFloatMissing)
				{
					// Normalizing value
					
					value -= constants::kKelvin;
				}
				
				if (!myTargetInfo->Value(value))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}

			}

			/*
			 * Newbase normalizes scanning mode to bottom left -- if that's not what
			 * the target scanning mode is, we have to swap the data back.
			 */

			SwapTo(myTargetInfo, kBottomLeft);

			if (itsConfiguration->StatisticsEnabled())
			{
				itsConfiguration->Statistics()->AddToMissingCount(missingCount);
				itsConfiguration->Statistics()->AddToValueCount(count);
			}

			/*
			 * Now we are done for this level
			 *
			 * Clone info-instance to writer since it might change our descriptor places
			 */

			myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));
		}

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}


inline
double stability::CTI(double TD850, double T500) const
{
	return TD850 - T500;
}

inline
double stability::VTI(double T850, double T500) const
{
	return T850 - T500;
}

inline
double stability::TTI(double T850, double T500, double TD850) const
{
	return CTI(TD850, T500) + VTI(T850, T500);
}

inline
double stability::KI(double T850, double T700, double T500, double TD850, double TD700) const
{
	return T850 - T500 + TD850 - (T700 - TD700);
}

inline
double stability::LI(double T500) const
{
	return kFloatMissing;
}

inline
double stability::SI(double T850, double T500, double TD850) const
{
	vector<double> LCL = util::LCL(85000, T850, TD850);

	double si = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL[0] <= 850)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		// DALR = 9.8C / km
		double dryT = util::DryLift(T850, 85000, TARGET_PRESSURE);
		si = T500 - dryT;

	}
	else
	{
		double wetT = util::MoistLift(85000, T850, TD850, TARGET_PRESSURE);

		si = T500 - wetT;

	}

	return si;
}

