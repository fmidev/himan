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
#include <algorithm> // for std::transform
#include <functional> // for std::plus

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "hitool.h"

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

	if (itsConfiguration->Exists("si") && itsConfiguration->GetValue("si") == "true")
	{
		// Showalter Index
		param si("SI-N", 4750, 0, 7, 13);
		theParams.push_back(si);
	}

	if (itsConfiguration->Exists("li") && itsConfiguration->GetValue("li") == "true")
	{
		// Lifted index
		param li("LI-N", 4751, 0, 7, 192);
		theParams.push_back(li);
	}

	if (itsConfiguration->Exists("cti") && itsConfiguration->GetValue("cti") == "true")
	{
		// Cross totals index
		param cti("CTI-N", 4751);
		theParams.push_back(cti);
	}

	if (itsConfiguration->Exists("vti") && itsConfiguration->GetValue("vti") == "true")
	{
		// Vertical Totals index
		param vti("VTI-N", 4754);
		theParams.push_back(vti);
	}

	if (itsConfiguration->Exists("tti") && itsConfiguration->GetValue("tti") == "true")
	{
		// Total Totals index
		param tti("TTI-N", 4755, 0, 7, 4);
		theParams.push_back(tti);
	}

	if (itsConfiguration->Exists("srh") && itsConfiguration->GetValue("srh") == "true")
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

	const param TParam("T-K");
	const param TDParam("TD-C");
	const param HParam("HL-M");
	
	level T850Level(himan::kPressure, 850, "PRESSURE");
	level T700Level(himan::kPressure, 700, "PRESSURE");
	level T500Level(himan::kPressure, 500, "PRESSURE");
	level groundLevel(himan::kHeight, 0, "HEIGHT");
	
	shared_ptr<hitool> h;
	
	vector<double> T500mVector, TD500mVector, P500mVector, H0mVector, H500mVector;
	
	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("stabilityThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		shared_ptr<info> T850Info;
		shared_ptr<info> T700Info;
		shared_ptr<info> T500Info;
		shared_ptr<info> TD850Info;
		shared_ptr<info> TD700Info;
		shared_ptr<info> HInfo;

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

					if (!TD850Info)
					{
						TD850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TDParam);
					}

					if (!TD700Info)
					{
						TD700Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T700Level,
									 TDParam);
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

					if (!TD850Info)
					{
						TD850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TDParam);
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

					// Fetch height of ground

					HInfo = theFetcher->Fetch(itsConfiguration,
									myTargetInfo->Time(),
									groundLevel,
									HParam);
					
					// Fetch average values of T, TD and P over vertical height range 0 ... 500m OVER GROUND
					
					if (!h)
					{
						h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));
						h->Configuration(itsConfiguration);
						
						H500mVector.resize(myTargetInfo->SizeLocations(), 500.);
					}

					if (H0mVector.empty())
					{
						H0mVector = myTargetInfo->Data()->Values();
					}

					if (H500mVector.empty())
					{
						assert(!H0mVector.empty());

						// True upper height = 500 meters + height of ground
						transform(H0mVector.begin(), H0mVector.end(), H500mVector.begin(), H500mVector.begin(), plus<double>());

					}

					h->Time(myTargetInfo->Time());
					
					T500mVector = h->VerticalAverage(param("T-K"), H0mVector, H500mVector);
					TD500mVector = h->VerticalAverage(param("TD-C"), H0mVector, H500mVector);
					P500mVector = h->VerticalAverage(param("P-PA"), H0mVector, H500mVector);

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

					if (!TD850Info)
					{
						TD850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TDParam);
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

					if (!TD850Info)
					{
						TD850Info = theFetcher->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 T850Level,
									 TDParam);
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
			shared_ptr<NFmiGrid> TD850Grid(TD850Info->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> TD700Grid(TD700Info->Grid()->ToNewbaseGrid());

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

			if (TD850Info)
			{
				TD850Grid = shared_ptr<NFmiGrid> (TD850Info->Grid()->ToNewbaseGrid());
			}

			if (TD700Info)
			{
				TD700Grid = shared_ptr<NFmiGrid> (TD700Info->Grid()->ToNewbaseGrid());
			}

			size_t missingCount = 0;
			size_t count = 0;

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			bool equalGrids = CompareGrids({myTargetInfo->Grid(), T850Info->Grid(),
								T700Info->Grid(), T500Info->Grid(), TD850Info->Grid(),
								TD700Info->Grid()});

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			string deviceType = "CPU";

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T850 = kFloatMissing;
				double T700 = kFloatMissing;
				double T500 = kFloatMissing;
				double TD850 = kFloatMissing;
				double TD700 = kFloatMissing;

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
				if (TD850Grid)
				{
					InterpolateToPoint(targetGrid, TD850Grid, equalGrids, TD850);
					assert(TD850 > 0);
				}
				if (TD700Grid)
				{
					InterpolateToPoint(targetGrid, TD700Grid, equalGrids, TD700);
					assert(TD700 > 0);
				}

				double value = kFloatMissing;
				
				if (parName == "KINDEX-N")
				{
					if (T850 == kFloatMissing || T700 == kFloatMissing || T500 == kFloatMissing || TD850 == kFloatMissing || TD700 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = KI(T850, T700, T500, TD850, TD700);
						
						if (value != kFloatMissing)
						{
							// Normalizing value

							value -= constants::kKelvin;
						}
					}

				}
				else if (parName == "SI-N")
				{
					if (T850 == kFloatMissing || T500 == kFloatMissing || TD850 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = SI(T850, T500, TD850);
					}
				}
				else if (parName == "LI-N")
				{
					size_t locationIndex = myTargetInfo->LocationIndex();

					double T500m = T500mVector[locationIndex];
					double TD500m = TD500mVector[locationIndex];
					double P500m = P500mVector[locationIndex];

					if (T500 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = LI(T500, T500m, TD500m, P500m);
					}
				}
				else if (parName == "CTI-N")
				{
					if (T500 == kFloatMissing || TD850 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = CTI(T500, TD850);
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
					if (T850 == kFloatMissing || T500 == kFloatMissing || TD850 == kFloatMissing)
					{
						missingCount++;
					}
					else
					{
						value = TTI(T850, T500, TD850);
					}
				}
				
				if (!myTargetInfo->Value(value))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}

			}

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

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			SwapTo(myTargetInfo, kBottomLeft);
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
double stability::LI(double T500, double T500m, double TD500m, double P500m) const
{
	vector<double> LCL = util::LCL(50000, T500m, TD500m);

	double li = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL[0] == kFloatMissing)
	{
		return li;
	}

	if (LCL[0] <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = util::DryLift(P500m, T500m, TARGET_PRESSURE);

		if (dryT != kFloatMissing)
		{
			li = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud

		double wetT = util::MoistLift(P500m, T500m, TD500m, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			li = T500 - wetT;
		}
	}

	return li;
}

inline
double stability::SI(double T850, double T500, double TD850) const
{
	vector<double> LCL = util::LCL(85000, T850, TD850);

	double si = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL[0] == kFloatMissing)
	{
		return si;
	}
	
	if (LCL[0] <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = util::DryLift(85000, T850, TARGET_PRESSURE);
		
		if (dryT != kFloatMissing)
		{
			si = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud
		
		double wetT = util::MoistLift(85000, T850, TD850, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			si = T500 - wetT;
		}
	}

	return si;
}

/*
inline
double si::StormRelativeHelicity(double UID, double VID, double U_lower, double U_higher, double V_lower, double V_higher)
{
	return ((UID - U_lower) * (V_lower - V_higher)) - ((VID - V_lower) * (U_lower - U_higher));
}
*/
