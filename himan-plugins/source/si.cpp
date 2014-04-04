/**
 * @file si.cpp
 *
 * @date Feb 13, 2014
 * @author partio
 */

#include "si.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "fetcher.h"
#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "NFmiQueryData.h"
#include "NFmiSoundingIndexCalculator.h"

boost::shared_ptr<NFmiQueryData> make_shared_ptr(std::shared_ptr<NFmiQueryData>& ptr)
{
    return boost::shared_ptr<NFmiQueryData>(ptr.get(), [ptr](NFmiQueryData*) mutable {ptr.reset();});
}

std::shared_ptr<NFmiQueryData> make_shared_ptr(boost::shared_ptr<NFmiQueryData>& ptr)
{
    return std::shared_ptr<NFmiQueryData>(ptr.get(), [ptr](NFmiQueryData*) mutable {ptr.reset();});
}

si::si() : itsBottomLevel(kHPMissingInt)
{
	itsClearTextFormula = "<multiple algorithms>";

	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("si"));
}

void si::Process(std::shared_ptr<const plugin_configuration> conf)
{
	compiled_plugin_base::Init(conf);

	/*
	 * Set target parameters:
	 * - name 
	 * - univ_id 
	 * - grib2 descriptor 0'00'000
	 *
	 */

	vector<param> theParams;

	theParams.push_back(param("DUMMY"));

	// GRIB 1

	SetParams(theParams);

	shared_ptr<neons> theNeons = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

	itsBottomLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
	itsTopLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "first hybrid level number"));

	Start();
	
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void si::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));
	auto q = dynamic_pointer_cast <querydata> (plugin_factory::Instance()->Plugin("querydata"));

	// Required source parameters

	const param TParam("T-K");
	//const param TDParam("TD-C");
	const param PParam("P-HPA");
	const param RHParam("RH-PRCNT");
	const param HParam("HL-M");
	const param FFParam("FF-MS");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("siThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		// Source infos

		shared_ptr<info> sourceInfo;
		
		for (int levelNumber = itsBottomLevel; levelNumber >= itsTopLevel; levelNumber--)
		{
			level curLevel(kHybrid, levelNumber, "HYBRID");
			
			shared_ptr<info> tempInfo;
			
			try
			{

				// Temperature

				tempInfo = f->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 curLevel,
									 TParam);

				assert(tempInfo->Param().Unit() == kK);

				// grib-plugin does not set universal id number since it does
				// not know anything about it, but we need it in smarttools-library

				tempInfo->Param().UnivId(4);

				if (!sourceInfo)
				{
					sourceInfo = tempInfo;
				}
				else
				{
					sourceInfo->Merge(tempInfo);
				}

				tempInfo.reset();

				// Humidity
				
				tempInfo = f->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 curLevel,
									 RHParam);

				tempInfo->Param().UnivId(13);

				sourceInfo->Merge(tempInfo);

				tempInfo.reset();

				// Pressure

				tempInfo = f->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 curLevel,
									 PParam);

				tempInfo->Param().UnivId(1);

				sourceInfo->Merge(tempInfo);

				tempInfo.reset();

				// Height

				tempInfo = f->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 curLevel,
									 HParam);

				tempInfo->Param().UnivId(3);
				
				sourceInfo->Merge(tempInfo);

				tempInfo.reset();

				// Wind speed

				tempInfo = f->Fetch(itsConfiguration,
									 myTargetInfo->Time(),
									 curLevel,
									 FFParam);

				tempInfo->Param().UnivId(21);

				sourceInfo->Merge(tempInfo);

				tempInfo.reset();

			}
			catch (HPExceptionType e)
			{
				switch (e)
				{
					case kFileDataNotFound:
						itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
						myTargetInfo->Data()->Fill(kFloatMissing);

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
		}

		size_t missingCount = 0;
		size_t count = 0;

		// Scale to correct units

		itsLogger->Info("Scaling source data");
		
		sourceInfo->First();

		bool ret = sourceInfo->Param(TParam);
		assert(ret);

		sourceInfo->FirstTime();
		assert(sourceInfo->SizeTimes() == 1);
		
		for (sourceInfo->ResetLevel(); sourceInfo->NextLevel();)
		{
			ScaleBase(sourceInfo, 1, -constants::kKelvin);
		}

		// data read from neons does not have correct fmi producer id, copy producer
		// info from target info
		
		sourceInfo->Producer(myTargetInfo->Producer());

		// info: convert to querydata

		shared_ptr<NFmiQueryData> qdata = q->CreateQueryData(sourceInfo, false);

		// querydata: std::shared_ptr to boost::shared_ptr
		boost::shared_ptr<NFmiQueryData> bqdata = make_shared_ptr(qdata);

		myThreadedLogger->Info("Calculating sounding index");
		
		// got boost::shared_ptr
		boost::shared_ptr<NFmiQueryData> bsidata = NFmiSoundingIndexCalculator::CreateNewSoundingIndexData(bqdata, "ASDF", false, 0);

		// querydata: boost::shared_ptr to std::shared_ptr
		shared_ptr<NFmiQueryData> sidata = make_shared_ptr(bsidata);
		
		// querydata: convert to info
		myTargetInfo = q->CreateInfo(sidata);

		// Set correct target level to output data
		
		myTargetInfo->FirstLevel();
		myTargetInfo->Level().Type(kHeight);

		string deviceType = "CPU";

		/*
		 * Newbase normalizes scanning mode to bottom left -- if that's not what
		 * the target scanning mode is, we have to swap the data back.
		 */

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			SwapTo(myTargetInfo, kBottomLeft);

			param& p = myTargetInfo->Param();
			
			switch (p.UnivId())
			{
				case 4720:
					p.Name("LCL-HPA");
					break;
				
				case 4721:
					p.Name("LFC-HPA");
					break;

				case 4722:
					p.Name("EL-HPA");
					break;

				case 4723:
					p.Name("CAPE-JKG");
					break;

				case 4724:
					p.Name("CAPE-0-3");
					break;

				case 4725:
					p.Name("CIN-N");
					break;
					
				case 4726:
					p.Name("LCL-M");
					break;

				case 4727:
					p.Name("LFC-M");
					break;

				case 4728:
					p.Name("EL-M");
					break;

				case 4729:
					p.Name("CAPE1040");
					break;

				case 4730:
					p.Name("LCL-500-HPA");
					break;

				case 4731:
					p.Name("LFC-500-HPA");
					break;

				case 4732:
					p.Name("EL-500-HPA");
					break;

				case 4733:
					p.Name("CAPE-500");
					break;

				case 4734:
					p.Name("CAPE-0-3-500");
					break;

				case 4735:
					p.Name("CIN-500-N");
					break;

				case 4736:
					p.Name("LCL-500-M");
					break;

				case 4737:
					p.Name("LFC-500-M");
					break;

				case 4738:
					p.Name("LFC-500-M");
					break;

				case 4739:
					p.Name("CAPE1040-500");
					break;

				case 4740:
					p.Name("LCL-MU-HPA");
					break;

				case 4741:
					p.Name("LFC-MU-HPA");
					break;

				case 4742:
					p.Name("EL-MU-HPA");
					break;

				case 4743:
					p.Name("CAPE-MU-JKG");
					break;

				case 4744:
					p.Name("CAPE-0-3-MU");
					break;

				case 4745:
					p.Name("CIN-MU-N");
					break;

				case 4746:
					p.Name("LCL-MU-M");
					break;

				case 4747:
					p.Name("LFC-MU-M");
					break;

				case 4748:
					p.Name("EL-MU-M");
					break;

				case 4749:
					p.Name("CAPE1040-MU");
					break;
					
				case 4750:
					p.Name("SI-N");
					break;

				case 4751:
					p.Name("LI-N");
					break;

				case 4752:
					p.Name("KINDEX-N");
					break;

				case 4753:
					p.Name("CTI-N");
					break;

				case 4754:
					p.Name("VTI-N");
					break;

				case 4755:
					p.Name("TTI-N");
					break;

				case 4770:
					p.Name("WSH-KT");
					break;

				case 4771:
					p.Name("WSH-1-KT");
					break;

				case 4772:
					p.Name("HLCY-M2S2");
					break;

				case 4773:
					p.Name("HLCY-1-M2S2");
					break;

				case 4774:
					p.Name("FF1500-MS");
					break;

				case 4775:
					p.Name("TPE3-C");
					break;
					
				default:
					throw runtime_error("Unkown sounding parameter calculated: " + myTargetInfo->Param().Name());
					break;
			}

			for (myTargetInfo->ResetLocation(); myTargetInfo->NextLocation();)
			{
				count++;

				if (myTargetInfo->Value() == kFloatMissing)
				{
					missingCount++;
				}
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

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}

	}
}

void si::ScaleBase(shared_ptr<info> anInfo, double scale, double base)
{
	if (scale == 1 && base == 0)
	{
		return;
	}
	
	for (anInfo->ResetLocation(); anInfo->NextLocation(); )
	{
		double v = anInfo->Value();
		anInfo->Value(v * scale + base);
	}
}
