/**
 * @file hybrid_height.cpp
 *
 * @date Apr 5, 2013
 * @author peramaki
 */

#include "hybrid_height.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "radon.h"
#include "writer.h"
#include "cache.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("hybrid_height");

const himan::param ZParam("Z-M2S2");
const himan::params GPParam { himan::param("LNSP-N") , himan::param("P-PA") };
const himan::param PParam("P-HPA");
const himan::param TParam("T-K");
const himan::param TGParam("TG-K");

hybrid_height::hybrid_height() : itsBottomLevel(kHPMissingInt), itsUseWriterThreads(false)
{
	itsClearTextFormula = "HEIGHT = prevH + (287/9.81) * (T+prevT)/2 * log(prevP / P)";
	itsLogger = logger_factory::Instance()->GetLog(itsName);

}

hybrid_height::~hybrid_height() {}

void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * For hybrid height we must go through the levels backwards.
	 */

	itsInfo->LevelOrder(kBottomToTop);

	HPDatabaseType dbtype = conf->DatabaseType();
		
	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);

		itsBottomLevel = boost::lexical_cast<int> (n->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
	}

	if ((dbtype == kRadon || dbtype == kNeonsAndRadon) && itsBottomLevel == kHPMissingInt)
	{
		auto r = GET_PLUGIN(radon);

		itsBottomLevel = boost::lexical_cast<int> (r->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
	}
	
	itsUseGeopotential = (itsConfiguration->SourceProducer().Id() == 1 || itsConfiguration->SourceProducer().Id() == 199);
	
	// Using separate writer threads is only efficient when we are calculating with iteration (ECMWF)
	// and if we are using external packing like gzip. In those condition it should give according to initial
	// tests a ~30% increase in total calculation speed.
	
	itsUseWriterThreads = (itsConfiguration->FileCompression() != kNoCompression 
			&& (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles) 
			&& !itsUseGeopotential);

	PrimaryDimension(kTimeDimension);

	SetParams({param("HL-M", 3, 0, 3, 6)});

	Start();
	
	if (itsUseWriterThreads)
	{
		itsWriterGroup.join_all();
	}
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void hybrid_height::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Info("Calculating time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()) + " level " + static_cast<string> (myTargetInfo->Level()));

	if (itsUseGeopotential)
	{
		bool ret = WithGeopotential(myTargetInfo);

		if (!ret)
		{
			myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + static_cast<string> (myTargetInfo->Level()));
			return;
		}
	}
	else
	{
		bool ret = WithIteration(myTargetInfo);

		if (!ret)
		{
			myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + static_cast<string> (myTargetInfo->Level()));
			return;
		}
	}
	
	string deviceType = "CPU";

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}

bool hybrid_height::WithGeopotential(info_t& myTargetInfo)
{
	himan::level H0(himan::kHeight, 0);
	
	if ( itsConfiguration->SourceProducer().Id() == 131)
	{
		H0 = level(himan::kHybrid, 1, "LNSP");
	}

	auto GPInfo = Fetch(myTargetInfo->Time(), myTargetInfo->Level(), ZParam, myTargetInfo->ForecastType(), false);
	auto zeroGPInfo = Fetch(myTargetInfo->Time(), H0, ZParam, myTargetInfo->ForecastType(), false);
	
	if (!GPInfo || !zeroGPInfo)
	{
		return false;
	}
	
	SetAB(myTargetInfo, GPInfo);
	
	LOCKSTEP(myTargetInfo, GPInfo, zeroGPInfo)
	{
		double GP = GPInfo->Value();
		double zeroGP = zeroGPInfo->Value();
		
		if (GP == kFloatMissing || zeroGP == kFloatMissing)
		{
			continue;
		}

		double height = (GP - zeroGP) * himan::constants::kIg;
			
		myTargetInfo->Value(height);
	}
	
	return true;
}

bool hybrid_height::WithIteration(info_t& myTargetInfo)
{

	himan::level H0(himan::kHeight, 0);
	himan::level H2(himan::kHeight, 2);
	
	if ( itsConfiguration->SourceProducer().Id() == 131)
	{
		H2 = level(himan::kHybrid, 137, "GROUND");
		H0 = level(himan::kHybrid, 1, "LNSP");
	}

	const auto forecastTime = myTargetInfo->Time();
	const auto forecastType = myTargetInfo->ForecastType();
	
	level prevLevel;

	bool firstLevel = false;

	if (myTargetInfo->Level().Value() == itsBottomLevel)
	{
		firstLevel = true;
	}
	else
	{
		prevLevel = level(myTargetInfo->Level());
		prevLevel.Value(myTargetInfo->Level().Value() + 1);

		prevLevel.Index(prevLevel.Index() + 1);
	}

	info_t prevTInfo, prevPInfo, prevHInfo;
	
	if (!firstLevel)
	{
		prevTInfo = Fetch(forecastTime, prevLevel, TParam, forecastType, false);
		prevPInfo = Fetch(forecastTime, prevLevel, PParam, forecastType, false);
		prevHInfo = Fetch(forecastTime, prevLevel, param("HL-M"), forecastType, false);
	}
	else
	{
		if ( itsConfiguration->SourceProducer().Id() == 131 )
		{
			prevPInfo = Fetch(forecastTime, H0, GPParam, forecastType, false);
			prevTInfo = Fetch(forecastTime, H2, TParam, forecastType, false);
		}
		else
		{
			prevPInfo = Fetch(forecastTime, H0, GPParam, forecastType, false);
			prevTInfo = Fetch(forecastTime, H2, TGParam, forecastType, false);
		}
	}

	auto PInfo = Fetch(forecastTime, myTargetInfo->Level(), PParam, forecastType, false);
	auto TInfo = Fetch(forecastTime, myTargetInfo->Level(), TParam, forecastType, false);	
	
	if (!prevTInfo || !prevPInfo || ( !prevHInfo && !firstLevel ) || !PInfo || !TInfo)
	{
		itsLogger->Error("Source data missing for level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()) + " step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", stopping processing");
		return false;
	}
	
	SetAB(myTargetInfo, TInfo);

	if (!firstLevel)
	{
		prevHInfo->ResetLocation();
		assert(prevLevel.Value() > myTargetInfo->Level().Value());
	}
	
	double scale = 1.;
	
	// First level for ECMWF is LNSP which needs to be converted
	// to regular pressure

	if (firstLevel)
	{
		if ( itsConfiguration->SourceProducer().Id() == 131 )
		{
			// LNSP to regular pressure
			for (double& val : prevPInfo->Data().Values())
			{
				val = exp (val);
			}
		}
		
		scale = 0.01;
	}

	vector<double> prevHV;

	if (firstLevel)
	{
		prevHV.resize(myTargetInfo->Data().Size(), 0);
	}
	else
	{
		prevHV = VEC(prevHInfo);
	}

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(PInfo), VEC(prevPInfo), VEC(TInfo), VEC(prevTInfo), prevHV))
	{
		double& result 	= tup.get<0>();
		double P 	= tup.get<1>();
		double prevP	= tup.get<2>();
		double T 	= tup.get<3>();
		double prevT 	= tup.get<4>();
		double prevH 	= tup.get<5>();

		if (prevT == kFloatMissing || prevP == kFloatMissing || T == kFloatMissing || P == kFloatMissing || prevH == kFloatMissing)
		{
			continue;
		}

		prevP *= scale;
		
		double deltaZ = 14.628 * (prevT + T) * log(prevP/P);
		double totalHeight = prevH + deltaZ;

		result = totalHeight;

	}

	// Check if we have data in grid. If all values are missing, it is impossible to continue
	// processing any level above this one.

	if (myTargetInfo->Data().Size() == myTargetInfo->Data().MissingCount())
	{
		itsLogger->Error("All data missing for level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()) + " step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", stopping processing");
		return false;
	}

	// If we are writing to a single file, launch a separate writing thread and let the main
	// thread proceed. This speeds up the processing in those machines where writing of files
	// is particularly slow (for example when external packing is used)
	
	if (itsUseWriterThreads)
	{
		// First the data needs to be added to cache, otherwise this current calculating
		// thread cannot find it in the next step

		if (itsConfiguration->UseCache())
		{
			auto c = GET_PLUGIN(cache);

			c->Insert(*myTargetInfo, true);
		}

		// Write to disk asynchronously
		boost::thread* t = new boost::thread(&hybrid_height::Write, this, boost::ref(*myTargetInfo));
		itsWriterGroup.add_thread(t);
		itsLogger->Trace("Writer thread started");
	}

	return true;
}

// This functions copies targetInfo intentionally!
void hybrid_height::Write(himan::info targetInfo)
{
	using namespace himan;
	auto aWriter = GET_PLUGIN(writer);

	assert(itsConfiguration->FileWriteOption() != kSingleFile);

	targetInfo.ResetParam();
	
	while (targetInfo.NextParam())
	{
		aWriter->ToFile(targetInfo, itsConfiguration);
	}
	
	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		compiled_plugin_base::DeallocateMemory(targetInfo);
	}
}

void hybrid_height::WriteToFile(const info& targetInfo, const write_options& writeOptions) 
{
	if (!itsUseWriterThreads)
	{
		compiled_plugin_base::WriteToFile(targetInfo, writeOptions);
	}
}
