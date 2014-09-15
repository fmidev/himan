/**
 * @file fetcher.cpp
 *
 * @date Nov 21, 2012
 * @author partio
 */

#include "fetcher.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/operations.hpp>
#include "timer_factory.h"
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "grib.h"
#include "neons.h"
#include "param.h"
#include "cache.h"
#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan::plugin;
using namespace std;

const unsigned int SLEEPSECONDS = 10;

shared_ptr<cache> itsCache;

fetcher::fetcher()
	: itsDoLevelTransform(true)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("fetcher"));
}

shared_ptr<himan::info> fetcher::Fetch(shared_ptr<const plugin_configuration> config,
										const forecast_time& requestedTime,
										const level& requestedLevel,
										const params& requestedParams,
										bool readPackedData)
{
	unsigned int waitedSeconds = 0;

	shared_ptr<info> ret;
	
	do
	{
		for (size_t i = 0; i < requestedParams.size(); i++)
		{
			try
			{
				ret = Fetch(config, requestedTime, requestedLevel, requestedParams[i], readPackedData, false);
				
				return ret;
			}
			catch (const HPExceptionType& e)
			{
				if (e != kFileDataNotFound)
				{
					throw;
				}
			}
			catch (const exception& e)
			{
				throw;
			}

		}
		if (config->FileWaitTimeout() > 0)
		{
			itsLogger->Debug("Sleeping for " + boost::lexical_cast<string> (SLEEPSECONDS) + " seconds (cumulative: " + boost::lexical_cast<string> (waitedSeconds) + ")");

			if (!config->ReadDataFromDatabase())
			{
				itsLogger->Warning("file_wait_timeout specified but file read from Neons is disabled");
			}

			sleep(SLEEPSECONDS);
		}

		waitedSeconds += SLEEPSECONDS;
	}
	while (waitedSeconds < config->FileWaitTimeout() * 60);

	string optsStr = "producer(s): ";

	for (size_t prodNum = 0; prodNum < config->SizeSourceProducers(); prodNum++)
	{
		optsStr += boost::lexical_cast<string> (config->SourceProducer(prodNum).Id()) + ",";
	}

	optsStr = optsStr.substr(0, optsStr.size()-1);

	optsStr += " origintime: " + requestedTime.OriginDateTime()->String() + ", step: " + boost::lexical_cast<string> (requestedTime.Step());

	optsStr += " param(s): ";
	
	for (size_t i = 0; i < requestedParams.size(); i++)
	{
		optsStr += requestedParams[i].Name() + ",";
	}

	optsStr = optsStr.substr(0, optsStr.size()-1);

	optsStr += " level: " + string(himan::HPLevelTypeToString.at(requestedLevel.Type())) + " " + boost::lexical_cast<string> (requestedLevel.Value());

	itsLogger->Warning("No valid data found with given search options " + optsStr);

	throw kFileDataNotFound;

}


shared_ptr<himan::info> fetcher::Fetch(shared_ptr<const plugin_configuration> config,
										const forecast_time& requestedTime,
										const level& requestedLevel,
										const param& requestedParam,
										bool readPackedData,
										bool controlWaitTime)
{

	unique_ptr<timer> t = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
	
	if (config->StatisticsEnabled())
	{
		t->Start();
	}
	
	vector<shared_ptr<info>> theInfos;
	unsigned int waitedSeconds = 0;

	level newLevel = requestedLevel;

	do
	{
		// Loop over all source producers if more than one specified

		for (size_t prodNum = 0; prodNum < config->SizeSourceProducers() && theInfos.empty(); prodNum++)
		{

			producer sourceProd(config->SourceProducer(prodNum));
			
			if (itsDoLevelTransform && (requestedLevel.Type() != kHybrid && requestedLevel.Type() != kPressure))
			{
				newLevel = LevelTransform(sourceProd, requestedParam, requestedLevel);

				if (newLevel != requestedLevel)
				{
					itsLogger->Trace("Transform level " + string(HPLevelTypeToString.at(requestedLevel.Type()))
								+ "/" + boost::lexical_cast<string> (requestedLevel.Value())
								+ " to " + HPLevelTypeToString.at(newLevel.Type())
								+ "/" + boost::lexical_cast<string> (newLevel.Value())
								+ " for producer " + boost::lexical_cast<string> (sourceProd.Id())
								+ ", parameter " + requestedParam.Name());
				}
			}			
			
			const search_options opts (requestedTime, requestedParam, newLevel, sourceProd, config);

			theInfos = FetchFromProducer(opts, readPackedData, (waitedSeconds == 0));
		}

		if (controlWaitTime && config->FileWaitTimeout() > 0)
		{
			itsLogger->Debug("Sleeping for " + boost::lexical_cast<string> (SLEEPSECONDS) + " seconds (cumulative: " + boost::lexical_cast<string> (waitedSeconds) + ")");

			if (!config->ReadDataFromDatabase())
			{
				itsLogger->Warning("file_wait_timeout specified but file read from Neons is disabled");
			}

			sleep(SLEEPSECONDS);
		}

		waitedSeconds += SLEEPSECONDS;
	}
	while (theInfos.empty() && controlWaitTime && waitedSeconds < config->FileWaitTimeout() * 60);

	if (config->StatisticsEnabled())
	{
		t->Stop();

		config->Statistics()->AddToFetchingTime(t->GetTime());
	}
	
	/*
	 *  Safeguard; later in the code we do not check whether the data requested
	 *  was actually what was requested.
	 */

	if (theInfos.size() == 0)
	{
		// If this function is called from multi-param Fetch(), do not print
		// any messages yet since we might have another source param coming

		if (controlWaitTime)
		{
			string optsStr = "producer(s): ";

			for (size_t prodNum = 0; prodNum < config->SizeSourceProducers(); prodNum++)
			{
				optsStr += boost::lexical_cast<string> (config->SourceProducer(prodNum).Id()) + ",";
			}

			optsStr = optsStr.substr(0, optsStr.size()-1);

			optsStr += " origintime: " + requestedTime.OriginDateTime()->String() + ", step: " + boost::lexical_cast<string> (requestedTime.Step());
			optsStr += " param: " + requestedParam.Name();
			optsStr += " level: " + string(himan::HPLevelTypeToString.at(requestedLevel.Type())) + " " + boost::lexical_cast<string> (requestedLevel.Value());

			itsLogger->Warning("No valid data found with given search options " + optsStr);
		}

		throw kFileDataNotFound;
	}

	// assert(theConfiguration->SourceProducer() == theInfos[0]->Producer());

	assert((theInfos[0]->Level()) == newLevel);

	assert((theInfos[0]->Time()) == requestedTime);

	assert((theInfos[0]->Param()) == requestedParam);

	auto baseInfo = make_shared<info> (*config->Info());
	baseInfo->First();

	if (*theInfos[0]->Grid() != *baseInfo->Grid())
	{
		itsLogger->Trace("Interpolating area");
		InterpolateArea(baseInfo->Grid(), {theInfos[0]->Grid()});
	}
	else if (theInfos[0]->Grid()->ScanningMode() != baseInfo->Grid()->ScanningMode())
	{
		// == operator does not test scanning mode !
		itsLogger->Trace("Swapping area");
#ifdef HAVE_CUDA
		if (theInfos[0]->Grid()->IsPackedData())
		{
			// must unpack before swapping

			util::Unpack({theInfos[0]->Grid()});

			// Only unpacked and interpolated data is stored to cache
			theInfos[0]->Grid()->PackedData()->Clear();

		}
#endif
		theInfos[0]->Grid()->Swap(baseInfo->Grid()->ScanningMode());

	}
	else
	{
		itsLogger->Trace("Grids are natively equal");
	}

	assert(*baseInfo->Grid() == *theInfos[0]->Grid());

	baseInfo.reset();

	return theInfos[0];

}

vector<shared_ptr<himan::info>> fetcher::FromFile(const vector<string>& files, const search_options& options, bool readContents, bool readPackedData)
{

	vector<shared_ptr<himan::info>> allInfos ;

	for (size_t i = 0; i < files.size(); i++)
	{
		string inputFile = files[i];

		if (!boost::filesystem::exists(inputFile))
		{
			itsLogger->Error("Input file '" + inputFile + "' does not exist");
			continue;
		}

		vector<shared_ptr<himan::info>> curInfos;

		switch (util::FileType(inputFile))
		{

		case kGRIB:
		case kGRIB1:
		case kGRIB2:
		{

			curInfos = FromGrib(inputFile, options, readContents, readPackedData);
			break;

		}

		case kQueryData:
		{
			curInfos = FromQueryData(inputFile, options, readContents);
			break;
		}

		case kNetCDF:
			itsLogger->Error("File is NetCDF");
			break;

		default:
			// Unknown file type, cannot proceed
			throw runtime_error("Input file is neither GRIB, NetCDF nor QueryData");
			break;
		}

		allInfos.insert(allInfos.end(), curInfos.begin(), curInfos.end());

		if (curInfos.size())
		{
			if (options.configuration->UseCache())
			{
				itsCache->Insert(allInfos);
			}

			break; // We found what we were looking for
		}

	}

	return allInfos;
}

vector<shared_ptr<himan::info> > fetcher::FromCache(const search_options& options)
{
	vector<shared_ptr<himan::info>> infos = itsCache->GetInfo(options);

	return infos;
}

vector<shared_ptr<himan::info> > fetcher::FromGrib(const string& inputFile, const search_options& options, bool readContents, bool readPackedData)
{

	shared_ptr<grib> g = dynamic_pointer_cast<grib> (plugin_factory::Instance()->Plugin("grib"));

	vector<shared_ptr<info>> infos = g->FromFile(inputFile, options, readContents, readPackedData);

	return infos;
}

vector<shared_ptr<himan::info>> fetcher::FromQueryData(const string& inputFile, const search_options& options, bool readContents)
{

	shared_ptr<querydata> q = dynamic_pointer_cast<querydata> (plugin_factory::Instance()->Plugin("querydata"));

	shared_ptr<info> i = q->FromFile(inputFile, options, readContents);

	vector<shared_ptr<info>> theInfos;

	theInfos.push_back(i);

	return theInfos;
}

himan::level fetcher::LevelTransform(const producer& sourceProducer, const param& targetParam,	const level& targetLevel) const
{

	level sourceLevel = targetLevel;

	if (sourceProducer.TableVersion() != kHPMissingInt)
	{
		shared_ptr<neons> n = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

		string lvlName = n->NeonsDB().GetGridLevelName(targetParam.Name(), targetLevel.Type(), 204, sourceProducer.TableVersion());

		if (lvlName.empty())
		{
			itsLogger->Trace("No level transformation found for param " + targetParam.Name() + " level " + HPLevelTypeToString.at(targetLevel.Type()));
			return targetLevel;
		}

		HPLevelType lvlType = kUnknownLevel;

		double lvlValue = targetLevel.Value();

		lvlType = HPStringToLevelType.at(boost::to_lower_copy(lvlName));

		if (lvlType == kGround)
		{
			lvlValue = 0;
		}
		
		sourceLevel = level(lvlType, lvlValue, lvlName);
	}
	else
	{
		sourceLevel = targetLevel;
	}

	return sourceLevel;
}

void fetcher::DoLevelTransform(bool theDoLevelTransform)
{
	itsDoLevelTransform = theDoLevelTransform;
}

bool fetcher::DoLevelTransform() const
{
	return itsDoLevelTransform;
}

vector<shared_ptr<himan::info>> fetcher::FetchFromProducer(const search_options& opts, bool readPackedData, bool fetchFromAuxiliaryFiles)
{

	vector<shared_ptr<info>> ret;
	
	itsLogger->Trace("Current producer: " + boost::lexical_cast<string> (opts.prod.Id()));

	// itsLogger->Trace("Current producer: " + sourceProd.Name());

	if (opts.configuration->UseCache())
	{

		// 1. Fetch data from cache

		if (!itsCache)
		{
			itsCache = dynamic_pointer_cast<plugin::cache> (plugin_factory::Instance()->Plugin("cache"));
		}

		ret = FromCache(opts);

		if (ret.size())
		{
			itsLogger->Trace("Data found from cache");

			if (dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->Statistics()->AddToCacheHitCount(1);
			}

			return ret;
		}
	}

	/*
	 *  2. Fetch data from auxiliary files specified at command line
	 *
	 *  Even if file_wait_timeout is specified, auxiliary files is searched
	 *  only once.
	 */

	if (!opts.configuration->AuxiliaryFiles().empty() && fetchFromAuxiliaryFiles)
	{
		ret = FromFile(opts.configuration->AuxiliaryFiles(), opts, true, readPackedData);

		if (!ret.empty())
		{
			itsLogger->Trace("Data found from auxiliary file(s)");

			if (dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->Statistics()->AddToCacheMissCount(1);
			}

			return ret;
		}
		else
		{
			itsLogger->Trace("Data not found from auxiliary file(s)");
		}
	}

	// 3. Fetch data from Neons

	vector<string> files;

	if (opts.configuration->ReadDataFromDatabase())
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		files = n->Files(opts);

		if (!files.empty())
		{
			ret = FromFile(files, opts, true, readPackedData);

			if (dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->Statistics()->AddToCacheMissCount(1);
			}

			return ret;
		}
	}

	return ret;

}

bool fetcher::InterpolateArea(const shared_ptr<grid>& base, initializer_list<shared_ptr<grid>> grids)
{
	if (grids.size() == 0)
	{
		return false;
	}

	const double kInterpolatedValueEpsilon = 0.00001; //<! Max difference between two grid points (if smaller, points are considered the same)

	// baseGrid is target_geom in json-file: it is the geometry that the user has
	// requested
	
	shared_ptr<NFmiGrid> baseGrid;

	for (auto it = grids.begin(); it != grids.end(); ++it)
	{
		if (!*it || *base == **it)
		{
			continue;
		}

		// new data backend
		
		auto targetData = make_shared<d_matrix_t> (base->Data()->SizeX(), base->Data()->SizeY());

		if (!baseGrid)
		{
			baseGrid = shared_ptr<NFmiGrid> (base->ToNewbaseGrid());
		}

#ifdef HAVE_CUDA

		if ((*it)->IsPackedData())
		{
			// We need to unpack
			util::Unpack({*it});

			// Only unpacked and interpolated data is stored to cache
			(*it)->PackedData()->Clear();
		}
#endif	
		auto interpGrid = shared_ptr<NFmiGrid> ((*it)->ToNewbaseGrid());

		// interpGrid does the actual interpolation, results are stored to targetData

		size_t i;

		for (baseGrid->Reset(), i = 0; baseGrid->Next(); i++)
		{
			const NFmiPoint targetLatLonPoint = baseGrid->LatLon(); // Target point in latitude longitude coordinates
			const NFmiPoint sourceGridPoint = interpGrid->LatLonToGrid(targetLatLonPoint); // Grid point that matches to said lat lon point in the source grid

			bool noInterpolation = (
						fabs(sourceGridPoint.X() - round(sourceGridPoint.X())) < kInterpolatedValueEpsilon &&
						fabs(sourceGridPoint.Y() - round(sourceGridPoint.Y())) < kInterpolatedValueEpsilon
			);

			double value = kFloatMissing;

			if (noInterpolation)
			{
				value = interpGrid->FloatValue(sourceGridPoint);
			}
			else
			{
				interpGrid->InterpolateToGridPoint(sourceGridPoint, value);
			}

			targetData->Set(i, value);

		}

		(*it)->Data(targetData);
		(*it)->BottomLeft(base->BottomLeft());
		(*it)->TopRight(base->TopRight());
		(*it)->Projection(base->Projection());
		(*it)->SouthPole(base->SouthPole());
		(*it)->Orientation(base->Orientation());

		// Newbase always normalizes data to +x+y
		// So if source scanning mode is eg. +x-y, newbase has transformed that
		// to +x+y but in info class the scanning mode is still marked as +x-y

		if (base->ScanningMode() != kBottomLeft)
		{
			HPScanningMode targetMode = base->ScanningMode();

			// this is what newbase did to the data

			(*it)->ScanningMode(kBottomLeft);

			// let's swap it back
			(*it)->Swap(targetMode);

			assert(targetMode == (*it)->ScanningMode());

		}
	}

	return true;

}
