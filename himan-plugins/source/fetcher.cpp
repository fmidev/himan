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
#include <NFmiQueryData.h>
#include "regular_grid.h"
#include "irregular_grid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "grib.h"
#include "neons.h"
#include "radon.h"
#include "param.h"
#include "cache.h"
#include "querydata.h"
#include "csv.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan::plugin;
using namespace std;

const unsigned int SLEEPSECONDS = 10;

shared_ptr<cache> itsCache;

fetcher::fetcher()
	: itsDoLevelTransform(true)
	, itsDoInterpolation(true)
	, itsUseCache(true)
	, itsApplyLandSeaMask(false)
	, itsLandSeaMaskThreshold(0.5)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("fetcher"));
}

shared_ptr<himan::info> fetcher::Fetch(shared_ptr<const plugin_configuration> config,
										forecast_time requestedTime,
										level requestedLevel,
										const params& requestedParams,
										forecast_type requestedType,
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
				ret = Fetch(config, requestedTime, requestedLevel, requestedParams[i], requestedType, readPackedData, false);
				
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

	optsStr += " origintime: " + requestedTime.OriginDateTime().String() + ", step: " + boost::lexical_cast<string> (requestedTime.Step());

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
										forecast_time requestedTime,
										level requestedLevel,
										param requestedParam,
										forecast_type requestedType,
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
			
			search_options opts (requestedTime, requestedParam, newLevel, sourceProd, requestedType, config);

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

			optsStr += " origintime: " + requestedTime.OriginDateTime().String() + ", step: " + boost::lexical_cast<string> (requestedTime.Step());
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
	assert(baseInfo->Dimensions().size());
	
	baseInfo->First();

	if (itsDoInterpolation)
	{
		if (!Interpolate(*baseInfo, theInfos))
		{
			// interpolation failed
			throw kFileDataNotFound;
		}
	}
	else
	{
		itsLogger->Trace("Interpolation disabled");
	}
	
	if (itsApplyLandSeaMask)
	{
		itsLogger->Trace("Applying land-sea mask with threshold " + boost::lexical_cast<string> (itsLandSeaMaskThreshold));

		itsApplyLandSeaMask = false;
	
		if (!ApplyLandSeaMask(config, *theInfos[0], requestedTime, requestedType))
		{
			itsLogger->Warning("Land sea mask apply failed");
		}
		
		itsApplyLandSeaMask = true;

	}

	/*
	 * Insert interpolated data to cache if
	 * 1. Cache is not disabled locally (itsUseCache) AND
	 * 2. Cache is not disabled globally (config->UseCache()) AND
	 * 3a. Caller has requested unpacked data (!readPackedData) OR
	 * 3b. Caller has requested packed data but we were unable to deliver it (readPackedData && !ret->Grid()->IsPackedData())
	 */
	
	if (itsUseCache && config->UseCache() && (!readPackedData || (readPackedData && !theInfos[0]->Grid()->IsPackedData())))
	{
		itsCache->Insert(*theInfos[0]);
	}
	else
	{
		itsLogger->Trace("Cache disabled (local: " + boost::lexical_cast<string> (itsUseCache) + " global: " + boost::lexical_cast<string> (config->UseCache()) + ")");
	}

	baseInfo.reset();

	return theInfos[0];

}

vector<shared_ptr<himan::info>> fetcher::FromFile(const vector<string>& files, search_options& options, bool readContents, bool readPackedData, bool readIfNotMatching)
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
			curInfos = FromGrib(inputFile, options, readContents, readPackedData, readIfNotMatching);
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

		case kCSV:
		{
			curInfos = FromCSV(inputFile, options);
			break;
		}

		default:
			// Unknown file type, cannot proceed
			throw runtime_error("Input file is neither GRIB, NetCDF, QueryData nor CSV");
			break;
		}

		allInfos.insert(allInfos.end(), curInfos.begin(), curInfos.end());

	}

	return allInfos;
}

vector<shared_ptr<himan::info> > fetcher::FromCache(search_options& options)
{
	vector<shared_ptr<himan::info>> infos = itsCache->GetInfo(options);

	return infos;
}

vector<shared_ptr<himan::info> > fetcher::FromGrib(const string& inputFile, search_options& options, bool readContents, bool readPackedData, bool forceCaching)
{

	auto g = GET_PLUGIN(grib);

	vector<shared_ptr<info>> infos = g->FromFile(inputFile, options, readContents, readPackedData, forceCaching);

	return infos;
}

vector<shared_ptr<himan::info>> fetcher::FromQueryData(const string& inputFile, search_options& options, bool readContents)
{

	auto q = GET_PLUGIN(querydata);

	shared_ptr<info> i = q->FromFile(inputFile, options, readContents);

	vector<shared_ptr<info>> theInfos;

	theInfos.push_back(i);

	return theInfos;
}

vector<shared_ptr<himan::info> > fetcher::FromCSV(const string& inputFile, search_options& options)
{

	auto c = GET_PLUGIN(csv);

	auto info = c->FromFile(inputFile, options);

	vector<info_t> infos;
	infos.push_back(info);

	return infos;
}

himan::level fetcher::LevelTransform(const producer& sourceProducer, const param& targetParam,	const level& targetLevel) const
{

	level sourceLevel = targetLevel;

	if (sourceProducer.TableVersion() != kHPMissingInt)
	{
		auto n = GET_PLUGIN(neons);

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

void fetcher::DoInterpolation(bool theDoInterpolation)
{
	itsDoInterpolation = theDoInterpolation;
}

bool fetcher::DoInterpolation() const
{
	return itsDoInterpolation;
}

void fetcher::UseCache(bool theUseCache)
{
	itsUseCache = theUseCache;
}

bool fetcher::UseCache() const
{
	return itsUseCache;
}


vector<shared_ptr<himan::info>> fetcher::FetchFromProducer(search_options& opts, bool readPackedData, bool fetchFromAuxiliaryFiles)
{

	vector<shared_ptr<info>> ret;
	
	itsLogger->Trace("Current producer: " + boost::lexical_cast<string> (opts.prod.Id()));

	// itsLogger->Trace("Current producer: " + sourceProd.Name());

	if (itsUseCache && opts.configuration->UseCache())
	{

		// 1. Fetch data from cache

		if (!itsCache)
		{
			itsCache = GET_PLUGIN(cache);
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
		ret = FromFile(opts.configuration->AuxiliaryFiles(), opts, true, readPackedData, !itsApplyLandSeaMask && !readPackedData);

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

	// 3. Fetch data from Neons or Radon

	vector<string> files;

	if (opts.configuration->ReadDataFromDatabase())
	{
		HPDatabaseType dbtype = opts.configuration->DatabaseType();

		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			// try neons first
			auto n = GET_PLUGIN(neons);

			itsLogger->Trace("Accessing Neons database");
			
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
		
		if (dbtype == kRadon || dbtype == kNeonsAndRadon)
		{
			// try radon next
		
			auto r = GET_PLUGIN(radon);
/*
			itsLogger->Trace("Accessing Radon database");
			
			files = r->Files(opts);

			if (!files.empty())
			{
				ret = FromFile(files, opts, true, readPackedData);

				if (dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->StatisticsEnabled())
				{
					dynamic_pointer_cast<const plugin_configuration> (opts.configuration)->Statistics()->AddToCacheMissCount(1);
				}

				return ret;
 			}
*/
		}
	}

	return ret;

}

bool fetcher::InterpolateArea(info& base, vector<info_t> infos) const
{
	if (infos.size() == 0)
	{
		return false;
	}

	// baseInfo geometry is target_geom in json-file: it is the geometry that the user has
	// requested

	shared_ptr<NFmiQueryData> baseData;
	NFmiFastQueryInfo baseInfo;

	auto q = GET_PLUGIN(querydata);

	for (auto it = infos.begin(); it != infos.end(); ++it)
	{
		if (!(*it))
		{
			continue;
		}
		
		//assert(base.Grid()->Type() == kRegularGrid);
		
		if (base.Grid()->Type() == kRegularGrid && (*it)->Grid()->Type() == kIrregularGrid)
		{
			itsLogger->Error("Unable to intepolate from irregular to regular grid");
			continue;
		}
			
		shared_ptr<grid> interpGrid;
		
		if (base.Grid()->Type() == kRegularGrid)
		{
			interpGrid = make_shared<regular_grid> ();
		}
		else
		{
			interpGrid = make_shared<irregular_grid> ();
		}

		// new data backend

		unpacked targetData(base.Data().SizeX(), base.Data().SizeY(), base.Data().SizeZ(), base.Data().MissingValue());

		if (!baseData)
		{
			baseData = q->CreateQueryData(base, true);
			baseInfo = NFmiFastQueryInfo(baseData.get());
		}

#ifdef HAVE_CUDA

		if ((*it)->Grid()->IsPackedData())
		{
			// We need to unpack
			util::Unpack({(*it)->Grid()});
		}
#endif
		// interpInfo does the actual interpolation, results are stored to targetData

		auto interpData = q->CreateQueryData(**it, true);
		NFmiFastQueryInfo interpInfo (interpData.get());

		size_t i;

		baseInfo.First();
		
		for (baseInfo.ResetLocation(), i = 0; baseInfo.NextLocation(); i++)
		{
			double value = interpInfo.InterpolatedValue(baseInfo.LatLon());
			
			targetData.Set(i, value);
		}

		interpGrid->Data(targetData);
		interpGrid->Projection(base.Grid()->Projection());
		interpGrid->SouthPole(base.Grid()->SouthPole());
		interpGrid->Orientation(base.Grid()->Orientation());
		interpGrid->AB((*it)->Grid()->AB());

		if (interpGrid->Type() == kRegularGrid)
		{
			regular_grid* _g = dynamic_cast<regular_grid*> (interpGrid.get());
			const regular_grid* _bg = dynamic_cast<const regular_grid*> (base.Grid());
			
			assert(_g && _bg);
			
			_g->ScanningMode(_bg->ScanningMode());
			_g->BottomLeft(_bg->BottomLeft());
			_g->TopRight(_bg->TopRight());
		
			// Newbase always normalizes data to +x+y
			// So if source scanning mode is eg. +x-y, we have to swap the interpolated
			// data to back to original scanningmode

			if (_g->ScanningMode() != kBottomLeft)
			{
				HPScanningMode targetMode = _bg->ScanningMode();

				// this is what newbase did to the data
				_g->ScanningMode(kBottomLeft);

				// let's swap it back
				_g->Swap(targetMode);

				assert(targetMode == _g->ScanningMode());
			}
		}
		else
		{
			dynamic_cast<irregular_grid*> (interpGrid.get())->Stations(dynamic_cast<irregular_grid*> (base.Grid())->Stations());
		}

		(*it)->Grid(interpGrid);
	}

	return true;

}

bool fetcher::ReorderPoints(info& base, vector<info_t> infos) const
{
	if (infos.size() == 0)
	{
		return false;
	}

	for (auto it = infos.begin(); it != infos.end(); ++it)
	{
		if (!(*it))
		{
			continue;
		}

		// Worst case: cartesian product ie O(mn) ~ O(n^2)

		auto targetStations = dynamic_cast<irregular_grid*> (base.Grid())->Stations();
		auto sourceStations = dynamic_cast<irregular_grid*> ((*it)->Grid())->Stations();
		auto sourceData = (*it)->Grid()->Data();
		auto newData = matrix<double> (targetStations.size(), 1, 1, kFloatMissing);

		if (targetStations.size() == 0 || sourceStations.size() == 0) return false;

		vector<station> newStations;

		for (size_t i = 0; i < targetStations.size(); i++)
		{
			station s1 = targetStations[i];
			
			bool found = false;
		
			for (size_t j = 0; j < sourceStations.size() && !found; j++)
			{
				station s2 = sourceStations[j];
				
				if (s1 == s2)
				{
					newStations.push_back(s1);
					newData.Set(i, sourceData.At(j));
					
					found = true;
				}
			}

			if (!found)
			{
				itsLogger->Trace("Failed, source data does not contain all the same points as target");
				return false;
			}
		}

		dynamic_cast<irregular_grid*> ((*it)->Grid())->Stations(newStations);
		(*it)->Grid()->Data(newData);
		
	}

	itsLogger->Trace("Success");
	
	return true;
}

bool fetcher::Interpolate(himan::info& baseInfo, vector<info_t>& theInfos) const
{
	bool needInterpolation = false;
	bool needPointReordering = false;

	/*
	 * Possible scenarios:
	 * 1. from regular to regular (basic area interpolation)
	 * 2. from regular to irregular (area to point)
	 * 3. from irregular to irregular (limited functionality, basically just point reordering)
	 * 4. from irregular to regular, not supported
	 */

	

	if (baseInfo.Grid()->Type() != theInfos[0]->Grid()->Type())
	{
		needInterpolation = true;
	}

	// 1.

	if (baseInfo.Grid()->Type() == kRegularGrid && theInfos[0]->Grid()->Type() == kRegularGrid)
	{
		if (*baseInfo.Grid() != *theInfos[0]->Grid())
		{
			needInterpolation = true;
		}		
		else if (baseInfo.Grid()->Type() == kRegularGrid && dynamic_cast<regular_grid*>(baseInfo.Grid())->ScanningMode() != dynamic_cast<regular_grid*>(theInfos[0]->Grid())->ScanningMode())
		{
			// == operator does not test scanning mode !
			itsLogger->Trace("Swapping area");
#ifdef HAVE_CUDA
			if (theInfos[0]->Grid()->IsPackedData())
			{
				// must unpack before swapping

				util::Unpack({theInfos[0]->Grid()});
			}
#endif
			dynamic_cast<regular_grid*>(theInfos[0]->Grid())->Swap(dynamic_cast<regular_grid*>(baseInfo.Grid())->ScanningMode());

		}
	}

	// 2.

	else if (baseInfo.Grid()->Type() == kIrregularGrid && theInfos[0]->Grid()->Type() == kRegularGrid)
	{
		needInterpolation = true;
	}

	// 3.

	else if (baseInfo.Grid()->Type() == kIrregularGrid && theInfos[0]->Grid()->Type() == kIrregularGrid)
	{
		if (*baseInfo.Grid() != *theInfos[0]->Grid())
		{
			needPointReordering = true;
		}
	}

	// 4.

	else if (baseInfo.Grid()->Type() == kRegularGrid && theInfos[0]->Grid()->Type() == kIrregularGrid)
	{
		throw runtime_error("Unable to extrapolate from points to grid");
	}

	if (needInterpolation)
	{
		itsLogger->Trace("Interpolating area");
		return InterpolateArea(baseInfo, theInfos);
	}
	else if (needPointReordering)
	{
		itsLogger->Trace("Reordering points to match");
		return ReorderPoints(baseInfo, theInfos);
	}
	else
	{
		itsLogger->Trace("Grids are natively equal");
	}
	
	return true;
}

bool fetcher::ApplyLandSeaMask(shared_ptr<const plugin_configuration> config, info& theInfo, forecast_time& requestedTime, forecast_type& requestedType)
{
	raw_time originTime = requestedTime.OriginDateTime();
	forecast_time firstTime(originTime, originTime); 
	
	try
	{
		itsApplyLandSeaMask = false;
		
		auto lsmInfo = Fetch(config, firstTime, level(kHeight, 0), param("LC-0TO1"), requestedType, false, false);
		
		itsApplyLandSeaMask = true;
		
		lsmInfo->First();
			
		assert(*lsmInfo->Grid() == *theInfo.Grid());
					
		assert(itsLandSeaMaskThreshold >= -1 && itsLandSeaMaskThreshold <= 1);
		assert(itsLandSeaMaskThreshold != 0);
		
#ifdef HAVE_CUDA
		if (theInfo.Grid()->IsPackedData())
		{
			// We need to unpack
			util::Unpack({theInfo.Grid()});
		}
#endif
		
		assert(!theInfo.Grid()->IsPackedData());
		
		double multiplier = (itsLandSeaMaskThreshold > 0) ? 1. : -1.;

		for (lsmInfo->ResetLocation(), theInfo.ResetLocation(); lsmInfo->NextLocation() && theInfo.NextLocation();)
		{
			double lsm = lsmInfo->Value();
				
			if (theInfo.Value() == kFloatMissing || lsm == kFloatMissing)
			{
				continue;
			}
				
			if (multiplier * lsm <= itsLandSeaMaskThreshold)
			{
				theInfo.Value(kFloatMissing);
			}				
		}
	}
	catch (HPExceptionType& e)
	{
		itsApplyLandSeaMask = true;
		return false;
	}
	
	return true;
}

bool fetcher::ApplyLandSeaMask() const
{
	return itsApplyLandSeaMask;
}

void fetcher::ApplyLandSeaMask(bool theApplyLandSeaMask)
{
	itsApplyLandSeaMask = theApplyLandSeaMask;
}

double fetcher::LandSeaMaskThreshold() const
{
	return itsLandSeaMaskThreshold;
}

void fetcher::LandSeaMaskThreshold(double theLandSeaMaskThreshold)
{
	if (theLandSeaMaskThreshold < -1 || theLandSeaMaskThreshold > 1)
	{
		itsLogger->Fatal("Invalid value for land sea mask threshold: " + boost::lexical_cast<string> (theLandSeaMaskThreshold));
		exit(1);
	}
	
	itsLandSeaMaskThreshold = theLandSeaMaskThreshold;
}
