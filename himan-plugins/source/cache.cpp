/*
 * cache.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: partio, perämäki
 */

#include "cache.h"
#include "logger_factory.h"
#include "info.h"
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace himan::plugin;

typedef boost::mutex MutexType;
typedef boost::lock_guard<MutexType> Lock;

static MutexType itsMutex;
cache* cache::itsInstance = NULL;
map<string, shared_ptr<himan::info>> itsCache;

cache::cache()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("cache"));
}

cache* cache::Instance()
{
	if (!itsInstance) 
	{
		itsInstance = new cache();
	}
	return itsInstance;
}

bool cache::Find(const string& uniqueName) 
{
	for (map<string, shared_ptr<himan::info>>::iterator it = itsCache.begin(); it != itsCache.end(); ++it)
	{
		if ( it->first == uniqueName)
			return true;
	}
	return false;
}

string cache::UniqueName(const search_options& options)
{
	string forecast_time = (*options.time.OriginDateTime()).String() + (*options.time.ValidDateTime()).String();
	string param = options.param.Name();
	string levelId = boost::lexical_cast<string>(options.level.Value());
	string level = boost::lexical_cast<string>(options.level.Index());
	string config1 = boost::lexical_cast<string>((*options.configuration).SourceProducer());
	string config2 = boost::lexical_cast<string>((*options.configuration).TargetProducer());
	return forecast_time + param + levelId + level + config1 + config2;
}

void cache::Insert(const search_options& options, vector<shared_ptr<himan::info>> infos)
{
	Lock lock(itsMutex);
	
	string uniqueName = UniqueName(options);
	
	if (!Find(uniqueName)) {
		for (size_t i = 0; i < infos.size(); i++)
		{
			itsCache.insert( pair<string, shared_ptr<himan::info>>(uniqueName, infos[i]));
			itsLogger->Debug("Data added to cache");
		}	
	}	
}

vector<shared_ptr<himan::info>> cache::GetInfo(const search_options& options) 
{
	string uniqueName = UniqueName(options);

	vector<shared_ptr<himan::info>> info;
	
	if (Find(uniqueName))
	{
		info.push_back(itsCache[uniqueName]);
	}
	return info;
}


