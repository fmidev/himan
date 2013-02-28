/**
 * @file cache.cpp
 *
 * @date Nov 21, 2012
 * @author perämäki
 */

#include "cache.h"
#include "logger_factory.h"
#include "info.h"
#include "plugin_factory.h"
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace himan::plugin;

typedef lock_guard<mutex> Lock;

cache::cache()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("cache"));
}

string cache::UniqueName(const shared_ptr<himan::info>& info)
{
	string forecast_time = "";
	if (!info->OriginDateTime().Empty()) 
	{
		forecast_time = info->OriginDateTime().String("%Y-%m-%d_%H:%M:%S");
	}
	string param = info->Param().Name();
	string level_id = boost::lexical_cast<string>(info->Level().Value());
	string level = boost::lexical_cast<string>(info->Level().Type());
	return forecast_time + '_' + param + '_' + level_id + '_' + level;

}

string cache::UniqueNameFromOptions(const search_options& options)
{
	string forecast_time = (*options.time.OriginDateTime()).String("%Y-%m-%d_%H:%M:%S");
	string param = (options.param).Name();
	string level_id = boost::lexical_cast<string>((options.level).Value());
	string level = boost::lexical_cast<string>((options.level).Type());
	return forecast_time + '_' + param + '_' + level_id + '_' + level;
}


void cache::Insert(vector<shared_ptr<himan::info>>& infos)
{		
	{
		for (size_t i = 0; i < infos.size(); i++)
		{
			string uniqueName = UniqueName(infos[i]);

			if (!(cache_pool::Instance()->Find(uniqueName)))

			cache_pool::Instance()->Insert(uniqueName, infos[i]);

		}	
	}
}

vector<shared_ptr<himan::info>> cache::GetInfo(const search_options& options) 
{
	string uniqueName = UniqueNameFromOptions(options);

	vector<shared_ptr<himan::info>> info;

	if (cache_pool::Instance()->Find(uniqueName))
	{
		info.push_back(cache_pool::Instance()->GetInfo(uniqueName));
	}

	return info;
}

cache_pool* cache_pool::itsInstance = NULL;

cache_pool::cache_pool()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("cache_pool"));
}

cache_pool* cache_pool::Instance()
{
	if (!itsInstance) 
	{
		itsInstance = new cache_pool();
	}

	return itsInstance;
}

bool cache_pool::Find(const string& uniqueName) 
{

	for (map<string, shared_ptr<himan::info>>::iterator it = itsCache.begin(); it != itsCache.end(); ++it)
	{
		if (it->first == uniqueName)
		{
			return true;
		}
	}

	return false;
}

void cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info>& info)
{
	Lock lock(itsInsertMutex);

	itsCache.insert(pair<string, shared_ptr<himan::info>>(uniqueName, info));
	itsLogger->Debug("Data added to cache. UniqueName: " + uniqueName);
	
}

shared_ptr<himan::info> cache_pool::GetInfo(const string& uniqueName)
{
	Lock lock(itsGetMutex);

	return itsCache[uniqueName];
}
