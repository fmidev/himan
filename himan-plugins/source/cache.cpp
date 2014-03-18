/**
 * @file cache.cpp
 *
 * @date Nov 21, 2012
 * @author perämäki
 */

#include "cache.h"
#include "logger_factory.h"
#include "info.h"
#include <time.h>
#include "plugin_factory.h"
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace himan::plugin;

typedef lock_guard<mutex> Lock;

cache::cache()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("cache"));
}

string cache::UniqueName(const shared_ptr<const himan::info> info)
{
	string forecast_time = info->Time().OriginDateTime()->String("%Y-%m-%d_%H:%M:%S");
	string valid_time = info->Time().ValidDateTime()->String("%Y-%m-%d_%H:%M:%S");
	string param = info->Param().Name();
	string level_value = boost::lexical_cast<string>(info->Level().Value());
	string level = HPLevelTypeToString.at(info->Level().Type());
	return forecast_time + '_' + valid_time + '_' + param + '_' + level + '_' + level_value;

}

string cache::UniqueNameFromOptions(const search_options& options)
{
	string forecast_time = (options.time.OriginDateTime())->String("%Y-%m-%d_%H:%M:%S");
	string valid_time = (options.time.ValidDateTime())->String("%Y-%m-%d_%H:%M:%S");
	string param = (options.param).Name();
	string level_value = boost::lexical_cast<string>((options.level).Value());
	string level = HPLevelTypeToString.at(options.level.Type());
	return forecast_time + '_' + valid_time + '_' + param + '_' + level + '_' + level_value;
}

void cache::Insert(shared_ptr<himan::info> anInfo, bool activeOnly)
{

	if (activeOnly)
	{
		SplitToPool(anInfo);
	}
	else
	{
		for (anInfo->ResetTime(); anInfo->NextTime(); )
		{
			for (anInfo->ResetLevel(); anInfo->NextLevel(); )
			{
				for (anInfo->ResetParam(); anInfo->NextParam(); )
				{
					SplitToPool(anInfo);
				}
			}
		}
	}
}

void cache::SplitToPool(const shared_ptr<info> anInfo)
{

	if (anInfo->Grid()->IsPackedData())
	{
		itsLogger->Trace("Ignoring cache push for packed data");
		return;
	}
	vector<param> params;
	vector<level> levels;
	vector<forecast_time> times;

	params.push_back(anInfo->Param());
	levels.push_back(anInfo->Level());
	times.push_back(anInfo->Time());

	auto newInfo = make_shared<info> (*anInfo);

	newInfo->Params(params);
	newInfo->Levels(levels);
	newInfo->Times(times);
	newInfo->Create(anInfo->Grid());
	newInfo->First();

	string uniqueName = UniqueName(newInfo);

	if (!(cache_pool::Instance()->Find(uniqueName)))
	{
		cache_pool::Instance()->Insert(uniqueName, newInfo);
	}
}

void cache::Insert(const vector<shared_ptr<himan::info>>& infos, bool activeOnly)
{		
	for (size_t i = 0; i < infos.size(); i++)
	{
		Insert(infos[i], activeOnly);
		//Clean();
	}	
}

vector<shared_ptr<himan::info>> cache::GetInfo(const search_options& options) 
{
	string uniqueName = UniqueNameFromOptions(options);

	vector<shared_ptr<himan::info>> info;

	if (cache_pool::Instance()->Find(uniqueName))
	{
		info.push_back(cache_pool::Instance()->GetInfo(uniqueName));
		itsLogger->Trace( "Found matching data for " + uniqueName);
	}

	return info;
}

void cache::Clean()
{
	cache_pool::Instance()->Clean();
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

void cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info> anInfo)
{
	Lock lock(itsInsertMutex);

	itsCache.insert(pair<string, shared_ptr<himan::info>>(uniqueName, anInfo));
	time_t timer;
	time(&timer);
	itsCacheItems.insert(pair<string, time_t>(uniqueName, timer));
	itsLogger->Trace("Data added to cache. UniqueName: " + uniqueName);
	
}

void cache_pool::Clean()
{
	Lock lock(itsDeleteMutex);

	for (map<string, time_t>::iterator it = itsCacheItems.begin(); it != itsCacheItems.end(); ++it)
	{
		time_t timer;
		time(&timer);
		if (timer - it->second > 10)
		{
			string name = it->first;
			itsCache.erase(name);
			itsCacheItems.erase(name);
			itsLogger->Trace("Data cleared from cache: " + name);
		}
	}
}

shared_ptr<himan::info> cache_pool::GetInfo(const string& uniqueName)
{
	Lock lock(itsGetMutex);

	return itsCache[uniqueName];
}
