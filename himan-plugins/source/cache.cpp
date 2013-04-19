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
	vector<param> params;
	vector<level> levels;
	vector<forecast_time> times;

	params.push_back(anInfo->Param());
	levels.push_back(anInfo->Level());
	times.push_back(anInfo->Time());

	shared_ptr<grid> aGrid = anInfo->Grid();

	shared_ptr<info> newInfo (new info(*anInfo));

	newInfo->Params(params);
	newInfo->Levels(levels);
	newInfo->Times(times);
	newInfo->Create();
	newInfo->First();
	newInfo->Grid(aGrid);

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

void cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info> anInfo)
{
	Lock lock(itsInsertMutex);

	itsCache.insert(pair<string, shared_ptr<himan::info>>(uniqueName, anInfo));
	itsLogger->Debug("Data added to cache. UniqueName: " + uniqueName);
	
}

shared_ptr<himan::info> cache_pool::GetInfo(const string& uniqueName)
{
	Lock lock(itsGetMutex);

	return itsCache[uniqueName];
}
