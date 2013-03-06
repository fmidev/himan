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

void cache::Insert(shared_ptr<himan::info> anInfo)
{

	for (anInfo->ResetTime(); anInfo->NextTime(); )
	{

		vector<forecast_time> times;
		times.push_back(anInfo->Time());

		for (anInfo->ResetLevel(); anInfo->NextLevel(); )
		{

			vector<level> levels;
			levels.push_back(anInfo->Level());

			for (anInfo->ResetParam(); anInfo->NextParam(); )
			{
				vector<param> params;
				params.push_back(anInfo->Param());

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
		}
	}
}

void cache::Insert(vector<shared_ptr<himan::info>>& infos)
{		
	for (size_t i = 0; i < infos.size(); i++)
	{
		Insert(infos[i]);
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
