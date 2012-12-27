/*
 * neons.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "neons.h"
#include "logger_factory.h"
#include <boost/thread/once.hpp>

using namespace std;
using namespace himan::plugin;

const int MAX_WORKERS = 16;
boost::once_flag flag = BOOST_ONCE_INIT;

neons::neons()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("neons"));

	boost::call_once(&himan::plugin::neons::InitPool, flag);

	//!< todo Calling ctor will always claim one connection --- probably should use some other kind of initialization

    itsNeonsDB = unique_ptr<NFmiNeonsDB> (NFmiNeonsDBPool::GetConnection());

}

neons::~neons()
{
	NFmiNeonsDBPool::Release(&(*itsNeonsDB)); // Return connection back to pool
	itsNeonsDB.release();
}

void neons::InitPool()
{
	NFmiNeonsDBPool::MaxWorkers(MAX_WORKERS);
}

vector<string> neons::Files(const search_options& options)
{

	vector<string> files;

	//!< @todo Use neonsdb connection pool

	itsNeonsDB->Connect(1);

	// const int kFMICodeTableVer = 204;

	string analtime = options.configuration->Info()->OriginDateTime().String("%Y%m%d%H%M");
	string levelvalue = boost::lexical_cast<string> (options.level->Value());

	map<string, string> producerInfo = itsNeonsDB->GetProducerDefinition(options.configuration->SourceProducer());

	if (producerInfo.empty())
	{
		itsLogger->Warning("Producer definition not found for producer " + boost::lexical_cast<string> (options.configuration->SourceProducer()));
		return files;
	}

	string ref_prod = producerInfo["ref_prod"];
	string proddef = producerInfo["producer_id"];
	string no_vers = producerInfo["no_vers"];

	//string param_name = itsNeonsDB->GetGridParameterName(options.param->UnivId(), kFMICodeTableVer, boost::lexical_cast<long>(no_vers));

    //string level_name = itsNeonsDB->GetGridLevelName(options.param->UnivId(), options.level->Type(), kFMICodeTableVer, boost::lexical_cast<long>(no_vers));

	string level_name = options.level->Name();

	vector<vector<string> > gridgeoms = itsNeonsDB->GetGridGeoms(ref_prod, analtime);

	if (gridgeoms.size() == 0)
	{
		itsLogger->Warning("No data found for given search options");
		return files;
	}

	for (size_t i = 0; i < gridgeoms.size(); i++)
	{
		// string geomname = gridgeoms[i][0];
		string tablename = gridgeoms[i][1];
		string dset = gridgeoms[i][2];

	    string query = "SELECT parm_name, lvl_type, lvl1_lvl2, fcst_per, file_location, file_server "
            "FROM "+tablename+" "
            "WHERE dset_id = "+dset+" "
            "AND parm_name = '"+options.param->Name()+"' "
            "AND lvl_type = '"+level_name+"' "
            "AND lvl1_lvl2 = " +levelvalue+" "
            "AND fcst_per = "+boost::lexical_cast<string> (options.time->Step())+" "
            "ORDER BY dset_id, fcst_per, lvl_type, lvl1_lvl2";

	    itsNeonsDB->Query(query);

	    vector<string> values = itsNeonsDB->FetchRow();

	    if (values.empty())
	    {
	    	break;
	    }

	    files.push_back(values[4]);

	}

	return files;

}
