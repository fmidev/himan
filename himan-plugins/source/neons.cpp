/*
 * neons.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "neons.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include <thread>
#include <sstream>

using namespace std;
using namespace himan::plugin;

#define HIMAN_AUXILIARY_INCLUDE

#include "util.h"

#undef HIMAN_AUXILIARY_INCLUDE

const int MAX_WORKERS = 16;
once_flag oflag;

neons::neons() : itsInit(false)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("neons"));

	// no lambda functions for gcc 4.4 :(
	// call_once(oflag, [](){ NFmiNeonsDBPool::MaxWorkers(MAX_WORKERS); });

	call_once(oflag, &himan::plugin::neons::InitPool, this);

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

void neons::Init()
{
    itsNeonsDB = unique_ptr<NFmiNeonsDB> (NFmiNeonsDBPool::GetConnection());

    itsInit = true;
}

vector<string> neons::Files(const search_options& options)
{

	if (!itsInit)
	{
		Init();
	}

	vector<string> files;

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

bool neons::Save(shared_ptr<const info> resultInfo)
{

	if (!itsInit)
	{
		Init();
	}

	shared_ptr<util> u = dynamic_pointer_cast<util> (plugin_factory::Instance()->Plugin("util"));

	string NeonsFileName = u->MakeNeonsFileName(resultInfo);

	stringstream query;
/*
	query 	<< "SELECT geom_name "
			<< "FROM grid_reg_geom "
			<< "WHERE row_cnt = " << info.nj
			<< " AND col_cnt = " << info.ni
			<< " AND lat_orig = " << info.lat
			<< " AND long_orig = " << info.lon
			<< " AND pas_latitude = " << info.dj
			<< " AND pas_longitude = " << info.di;


	query << "SELECT "
	         << "m.model_name, "
	         << "model_type, "
	         << "type_smt "
	         << "FROM grid_num_model_grib nu, "
	         << "grid_model m, "
	         << "grid_model_name na "
	         << "WHERE nu.model_id = " << info.process
	         << " AND nu.ident_id = " << info.centre
	         << " AND m.flag_mod = 0 "
	         << " AND nu.model_name = na.model_name "
	         << " AND m.model_name = na.model_name";

	query << "SELECT "
	       << "dset_id, "
	         << "table_name, "
	         << "rec_cnt_dset "
	         << "FROM as_grid "
	         << "WHERE "
	         << "model_type = '" << itsModelType << "'"
	         << " AND geom_name = '" << itsGeomName << "'"
	         << " AND dset_name = '" << dset_name << "'"
	         << " AND base_date = '" << info.base_date << "'";

	query  << "INSERT INTO " << itsTableName
	       << " (dset_id, parm_name, lvl_type, lvl1_lvl2, fcst_per, eps_specifier, file_location, file_server) "
	       << "VALUES ("
	       << itsDsetId << ", "
	       << "'" << info.parname << "', "
	       << "'" << info.levname << "', "
	       << info.lvl1_lvl2 << ", "
	       << info.fcst_per << ", "
	       << "'" << info.eps_specifier << "', "
	       << "'" << info.filename << "', "
	       << "'" << outFileHost << "')";
*/
	itsLogger->Info("Saved information on file '" + NeonsFileName + "' to neons");

	return true;
}
