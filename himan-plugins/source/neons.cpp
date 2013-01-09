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
#include "util.h"

using namespace std;
using namespace himan::plugin;

const int MAX_WORKERS = 16;
once_flag oflag;

neons::neons() : itsInit(false), itsNeonsDB()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("neons"));

    // no lambda functions for gcc 4.4 :(
    // call_once(oflag, [](){ NFmiNeonsDBPool::MaxWorkers(MAX_WORKERS); });

    call_once(oflag, &himan::plugin::neons::InitPool, this);
}

void neons::InitPool()
{
    NFmiNeonsDBPool::Instance()->MaxWorkers(MAX_WORKERS);
}

void neons::Init()
{
    if (!itsInit)
    {
        try
        {
            itsNeonsDB = unique_ptr<NFmiNeonsDB> (NFmiNeonsDBPool::Instance()->GetConnection());
        }
        catch (int e)
        {
            itsLogger->Fatal("Failed to get connection");
            exit(1);
        }

        itsInit = true;
    }
}

vector<string> neons::Files(const search_options& options)
{

    Init();

    vector<string> files;

    // const int kFMICodeTableVer = 204;

    string analtime = options.configuration->Info()->OriginDateTime().String("%Y%m%d%H%M%S");
    string levelvalue = boost::lexical_cast<string> (options.level.Value());

    map<string, string> producerInfo = itsNeonsDB->GetProducerDefinition(options.configuration->SourceProducer());

    if (producerInfo.empty())
    {
        itsLogger->Warning("Producer definition not found for producer " + boost::lexical_cast<string> (options.configuration->SourceProducer()));
        return files;
    }

    string ref_prod = producerInfo["ref_prod"];
    string proddef = producerInfo["producer_id"];
    string no_vers = producerInfo["no_vers"];

    //string param_name = itsNeonsDB->GetGridParameterName(options.param.UnivId(), kFMICodeTableVer, boost::lexical_cast<long>(no_vers));

    //string level_name = itsNeonsDB->GetGridLevelName(options.param.UnivId(), options.level.Type(), kFMICodeTableVer, boost::lexical_cast<long>(no_vers));

    string level_name = options.level.Name();

    vector<vector<string> > gridgeoms = itsNeonsDB->GetGridGeoms(ref_prod, analtime);

    if (gridgeoms.size() == 0)
    {
        itsLogger->Warning("No data found for given search options");
        return files;
    }

    for (size_t i = 0; i < gridgeoms.size(); i++)
    {
        string tablename = gridgeoms[i][1];
        string dset = gridgeoms[i][2];

        /// @todo GFS (or in fact codetable 2) has wrong temperature parameter defined

        string parm_name = options.param.Name();

        string query = "SELECT parm_name, lvl_type, lvl1_lvl2, fcst_per, file_location, file_server "
                       "FROM "+tablename+" "
                       "WHERE dset_id = "+dset+" "
                       "AND parm_name = '"+parm_name+"' "
                       "AND lvl_type = '"+level_name+"' "
                       "AND lvl1_lvl2 = " +levelvalue+" "
                       "AND fcst_per = "+boost::lexical_cast<string> (options.time.Step())+" "
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

bool neons::Save(shared_ptr<const info> resultInfo, const string& theFileName)
{
    Init();

    stringstream query;

    /*
     * 1. Get grid information
     * 2. Get model information
     * 3. Get data set information (ie model run)
     * 4. Insert or update
     */

    long lat_orig, lon_orig;

    if (resultInfo->ScanningMode() == kTopLeft)
    {
        lat_orig = static_cast<long> (resultInfo->TopRightLatitude()*1e3);
        lon_orig = static_cast<long> (resultInfo->BottomLeftLongitude() * 1e3);
    }
    else
    {
        throw runtime_error(ClassName() + ": unsupported scanning mode: " + boost::lexical_cast<string> (resultInfo->ScanningMode()));
    }


    /*
     * pas_latitude and pas_longitude cannot be checked programmatically
     * since f.ex. in the case for GFS in neons we have value 500 and
     * by calculating we have value 498. But not check these columns should
     * not matter as long as row_cnt, col_cnt, lat_orig and lon_orig match
     * (since pas_latitude and pas_longitude are derived from these anyway)
     */

    query 	<< "SELECT geom_name "
            << "FROM grid_reg_geom "
            << "WHERE row_cnt = " << resultInfo->Nj()
            << " AND col_cnt = " << resultInfo->Ni()
            << " AND lat_orig = " << lat_orig
            << " AND long_orig = " << lon_orig;
//			<< " AND pas_latitude = " << static_cast<long> (resultInfo->Dj() * 1e3)
//			<< " AND pas_longitude = " << static_cast<long> (resultInfo->Di() * 1e3);

    itsNeonsDB->Query(query.str());

    vector<string> row;

    row = itsNeonsDB->FetchRow();

    if (row.empty())
    {
        itsLogger->Warning("Grid geometry not found from neons");
        return false;
    }

    string geom_name = row[0];

    query.str("");

    query 	<< "SELECT "
            << "nu.model_id AS process, "
            << "nu.ident_id AS centre, "
            << "m.model_name, "
            << "model_type, "
            << "type_smt "
            << "FROM "
            << "grid_num_model_grib nu, "
            << "grid_model m, "
            << "grid_model_name na, "
            << "fmi_producers f "
            << "WHERE f.producer_id = " << resultInfo->Producer().Id()
            << " AND nu.model_name = f.ref_prod "
            << " AND m.flag_mod = 0 "
            << " AND nu.model_name = na.model_name "
            << " AND m.model_name = na.model_name ";

    itsNeonsDB->Query(query.str());

    row = itsNeonsDB->FetchRow();

    if (row.empty())
    {
        itsLogger->Warning("Producer definition not found from neons (id: " + boost::lexical_cast<string> (resultInfo->Producer().Id()) + ")");
        return false;
    }

    string process = row[0];
    string centre = row[1];
    string model_name = row[2];
    string model_type = row[3];

    /*
    	query 	<< "SELECT "
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

    */

    query.str("");

    query	<< "SELECT "
            << "dset_id, "
            << "table_name, "
            << "rec_cnt_dset "
            << "FROM as_grid "
            << "WHERE "
            << "model_type = '" << model_type << "'"
            << " AND geom_name = '" << geom_name << "'"
            << " AND dset_name = 'AF'"
            << " AND base_date = '" << resultInfo->OriginDateTime().String("%Y%m%d%H%M") << "'";

    itsNeonsDB->Query(query.str());

    row = itsNeonsDB->FetchRow();

    if (row.empty())
    {
        itsLogger->Warning("Data set definition not found from neons");
        return false;
    }

    string table_name = row[1];
    string dset_id = row[0];

    string host = "himan_test_host";
    string eps_specifier = "0";

    query.str("");

    query << "LOCK TABLE as_grid IN SHARE MODE";

    itsNeonsDB->Execute(query.str());

    query.str("");

    query << "UPDATE as_grid "
           << "SET rec_cnt_dset = "
           << "rec_cnt_dset + 1, "
           << "date_maj_dset = sysdate "
           << "WHERE dset_id = " << dset_id;

    try
    {
    	itsNeonsDB->Execute(query.str());
    }
    catch (int e)
    {
       itsLogger->Error("Error code: " + boost::lexical_cast<string> (e));
       itsLogger->Error("Query: " + query.str());
   	   itsNeonsDB->Rollback();
   	   return false;
    }

    query.str("");

    query  << "INSERT INTO " << table_name
           << " (dset_id, parm_name, lvl_type, lvl1_lvl2, fcst_per, eps_specifier, file_location, file_server) "
           << "VALUES ("
           << dset_id << ", "
           << "'" << resultInfo->Param().Name() << "', "
           << "'" << resultInfo->Level().Name() << "', "
           << resultInfo->Level().Value() << ", "
           << resultInfo->Time().Step() << ", "
           << "'" << eps_specifier << "', "
           << "'" << theFileName << "', "
           << "'" << host << "')";


    try
    {
    	itsNeonsDB->Execute(query.str());
    	itsNeonsDB->Commit();
    }
    catch (int e)
    {
        itsLogger->Error("Error code: " + boost::lexical_cast<string> (e));
        itsLogger->Error("Query: " + query.str());
        itsNeonsDB->Rollback();
        return false;
    }

    itsLogger->Info("Saved information on file '" + theFileName + "' to neons");

    return true;
}

map<string,string> neons::ProducerInfo(long fmiProducerId)
{

    Init();

    string query = "SELECT n.model_id, n.ident_id, n.model_name FROM grid_num_model_grib n "
                   "WHERE n.model_name = (SELECT ref_prod FROM fmi_producers WHERE producer_id = "
                   + boost::lexical_cast<string> (fmiProducerId) + ")";

    itsNeonsDB->Query(query);

    vector<string> row = itsNeonsDB->FetchRow();

    map<string,string> ret;

    if (row.empty())
    {
        return ret;
    }

    ret["process"] = row[0];
    ret["centre"] = row[1];
    ret["name"] = row[2];

    return ret;
}

std::string neons::LatestTime(const producer& prod)
{

	Init();

	string query = "SELECT table_name, dset_id, to_char(base_date,'YYYYMMDDHH24MI') "
					"FROM as_grid "
					"WHERE model_type = '" +prod.Name() + "' "
			//		"AND base_date > TO_DATE(SYSDATE - " +hours_in_interest +"/24 - " +offset + "/24) "
					"AND rec_cnt_dset > 0 "
					"ORDER BY base_date DESC";

	itsNeonsDB->Query(query);

	vector<string> row = itsNeonsDB->FetchRow();

	if (row.size() == 0)
	{
		itsLogger->Error("Unable to find latest model run for producer '" + prod.Name() + "'");
		return "";
	}

	return row[2];
}

NFmiNeonsDB& neons::NeonsDB()
{
	Init();
	return *itsNeonsDB.get();
}

/// Gets grib parameter name based on number and code table
/**
 *  \par FmiParameterId - parameter number
 *  \par CodeTableVersion  - code table number
 */
std::string neons::GribParameterName(const long fmiParameterId, const long codeTableVersion) 
{
    
    Init();
    
    std::string paramName = itsNeonsDB->GetGridParameterName(fmiParameterId, codeTableVersion, 204);
    return paramName;   
}
