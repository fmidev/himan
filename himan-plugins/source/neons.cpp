/**
 * @file neons.cpp
 *
 */

#include "neons.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <sstream>
#include <thread>

using namespace std;
using namespace himan::plugin;

const int MAX_WORKERS = 16;
static std::once_flag oflag;

void neons::Init()
{
	if (!itsInit)
	{
		try
		{
			call_once(oflag, [&]() {
				NFmiNeonsDBPool::Instance()->ReadWriteTransaction(true);
				NFmiNeonsDBPool::Instance()->Username("wetodb");
				NFmiNeonsDBPool::Instance()->Password(util::GetEnv("NEONS_WETODB_PASSWORD"));

				if (NFmiNeonsDBPool::Instance()->MaxWorkers() < MAX_WORKERS)
				{
					NFmiNeonsDBPool::Instance()->MaxWorkers(MAX_WORKERS);
				}
			});

			itsNeonsDB = std::unique_ptr<NFmiNeonsDB>(NFmiNeonsDBPool::Instance()->GetConnection());
		}
		catch (int e)
		{
			itsLogger.Fatal("Failed to get connection");
			abort();
		}

		itsInit = true;
	}
}

neons::neons() : itsInit(false), itsNeonsDB()
{
	itsLogger = logger("neons");
}

void neons::PoolMaxWorkers(int maxWorkers) { NFmiNeonsDBPool::Instance()->MaxWorkers(maxWorkers); }
vector<string> neons::Files(search_options& options)
{
	Init();

	vector<string> files;

	string analtime = options.time.OriginDateTime().String("%Y%m%d%H%M%S");
	string levelvalue = boost::lexical_cast<string>(options.level.Value());

	string ref_prod = options.prod.Name();
	long no_vers = options.prod.TableVersion();

	string level_name = HPLevelTypeToString.at(options.level.Type());

	vector<vector<string>> gridgeoms;
	vector<string> sourceGeoms = options.configuration->SourceGeomNames();

	if (sourceGeoms.empty())
	{
		// Get all geometries
		gridgeoms = itsNeonsDB->GetGridGeoms(ref_prod, analtime);
	}
	else
	{
		for (size_t i = 0; i < sourceGeoms.size(); i++)
		{
			vector<vector<string>> geoms = itsNeonsDB->GetGridGeoms(ref_prod, analtime, sourceGeoms[i]);
			gridgeoms.insert(gridgeoms.end(), geoms.begin(), geoms.end());
		}
	}

	if (gridgeoms.empty())
	{
		// No geometries found, fetcher checks this
		return files;
	}
	/*
	string neonsForecastType = "";

	switch (options.ftype.Type())
	{
	    default:
	    case kDeterministic:
	    case kAnalysis:
	        neonsForecastType = "0";
	        break;
	    case kEpsControl:
	        neonsForecastType = "3";
	        break;
	    case kEpsPerturbation:
	        neonsForecastType = "4_" + boost::lexical_cast<string> (options.ftype.Value());
	        break;
	}
	*/
	for (size_t i = 0; i < gridgeoms.size(); i++)
	{
		string tablename = gridgeoms[i][1];
		string dset = gridgeoms[i][2];

		/// @todo GFS (or in fact codetable 2) has wrong temperature parameter defined

		string parm_name = options.param.Name();

		if (parm_name == "T-K" && no_vers == 2)
		{
			parm_name = "T-C";
		}

		string query =
		    "SELECT parm_name, lvl_type, lvl1_lvl2, fcst_per, file_location, file_server "
		    "FROM " +
		    tablename +
		    " "
		    "WHERE dset_id = " +
		    dset +
		    " "
		    "AND parm_name = upper('" +
		    parm_name +
		    "') "
		    "AND lvl_type = upper('" +
		    level_name +
		    "') "
		    "AND lvl1_lvl2 = " +
		    levelvalue +
		    " "
		    "AND fcst_per = " +
		    boost::lexical_cast<string>(options.time.Step()) +
		    " "
		    // eps-specifier commented out until all data is loaded with grid_to_neons
		    //				   "AND eps_specifier = '"+neonsForecastType+"' "
		    "ORDER BY dset_id, fcst_per, lvl_type, lvl1_lvl2";

		itsNeonsDB->Query(query);

		vector<string> values = itsNeonsDB->FetchRow();

		if (values.empty())
		{
			continue;
		}

		itsLogger.Trace("Found data for parameter " + parm_name + " from neons geometry " + gridgeoms[i][0]);

		files.push_back(values[4]);

		break;  // discontinue loop on first positive match
	}

	return files;
}

bool neons::Save(const info& resultInfo, const string& theFileName)
{
	Init();

	stringstream query;

	if (resultInfo.Grid()->Class() != kRegularGrid)
	{
		itsLogger.Error("Only grid data can be stored to neons for now");
		return false;
	}

	/*
	 * 1. Get grid information
	 * 2. Get model information
	 * 3. Get data set information (ie model run)
	 * 4. Insert or update
	 */

	himan::point firstGridPoint = resultInfo.Grid()->FirstPoint();

	query << "SELECT geom_name "
	      << "FROM grid_reg_geom "
	      << "WHERE row_cnt = " << resultInfo.Grid()->Nj() << " AND col_cnt = " << resultInfo.Grid()->Ni()
	      << " AND lat_orig = " << (firstGridPoint.Y() * 1e3) << " AND long_orig = " << (firstGridPoint.X() * 1e3);

	itsNeonsDB->Query(query.str());

	vector<string> row;

	row = itsNeonsDB->FetchRow();

	if (row.empty())
	{
		itsLogger.Warning("Grid geometry not found from neons");
		return false;
	}

	string geom_name = row[0];

	query.str("");

	query << "SELECT "
	      << "nu.model_id AS process, "
	      << "nu.ident_id AS centre, "
	      << "m.model_name, "
	      << "m.model_type, "
	      << "type_smt "
	      << "FROM "
	      << "grid_num_model_grib nu, "
	      << "grid_model m, "
	      << "grid_model_name na, "
	      << "fmi_producers f "
	      << "WHERE f.producer_id = " << resultInfo.Producer().Id() << " AND m.model_type = f.ref_prod "
	      << " AND nu.model_name = m.model_name "
	      << " AND m.flag_mod = 0 "
	      << " AND nu.model_name = na.model_name "
	      << " AND m.model_name = na.model_name ";

	itsNeonsDB->Query(query.str());

	row = itsNeonsDB->FetchRow();

	if (row.empty())
	{
		itsLogger.Warning("Producer definition not found from neons (id: " +
						  boost::lexical_cast<string>(resultInfo.Producer().Id()) + ")");
		return false;
	}

	string model_type = row[3];

	query.str("");

	query << "SELECT "
	      << "dset_id, "
	      << "table_name, "
	      << "rec_cnt_dset "
	      << "FROM as_grid "
	      << "WHERE "
	      << "model_type = '" << model_type << "'"
	      << " AND geom_name = '" << geom_name << "'"
	      << " AND dset_name = 'AF'"
	      << " AND base_date = '" << resultInfo.Time().OriginDateTime().String("%Y%m%d%H%M") << "'";

	itsNeonsDB->Query(query.str());

	row = itsNeonsDB->FetchRow();

	if (row.empty())
	{
		itsLogger.Warning("Data set definition not found from neons");
		return false;
	}

	string table_name = row[1];
	string dset_id = row[0];

	string eps_specifier = "0";

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
		itsLogger.Error("Error code: " + boost::lexical_cast<string>(e));
		itsLogger.Error("Query: " + query.str());
		itsNeonsDB->Rollback();
		return false;
	}

	query.str("");

	string host = "undetermined host";

	char* hostname = getenv("HOSTNAME");

	if (hostname != NULL)
	{
		host = string(hostname);
	}

	/*
	 * We have our own error loggings for unique key violations
	 */

	itsNeonsDB->Verbose(false);

	query << "INSERT INTO " << table_name
	      << " (dset_id, parm_name, lvl_type, lvl1_lvl2, fcst_per, eps_specifier, file_location, file_server) "
	      << "VALUES (" << dset_id << ", "
	      << "'" << resultInfo.Param().Name() << "', "
	      << "upper('" << HPLevelTypeToString.at(resultInfo.Level().Type()) << "'), " << resultInfo.Level().Value()
	      << ", " << resultInfo.Time().Step() << ", "
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
		if (e == 1)
		{
			// unique key violation

			try
			{
				/*
				 * Neons table definition has invalid primary key definition:
				 *
				 * file_location and file_server are a part of the primary key meaning that
				 * a table can have multiple versions of a single file from multiple servers.
				 * This is unfortunate and to bypass it we must first delete all versions
				 * of the file and then INSERT again.
				 */

				query.str("");

				query << "DELETE FROM " << table_name << " WHERE "
				      << "dset_id = " << dset_id << " AND parm_name = '" << resultInfo.Param().Name() << "'"
				      << " AND lvl_type = upper('" << HPLevelTypeToString.at(resultInfo.Level().Type()) << "')"
				      << " AND lvl1_lvl2 = " << resultInfo.Level().Value()
				      << " AND fcst_per = " << resultInfo.Time().Step();

				itsNeonsDB->Execute(query.str());

				query.str("");

				query << "INSERT INTO " << table_name << " (dset_id, parm_name, lvl_type, lvl1_lvl2, fcst_per, "
				                                         "eps_specifier, file_location, file_server) "
				      << "VALUES (" << dset_id << ", "
				      << "'" << resultInfo.Param().Name() << "', "
				      << "upper('" << HPLevelTypeToString.at(resultInfo.Level().Type()) << "'), "
				      << resultInfo.Level().Value() << ", " << resultInfo.Time().Step() << ", "
				      << "'" << eps_specifier << "', "
				      << "'" << theFileName << "', "
				      << "'" << host << "')";

				itsNeonsDB->Execute(query.str());

				itsNeonsDB->Commit();
			}
			catch (int e)
			{
				itsLogger.Fatal("Error code: " + boost::lexical_cast<string>(e));
				itsLogger.Fatal("Query: " + query.str());

				itsNeonsDB->Rollback();

				abort();
			}
		}
		else
		{
			itsLogger.Error("Error code: " + boost::lexical_cast<string>(e));
			itsLogger.Error("Query: " + query.str());

			itsNeonsDB->Rollback();

			return false;
		}
	}

	itsLogger.Trace("Saved information on file '" + theFileName + "' to neons");

	return true;
}

string neons::GribParameterName(long fmiParameterId, long codeTableVersion, long timeRangeIndicator, long levelType)
{
	Init();

	string paramName = itsNeonsDB->GetGridParameterName(fmiParameterId, codeTableVersion, codeTableVersion,
	                                                    timeRangeIndicator, levelType);
	return paramName;
}

string neons::GribParameterName(long fmiParameterId, long category, long discipline, long producer, long levelType)
{
	Init();

	string paramName = itsNeonsDB->GetGridParameterNameForGrib2(fmiParameterId, category, discipline, producer);
	return paramName;
}

string neons::ProducerMetaData(long producerId, const string& attribute) const
{
	string ret;

	if (attribute == "last hybrid level number")
	{
		switch (producerId)
		{
			case 1:
			case 199:
			case 210:
			case 230:
				ret = "65";
				break;

			case 131:
			case 240:
				ret = "137";
				break;

			default:
				throw runtime_error(ClassName() + ": Producer not supported");
				break;
		}
	}
	else if (attribute == "first hybrid level number")
	{
		switch (producerId)
		{
			case 1:
			case 199:
			case 210:
			case 230:
				ret = "1";
				break;

			case 131:
			case 240:
				ret = "24";
				break;

			default:
				throw runtime_error(ClassName() + ": Producer not supported");
				break;
		}
	}
	else
	{
		throw runtime_error(ClassName() + ": Attribute not recognized");
	}

	return ret;

	// In the future maybe something like this:

	// Init();

	// string query = "SELECT value FROM producers_eav WHERE producer_id = " + boost::lexical_cast<string> (producerId)
	// + " AND attribute = '" + attribute + "'";
}
