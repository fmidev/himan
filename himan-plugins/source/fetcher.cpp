/*
 * fetcher.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: partio
 */

#include "fetcher.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <fstream>
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "grib.h"
#include "neons.h"
#include "param.h"
#include "cache.h"
#include "util.h"
#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan::plugin;
using namespace std;

fetcher::fetcher()
{
	itsLogger = logger_factory::Instance()->GetLog("fetcher");
}

shared_ptr<himan::info> fetcher::Fetch(shared_ptr<const configuration> theConfiguration,
                                       shared_ptr<const forecast_time> theValidTime,
                                       shared_ptr<const level> theLevel,
                                       shared_ptr<const param> theParam)
{

	// 1. Fetch data from cache

	// FromCache()

	// 2. Fetch data from auxiliary files specified at command line

	// while (loop_aux_files) {
	//   FromFile()
	// }

	// 3. Fetch data from Neons

	// GetFileNameFromNeons()


	// FromFile()



	string theFile;

	if (theParam->Name() == "T-K")
	{
		theFile = "T_" + boost::lexical_cast<string> (theLevel->Type()) +
		          "_" + boost::lexical_cast<string> (theLevel->Value()) +
		          "_" + boost::lexical_cast<string> (theValidTime->Step()) +
		          ".grib2";
	}

	else if (theParam->Name() == "P-HPA")
	{
		theFile = "P.grib2";
	}

	else if (theParam->Name() == "VV-PAS")
	{
		theFile = "VV-PAS_" + boost::lexical_cast<string> (theLevel->Type()) +
		          "_" + boost::lexical_cast<string> (theLevel->Value()) +
		          "_" + boost::lexical_cast<string> (theValidTime->Step()) +
		          ".grib2";
	}
	else
	{
		throw runtime_error("Fetching of param " + theParam->Name() + " not supported yet");
	}

	const search_options opts { theValidTime, theParam, theLevel, theConfiguration } ;

	vector<shared_ptr<info>> theInfos = FromFile(theFile, opts, true);

	if (theInfos.size() == 0)
	{
		throw runtime_error(ClassName() + ": No valid data found from file '" + theFile + "'");
	}

	/*
	 *  Safeguard; later in the code we do not check whether the data requested
	 *  was actually what was requested.
	 */
	/*
		assert(theConfiguration.SourceProducer() == theInfos[0]->Producer());

		assert(*(theInfos[0]->Level()) == theLevel);

		assert(*(theInfos[0]->Time()) == theValidTime);

		assert(*(theInfos[0]->Param()) == theParam);
	*/
	return theInfos[0];

}

/*
 * FromFile()
 *
 * Get data and metadata from a file. Returns a vector of infos, mainly because one grib file can
 * contain many grib messages. If read file is querydata, the vector size is always one (or zero if the
 * read fails).
 */

vector<shared_ptr<himan::info>> fetcher::FromFile(const string& theInputFile, const search_options& options, bool theReadContents)
{

	vector<shared_ptr<himan::info>> theInfos ;

	shared_ptr<util> theUtil = dynamic_pointer_cast<util> (plugin_factory::Instance()->Plugin("util"));

	switch (theUtil->FileType(theInputFile))
	{

		case kGRIB:
		case kGRIB1:
		case kGRIB2:
			{

				theInfos = FromGrib(theInputFile, options, theReadContents);
				break;

			}

		case kQueryData:
			{
				// Create querydata instance
				itsLogger->Info("File is QueryData");

				theInfos = FromQueryData(theInputFile, options, theReadContents);
				break;
			}
		case kNetCDF:
			// Create NetCDF instance
			cout << "File is NetCDF" << endl;
			break;

		default:
			// Unknown file type, cannot proceed
			throw runtime_error("Input file is neither GRID, NetCDF nor QueryData");
			break;
	}

	return theInfos;
}

/*
 * FromGrib()
 *
 * Read data and metadata from grib file. This function does not create any newbase-
 * specific parameter descriptors, and does *not* connect to neons to get external
 * metadata.
 */

vector<shared_ptr<himan::info> > fetcher::FromGrib(const string& theInputFile, const search_options& options, bool theReadContents)
{

	shared_ptr<grib> g = dynamic_pointer_cast<grib> (plugin_factory::Instance()->Plugin("grib"));

	vector<shared_ptr<info> > theInfos;

	theInfos = g->FromFile(theInputFile, options, theReadContents);

	return theInfos;
}

vector<shared_ptr<himan::info>> fetcher::FromQueryData(const string& theInputFile, const search_options& options, bool theReadContents)
{

	itsLogger->Debug("Reading metadata from file '" + theInputFile + "'");

	vector<shared_ptr<info>> theInfos;

	// Create querydata instance

	shared_ptr<querydata> q = dynamic_pointer_cast<querydata> (plugin_factory::Instance()->Plugin("querydata"));

	shared_ptr<info> i = q->FromFile(theInputFile, options, theReadContents);

	theInfos.push_back(i);

	return theInfos;
}
