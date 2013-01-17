/**
 * @file grib.cpp
 *
 * @date Nov 20, 2012
 * @author partio
 */

#include "grib.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include "producer.h"

using namespace std;
using namespace himan::plugin;

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

grib::grib()
{

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("grib"));

	itsGrib = shared_ptr<NFmiGrib> (new NFmiGrib());
}

shared_ptr<NFmiGrib> grib::Reader()
{
	return itsGrib;
}

bool grib::ToFile(shared_ptr<info> theInfo, const string& theOutputFile, HPFileType theFileType, bool theActiveOnly)
{

	static long edition = static_cast<long> (theFileType);

	if (theActiveOnly)
	{
		// Write only that data which is currently set at descriptors

		/* Section 0 */

		itsGrib->Message()->Edition(edition);

		if (theFileType == kGRIB1)
		{
			itsGrib->Message()->Table2Version(theInfo->Param().GribTableVersion()) ;
		}
		else if (theFileType == kGRIB2)
		{
			itsGrib->Message()->ParameterDiscipline(theInfo->Param().GribDiscipline()) ;
		}

		/* Section 1 */

		itsGrib->Message()->Centre(theInfo->Producer().Centre());
		itsGrib->Message()->Process(theInfo->Producer().Process());

		if (theFileType == kGRIB1)
		{
			itsGrib->Message()->DataDate(boost::lexical_cast<long> (theInfo->Time().OriginDateTime()->String("%Y%m%d")));
			itsGrib->Message()->DataTime(boost::lexical_cast<long> (theInfo->Time().OriginDateTime()->String("%H%M")));
		}
		else if (theFileType == kGRIB2)
		{
			// Origin time
			itsGrib->Message()->Year(theInfo->Time().OriginDateTime()->String("%Y"));
			itsGrib->Message()->Month(theInfo->Time().OriginDateTime()->String("%m"));
			itsGrib->Message()->Day(theInfo->Time().OriginDateTime()->String("%d"));
			itsGrib->Message()->Hour(theInfo->Time().OriginDateTime()->String("%H"));
			itsGrib->Message()->Minute(theInfo->Time().OriginDateTime()->String("%M"));
			itsGrib->Message()->Second("0");
		}

		itsGrib->Message()->StartStep(theInfo->Time().Step());
		itsGrib->Message()->EndStep(theInfo->Time().Step());

		/* Section 4 */

		if (theFileType == kGRIB1)
		{
			itsGrib->Message()->ParameterNumber(theInfo->Param().GribParameter()) ;
		}
		else if (theFileType == kGRIB2)
		{
			itsGrib->Message()->ParameterCategory(theInfo->Param().GribCategory()) ;
			itsGrib->Message()->ParameterNumber(theInfo->Param().GribParameter()) ;
		}

		himan::point firstGridPoint = theInfo->FirstGridPoint();
		himan::point lastGridPoint = theInfo->LastGridPoint();

		switch (theInfo->Projection())
		{
		case kLatLonProjection:
		{
			long gridType = 0; // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message()->GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message()->GridType(gridType);

			itsGrib->Message()->X0(firstGridPoint.X());
			itsGrib->Message()->Y0(firstGridPoint.Y());
			itsGrib->Message()->X1(lastGridPoint.X());
			itsGrib->Message()->Y1(lastGridPoint.Y());

			itsGrib->Message()->iDirectionIncrement(theInfo->Di());
			itsGrib->Message()->jDirectionIncrement(theInfo->Dj());
			break;
		}
		case kRotatedLatLonProjection:
		{

			long gridType = 10; // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message()->GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message()->GridType(gridType);

			itsGrib->Message()->X0(firstGridPoint.X());
			itsGrib->Message()->Y0(firstGridPoint.Y());
			itsGrib->Message()->X1(lastGridPoint.X());
			itsGrib->Message()->Y1(lastGridPoint.Y());

			itsGrib->Message()->SouthPoleX(theInfo->SouthPole().X());
			itsGrib->Message()->SouthPoleY(theInfo->SouthPole().Y());

			itsGrib->Message()->iDirectionIncrement(theInfo->Di());
			itsGrib->Message()->jDirectionIncrement(theInfo->Dj());

			if (edition == 2)
			{
				gridType = itsGrib->Message()->GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message()->GridType(gridType);

			break;
		}

		case kStereographicProjection:
		{
			long gridType = 5; // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message()->GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message()->GridType(gridType);

			itsGrib->Message()->X0(theInfo->BottomLeft().X());
			itsGrib->Message()->Y0(theInfo->BottomLeft().Y());

			// missing iDirectionIncrementInMeters
			itsGrib->Message()->GridOrientation(theInfo->Orientation());
			break;
		}
		default:
			throw runtime_error(ClassName() + ": invalid projection while writing grib: " + boost::lexical_cast<string> (theInfo->Projection()));
			break;
		}

		itsGrib->Message()->SizeX(theInfo->Ni());
		itsGrib->Message()->SizeY(theInfo->Nj());

		if (edition == 2)
		{
			itsGrib->Message()->TypeOfGeneratingProcess(1); // Forecast
		}

		// Level

		itsGrib->Message()->LevelValue(static_cast<long> (theInfo->Level().Value()));

		// Himan levels equal to grib 1

		if (edition == 1)
		{
			itsGrib->Message()->LevelTypeToAnotherEdition(theInfo->Level().Type(),1);
		}
		else if (edition == 2)
		{
			itsGrib->Message()->LevelType(theInfo->Level().Type());
		}

		itsGrib->Message()->Bitmap(true);

		itsGrib->Message()->Values(theInfo->Data()->Values(), theInfo->Ni() * theInfo->Nj());

		itsGrib->Message()->PackingType("grid_jpeg");
		itsGrib->Message()->Write(theOutputFile);

		itsLogger->Info("Wrote file '" + theOutputFile + "'");
	}

	return true;

}

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool theReadContents)
{

	shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));
	vector<shared_ptr<himan::info>> theInfos;

	itsGrib->Open(theInputFile);

	itsLogger->Debug("Reading file '" + theInputFile + "'");

	int foundMessageNo = 0;

	long fmiProducer = options.configuration->SourceProducer();

	map<string, string> producermap = n->ProducerInfo(fmiProducer);

	if (producermap.empty())
	{
		throw runtime_error(ClassName() + ": Process and centre information for producer " + boost::lexical_cast<string> (fmiProducer) + " not found from neons");
	}

	himan::producer sourceProducer(fmiProducer,
			boost::lexical_cast<long> (producermap["centre"]),
			boost::lexical_cast<long> (producermap["process"]),
			producermap["name"]);

	while (itsGrib->NextMessage())
	{

		foundMessageNo++;

		/*
		 * One grib file may contain many grib messages. Loop though all messages
		 * and get all that match our search options.
		 *
		 */

		//<!todo Should we actually return all matching messages or only the first one

		long centre = itsGrib->Message()->Centre();
		long process = itsGrib->Message()->Process();

		if (sourceProducer.Process() != process || sourceProducer.Centre() != centre)
		{
			itsLogger->Trace("centre/process do not match: " + boost::lexical_cast<string> (sourceProducer.Process()) + " vs " + boost::lexical_cast<string> (process));
			itsLogger->Trace("centre/process do not match: " + boost::lexical_cast<string> (sourceProducer.Centre()) + " vs " + boost::lexical_cast<string> (centre));
			//continue;
		}

		param p;

		long number = itsGrib->Message()->ParameterNumber();

		if (itsGrib->Message()->Edition() == 1)
		{
			long no_vers = itsGrib->Message()->Table2Version();

			p.Name(n->GribParameterName(number, no_vers));		   
			p.GribParameter(number);
			p.GribTableVersion(no_vers);

		}
		else
		{

			long category = itsGrib->Message()->ParameterCategory();
			long discipline = itsGrib->Message()->ParameterDiscipline();
			long producer = options.configuration->SourceProducer();
			map<std::string, std::string> producermap;
			
			producermap = n->ProducerInfo(producer);
			
			if (producermap.empty())
			{
				throw runtime_error("Unknown producer: " + boost::lexical_cast<string> (producer));
			}

			long process = boost::lexical_cast<long>(producermap["process"]);
			p.Name(n->GribParameterName(number, category, discipline, process));

			p.GribParameter(number);
			p.GribDiscipline(discipline);
			p.GribCategory(category);

			if (p.Name() == "T-C" && producermap["centre"] == "7")
			{
				p.Name("T-K");
			}
		}

		if (itsGrib->Message()->ParameterUnit() == "K")
		{
		   	p.Unit(kK);
		}
		else if (itsGrib->Message()->ParameterUnit() == "Pa s-1")
		{
		   	p.Unit(kPas);
		}
		else
		{
			itsLogger->Warning("Unable to determine himan parameter unit for grib unit " + itsGrib->Message()->ParameterUnit());
		}

		if (p != options.param)
		{
			itsLogger->Trace("Parameter does not match: " + options.param.Name() + " vs " + p.Name());
			continue;
		}

		string dataDate = boost::lexical_cast<string> (itsGrib->Message()->DataDate());

		/*
		 * dataTime is HH24MM in long datatype.
		 *
		 * So, for example analysistime 00 is 0, and 06 is 600.
		 *
		 */

		long dt = itsGrib->Message()->DataTime();
		long step = itsGrib->Message()->EndStep();
		string dataTime = boost::lexical_cast<string> (dt);

		if (dt < 1000)
		{
			dataTime = "0" + dataTime;
		}

		if (itsGrib->Message()->Edition() == 2) 
		{
			step = itsGrib->Message()->ForecastTime();
		}
		
		string originDateTimeStr = dataDate + dataTime;

		raw_time originDateTime (originDateTimeStr, "%Y%m%d%H%M");

		forecast_time t (originDateTime, originDateTime);

		t.ValidDateTime()->Adjust("hours", static_cast<int> (step));

		if (t != options.time)
		{
			itsLogger->Trace("Times do not match");
			itsLogger->Trace("OriginDateTime: " + options.time.OriginDateTime()->String() + " (requested) vs " + t.OriginDateTime()->String() + " (found)");
			itsLogger->Trace("ValidDateTime: " + options.time.ValidDateTime()->String() + " (requested) vs " + t.ValidDateTime()->String() + " (found)");
			continue;
		}

		long gribLevel = itsGrib->Message()->NormalizedLevelType();

		himan::HPLevelType levelType;

		switch (gribLevel)
		{
		case 1:
			levelType = himan::kGround;
			break;

		case 100:
			levelType = himan::kPressure;
			break;

		case 102:
			levelType = himan::kMeanSea;
			break;

		case 105:
			levelType = himan::kHeight;
			break;

		case 109:
			levelType = himan::kHybrid;
			break;

		default:
			throw runtime_error(ClassName() + ": Unsupported level type: " + boost::lexical_cast<string> (gribLevel));

		}

		level l (levelType, static_cast<float> (itsGrib->Message()->LevelValue()));

		if (l != options.level)
		{
			itsLogger->Trace("Level does not match");
			continue;
		}

		// END VALIDATION OF SEARCH PARAMETERS

		shared_ptr<info> newInfo (new info());

		producer prod(itsGrib->Message()->Centre(), process);

		newInfo->Producer(prod);

		vector<param> theParams;

		theParams.push_back(p);

		newInfo->Params(theParams);

		vector<forecast_time> theTimes;

		theTimes.push_back(t);

		newInfo->Times(theTimes);

		vector<level> theLevels;

		theLevels.push_back(l);

		newInfo->Levels(theLevels);


		/*
		 * Get area information from grib.
		 */

		switch (itsGrib->Message()->NormalizedGridType())
		{
		case 0:
			newInfo->Projection(kLatLonProjection);
			break;

		case 10:
			newInfo->Projection(kRotatedLatLonProjection);
			newInfo->SouthPole(himan::point(itsGrib->Message()->SouthPoleX(), itsGrib->Message()->SouthPoleY()));
			break;

		default:
			throw runtime_error(ClassName() + ": Unsupported projection: " + boost::lexical_cast<string> (itsGrib->Message()->NormalizedGridType()));
			break;

		}

		size_t ni = itsGrib->Message()->SizeX();
		size_t nj = itsGrib->Message()->SizeY();

		bool iNegative = itsGrib->Message()->IScansNegatively();
		bool jPositive = itsGrib->Message()->JScansPositively();

		HPScanningMode m = kUnknownScanningMode;

		if (!iNegative && !jPositive)
		{
			m = kTopLeft;
		}
		else if (iNegative && !jPositive)
		{
			m = kTopRight;
		}
		else if (iNegative && jPositive)
		{
			m = kBottomRight;
		}
		else if (!iNegative && jPositive)
		{
			m = kBottomLeft;
		}
		else
		{
			throw runtime_error("WHAT?");
		}

		newInfo->ScanningMode(m);

		newInfo->SetCoordinatesFromFirstGridPoint(himan::point(itsGrib->Message()->X0(), itsGrib->Message()->Y0()), ni, nj, itsGrib->Message()->iDirectionIncrement(),itsGrib->Message()->jDirectionIncrement());

		/*
		 * Read data from grib *
		 */

		size_t len = 0;
		double* d = 0;

		if (theReadContents)
		{
			len = itsGrib->Message()->ValuesLength();

			d = itsGrib->Message()->Values();
		}

		newInfo->Create();

		// Set descriptors

		newInfo->Param(p);
		newInfo->Time(t);
		newInfo->Level(l);

		shared_ptr<d_matrix_t> dm = shared_ptr<d_matrix_t> (new d_matrix_t(ni, nj));

		dm->Set(d, len);

		newInfo->Data(dm);

		theInfos.push_back(newInfo);

		if (d)
		{
			free(d);
		}

		break ; // We found what we were looking for
	}

	if (theInfos.size())
	{
		// This will be broken when/if we return multiple infos from this function
		itsLogger->Trace("Data found from message " + boost::lexical_cast<string> (foundMessageNo) + "/" + boost::lexical_cast<string> (itsGrib->MessageCount()));
	}

	return theInfos;
}
