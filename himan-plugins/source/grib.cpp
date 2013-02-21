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
#include "util.h"
#include "grid.h"

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

bool grib::ToFile(shared_ptr<info> info, const string& outputFile, HPFileType fileType, bool theActiveOnly)
{

	if (theActiveOnly)
	{
		// Write only that data which is currently set at descriptors

		WriteGrib(info, outputFile, fileType);
	}

	else
	{
		info->ResetTime();

        while (info->NextTime())
		{
        	info->ResetLevel();

			while (info->NextLevel())
			{
				info->ResetParam();

				while (info->NextParam())
				{
					if (!WriteGrib(info, outputFile, fileType, true))
					{
						itsLogger->Error("Error writing grib to file");
					}
				}
			}
		}
	}

	return true;

}

bool grib::WriteGrib(shared_ptr<const info> info, const string& outputFile, HPFileType fileType, bool appendToFile)
{
	const static long edition = static_cast<long> (fileType);

	/* Section 0 */

	itsGrib->Message()->Edition(edition);

	shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

	long no_vers = 0; // We might need this later on

	if (info->Producer().Centre() == kHPMissingInt)
	{
		map<string, string> producermap = n->NeonsDB().GetGridModelDefinition(info->Producer().Id()); //n->ProducerInfo(info->Producer().Id());

		itsGrib->Message()->Centre(boost::lexical_cast<long> (producermap["ident_id"]));
		itsGrib->Message()->Process(boost::lexical_cast<long> (producermap["model_id"]));

		no_vers = boost::lexical_cast<long> (producermap["no_vers"]);
	}
	else
	{
		itsGrib->Message()->Centre(info->Producer().Centre());
		itsGrib->Message()->Process(info->Producer().Process());
	}

	/*
	 * For grib1, get param_id from neons since its dependant on the table2version
	 *
	 * For grib2, assume the plugin has set the correct numbers since they are "static".
	 */

	if (edition == 1)
	{
		if (info->Producer().TableVersion() != kHPMissingInt)
		{
			no_vers = info->Producer().TableVersion();
		}
		else if (no_vers == 0)
		{
			map<string, string> producermap = n->NeonsDB().GetGridModelDefinition(info->Producer().Id()); //n->ProducerInfo(info->Producer().Id());
			no_vers = boost::lexical_cast<long> (producermap["no_vers"]);
		}

		long parm_id = n->NeonsDB().GetGridParameterId(no_vers, info->Param().Name());
		itsGrib->Message()->ParameterNumber(parm_id);
		itsGrib->Message()->Table2Version(no_vers);
	}
	else if (edition == 2)
	{
		itsGrib->Message()->ParameterNumber(info->Param().GribParameter());
		itsGrib->Message()->ParameterCategory(info->Param().GribCategory());
		itsGrib->Message()->ParameterDiscipline(info->Param().GribDiscipline()) ;
	}

	itsGrib->Message()->DataDate(boost::lexical_cast<long> (info->Time().OriginDateTime()->String("%Y%m%d")));
	itsGrib->Message()->DataTime(boost::lexical_cast<long> (info->Time().OriginDateTime()->String("%H%M")));

	itsGrib->Message()->StartStep(info->Time().Step());
	itsGrib->Message()->EndStep(info->Time().Step());

	himan::point firstGridPoint = info->Grid()->FirstGridPoint();
	himan::point lastGridPoint = info->Grid()->LastGridPoint();

	switch (info->Grid()->Projection())
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

		itsGrib->Message()->iDirectionIncrement(info->Di());
		itsGrib->Message()->jDirectionIncrement(info->Dj());
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

		itsGrib->Message()->SouthPoleX(info->Grid()->SouthPole().X());
		itsGrib->Message()->SouthPoleY(info->Grid()->SouthPole().Y());

		itsGrib->Message()->iDirectionIncrement(info->Grid()->Di());
		itsGrib->Message()->jDirectionIncrement(info->Grid()->Dj());

		itsGrib->Message()->GridType(gridType);

		itsGrib->Message()->UVRelativeToGrid(info->Grid()->UVRelativeToGrid());

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

		itsGrib->Message()->X0(info->Grid()->BottomLeft().X());
		itsGrib->Message()->Y0(info->Grid()->BottomLeft().Y());

		itsGrib->Message()->GridOrientation(info->Grid()->Orientation());
		itsGrib->Message()->XLengthInMeters(info->Grid()->Di());
		itsGrib->Message()->YLengthInMeters(info->Grid()->Dj());
		break;
	}
	default:
		throw runtime_error(ClassName() + ": invalid projection while writing grib: " + boost::lexical_cast<string> (info->Grid()->Projection()));
		break;
	}

	itsGrib->Message()->SizeX(info->Ni());
	itsGrib->Message()->SizeY(info->Nj());

	bool iNegative = itsGrib->Message()->IScansNegatively();
	bool jPositive = itsGrib->Message()->JScansPositively();

	switch (info->Grid()->ScanningMode())
	{
	case kTopLeft:
		iNegative = false;
		jPositive = false;
		break;

	case kTopRight:
		iNegative = true;
		jPositive = false;
		break;

	case kBottomLeft:
		iNegative = false;
		jPositive = true;
		break;

	case kBottomRight:
		iNegative = true;
		jPositive = true;
		break;

	default:
		throw runtime_error(ClassName() + ": Uknown scanning mode when writing grib");
		break;

	}

	itsGrib->Message()->IScansNegatively(iNegative);
	itsGrib->Message()->JScansPositively(jPositive);

	if (info->StepSizeOverOneByte()) // Forecast with stepvalues that don't fit in one byte
	{
		itsGrib->Message()->TimeRangeIndicator(10);

		long step = info->Time().Step();
		long p1 = (step & 0xFF00) >> 8;
		long p2 = step & 0x00FF;

		itsGrib->Message()->P1(p1);
		itsGrib->Message()->P2(p2);

	}
	else
	{
		itsGrib->Message()->TimeRangeIndicator(0); // Force forecast
	}

	if (edition == 2)
	{
		itsGrib->Message()->TypeOfGeneratingProcess(1); // Forecast
	}

	// Level

	itsGrib->Message()->LevelValue(static_cast<long> (info->Level().Value()));

	// Himan levels equal to grib 1

	if (edition == 1)
	{
		itsGrib->Message()->LevelType(info->Level().Type());
	}
	else if (edition == 2)
	{
		itsGrib->Message()->LevelType(itsGrib->Message()->LevelTypeToAnotherEdition(info->Level().Type(),1));
	}

	itsGrib->Message()->Bitmap(true);

	// itsGrib->Message()->BitsPerValue(16);

	itsGrib->Message()->Values(info->Data()->Values(), info->Ni() * info->Nj());

	if (edition == 1)
	{
		//itsGrib->Message()->PackingType("grid_second_order");

	}
	else if (edition == 2)
	{
		itsGrib->Message()->PackingType("grid_jpeg");
	}

	long timeUnit = 1; // hour

	if (info->Time().StepResolution() == kMinute)
	{
		timeUnit = 0;
	}

	itsGrib->Message()->UnitOfTimeRange(timeUnit);

	/*
	 *  GRIB 1
	 *
	 * 	BIT	VALUE	MEANING
	 *	1	0		Direction increments not given
	 *	1	1		Direction increments given
	 *	2	0		Earth assumed spherical with radius = 6367.47 km
	 *	2	1		Earth assumed oblate spheroid with size as determined by IAU in 1965: 6378.160 km, 6356.775 km, f = 1/297.0
	 *	3-4	0		reserved (set to 0)
	 *	5	0		u- and v-components of vector quantities resolved relative to easterly and northerly directions
	 * 	5	1		u and v components of vector quantities resolved relative to the defined grid in the direction of increasing x and y (or i and j) coordinates respectively
	 *	6-8	0		reserved (set to 0)
	 *
	 *	GRIB2
	 *
	 *	Bit No. 	Value 	Meaning
	 *	1-2			Reserved
	 *	3		0	i direction increments not given
	 *	3		1	i direction increments given
	 *	4		0	j direction increments not given
	 *	4		1	j direction increments given
	 *	5		0	Resolved u and v components of vector quantities relative to easterly and northerly directions
	 *	5		1	Resolved u and v components of vector quantities relative to the defined grid in the direction
	 *				of increasing x and y (or i and j) coordinates, respectively.
	 *	6-8			Reserved - set to zero
	 *
	 */

	if (edition == 1)
	{
		itsGrib->Message()->ResolutionAndComponentFlags(136); // 10001000
	}
	else
	{
		itsGrib->Message()->ResolutionAndComponentFlags(56); // 00111000
	}

	itsGrib->Message()->Write(outputFile, appendToFile);

	string verb = (appendToFile ? "Appended to " : "Wrote ");
	itsLogger->Info(verb + "file '" + outputFile + "'");

	return true;
}

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool theReadContents)
{

	shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));
	vector<shared_ptr<himan::info>> infos;

	itsGrib->Open(theInputFile);

	itsLogger->Debug("Reading file '" + theInputFile + "'");

	int foundMessageNo = 0;

	long fmiProducer = options.configuration->SourceProducer().Id();

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
			long producer = options.configuration->SourceProducer().Id();
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
		else if (itsGrib->Message()->ParameterUnit() == "%")
		{
		   	p.Unit(kPrcnt);
		}
		else if (itsGrib->Message()->ParameterUnit() == "m s**-1")
		{
			p.Unit(kMs);
		}
		else
		{
			itsLogger->Trace("Unable to determine himan parameter unit for grib unit " + itsGrib->Message()->ParameterUnit());
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

		string dataTime = boost::lexical_cast<string> (dt);

		if (dt < 1000)
		{
			dataTime = "0" + dataTime;
		}

		/* Determine time step */

		long step;

		if (itsGrib->Message()->TimeRangeIndicator() == 10)
		{
			long P1 = itsGrib->Message()->P1();
			long P2 = itsGrib->Message()->P2();

			step = (P1 << 8 ) | P2;
		}
		else
		{
			step = itsGrib->Message()->EndStep();

			if (itsGrib->Message()->Edition() == 2)
			{
				step = itsGrib->Message()->ForecastTime();
			}
		}
		
		string originDateTimeStr = dataDate + dataTime;

		raw_time originDateTime (originDateTimeStr, "%Y%m%d%H%M");

		forecast_time t (originDateTime, originDateTime);

		long unitOfTimeRange = itsGrib->Message()->UnitOfTimeRange();

		HPTimeResolution timeResolution = kHour;

		if (unitOfTimeRange == 0)
		{
			timeResolution = kMinute;
		}

		t.StepResolution(timeResolution);

		t.ValidDateTime()->Adjust(timeResolution, static_cast<int> (step));

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
            
		case 112:
			levelType = himan::kGndLayer;
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
		shared_ptr<grid> newGrid (new grid());

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
			newGrid->Projection(kLatLonProjection);
			break;

		case 10:
			newGrid->Projection(kRotatedLatLonProjection);
			newGrid->SouthPole(himan::point(itsGrib->Message()->SouthPoleX(), itsGrib->Message()->SouthPoleY()));
			break;

		default:
			throw runtime_error(ClassName() + ": Unsupported projection: " + boost::lexical_cast<string> (itsGrib->Message()->NormalizedGridType()));
			break;

		}

		size_t ni = itsGrib->Message()->SizeX();
		size_t nj = itsGrib->Message()->SizeY();

		//newGrid->Data()->Ni(ni);
		//newGrid->Data()->Nj(nj);

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

		newGrid->ScanningMode(m);

		if (newGrid->Projection() == kRotatedLatLonProjection)
		{
			newGrid->UVRelativeToGrid(itsGrib->Message()->UVRelativeToGrid());
		}

		pair<point,point> coordinates = util::CoordinatesFromFirstGridPoint(himan::point(itsGrib->Message()->X0(), itsGrib->Message()->Y0()), ni, nj, itsGrib->Message()->iDirectionIncrement(),itsGrib->Message()->jDirectionIncrement(), m);

		newGrid->BottomLeft(coordinates.first);
		newGrid->TopRight(coordinates.second);

		newInfo->Create(newGrid);

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


		// Set descriptors

		newInfo->Param(p);
		newInfo->Time(t);
		newInfo->Level(l);

		shared_ptr<d_matrix_t> dm = shared_ptr<d_matrix_t> (new d_matrix_t(ni, nj));

		dm->Set(d, len);

		newInfo->Grid()->Data(dm);

		infos.push_back(newInfo);

		if (d)
		{
			free(d);
		}

		break ; // We found what we were looking for
	}

	if (infos.size())
	{
		// This will be broken when/if we return multiple infos from this function
		itsLogger->Trace("Data found from message " + boost::lexical_cast<string> (foundMessageNo) + "/" + boost::lexical_cast<string> (itsGrib->MessageCount()));
	}

	return infos;
}
