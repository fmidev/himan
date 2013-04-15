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

#define READ_PACKED_DATA

grib::grib()
{

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("grib"));

	itsGrib = shared_ptr<NFmiGrib> (new NFmiGrib());
}

shared_ptr<NFmiGrib> grib::Reader()
{
	return itsGrib;
}

bool grib::ToFile(shared_ptr<info> anInfo, const string& outputFile, HPFileType fileType, HPFileWriteOption fileWriteOption)
{

	if (fileWriteOption == kNeons || fileWriteOption == kMultipleFiles)
	{
		// Write only that data which is currently set at descriptors

		WriteGrib(anInfo, outputFile, fileType);
	}

	else
	{
		anInfo->ResetTime();

		while (anInfo->NextTime())
	{
			anInfo->ResetLevel();

			while (anInfo->NextLevel())
			{
				anInfo->ResetParam();

				while (anInfo->NextParam())
				{
					if (!WriteGrib(anInfo, outputFile, fileType, true))
					{
						itsLogger->Error("Error writing grib to file");
					}
				}
			}
		}
	}

	return true;

}

bool grib::WriteGrib(shared_ptr<const info> anInfo, const string& outputFile, HPFileType fileType, bool appendToFile)
{
	const static long edition = static_cast<long> (fileType);

	/* Section 0 */

	itsGrib->Message()->Edition(edition);

	shared_ptr<neons> n; 

	long no_vers = anInfo->Producer().TableVersion(); // We might need this later on

	if (anInfo->Producer().Centre() == kHPMissingInt)
	{
		n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string, string> producermap = n->NeonsDB().GetGridModelDefinition(anInfo->Producer().Id());

		itsGrib->Message()->Centre(boost::lexical_cast<long> (producermap["ident_id"]));
		itsGrib->Message()->Process(boost::lexical_cast<long> (producermap["model_id"]));

		no_vers = boost::lexical_cast<long> (producermap["no_vers"]);
	}
	else
	{
		itsGrib->Message()->Centre(anInfo->Producer().Centre());
		itsGrib->Message()->Process(anInfo->Producer().Process());
	}

	/*
	 * For grib1, get param_id from neons since its dependant on the table2version
	 *
	 * For grib2, assume the plugin has set the correct numbers since they are "static".
	 */

	if (edition == 1)
	{
		if (anInfo->Producer().TableVersion() != kHPMissingInt)
		{
			no_vers = anInfo->Producer().TableVersion();
		}
		else if (no_vers == kHPMissingInt)
		{
			map<string, string> producermap = n->NeonsDB().GetGridModelDefinition(anInfo->Producer().Id());
			no_vers = boost::lexical_cast<long> (producermap["no_vers"]);
		}

		long parm_id = anInfo->Param().GribIndicatorOfParameter();

		if (parm_id == kHPMissingInt)
		{
			if (!n)
			{
				n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));
			}
			
			parm_id = n->NeonsDB().GetGridParameterId(no_vers, anInfo->Param().Name());
		}

		itsGrib->Message()->ParameterNumber(parm_id);
		itsGrib->Message()->Table2Version(anInfo->Producer().TableVersion());
	}
	else if (edition == 2)
	{
		itsGrib->Message()->ParameterNumber(anInfo->Param().GribParameter());
		itsGrib->Message()->ParameterCategory(anInfo->Param().GribCategory());
		itsGrib->Message()->ParameterDiscipline(anInfo->Param().GribDiscipline()) ;
	}

	itsGrib->Message()->DataDate(boost::lexical_cast<long> (anInfo->Time().OriginDateTime()->String("%Y%m%d")));
	itsGrib->Message()->DataTime(boost::lexical_cast<long> (anInfo->Time().OriginDateTime()->String("%H%M")));

	itsGrib->Message()->StartStep(anInfo->Time().Step());
	itsGrib->Message()->EndStep(anInfo->Time().Step());

	himan::point firstGridPoint = anInfo->Grid()->FirstGridPoint();
	himan::point lastGridPoint = anInfo->Grid()->LastGridPoint();

	switch (anInfo->Grid()->Projection())
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
		itsGrib->Message()->X1(lastGridPoint.X());
		itsGrib->Message()->Y0(firstGridPoint.Y());
		itsGrib->Message()->Y1(lastGridPoint.Y());
		
		itsGrib->Message()->iDirectionIncrement(anInfo->Di());
		itsGrib->Message()->jDirectionIncrement(anInfo->Dj());
		
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

		itsGrib->Message()->SouthPoleX(anInfo->Grid()->SouthPole().X());
		itsGrib->Message()->SouthPoleY(anInfo->Grid()->SouthPole().Y());

		itsGrib->Message()->iDirectionIncrement(anInfo->Grid()->Di());
		itsGrib->Message()->jDirectionIncrement(anInfo->Grid()->Dj());

		itsGrib->Message()->GridType(gridType);

		itsGrib->Message()->UVRelativeToGrid(anInfo->Grid()->UVRelativeToGrid());

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

		itsGrib->Message()->X0(anInfo->Grid()->BottomLeft().X());
		itsGrib->Message()->Y0(anInfo->Grid()->BottomLeft().Y());

		itsGrib->Message()->GridOrientation(anInfo->Grid()->Orientation());
		itsGrib->Message()->XLengthInMeters(anInfo->Grid()->Di());
		itsGrib->Message()->YLengthInMeters(anInfo->Grid()->Dj());
		break;
	}
	
	default:
		throw runtime_error(ClassName() + ": invalid projection while writing grib: " + boost::lexical_cast<string> (anInfo->Grid()->Projection()));
		break;
	}

	itsGrib->Message()->SizeX(anInfo->Ni());
	itsGrib->Message()->SizeY(anInfo->Nj());

	bool iNegative = itsGrib->Message()->IScansNegatively();
	bool jPositive = itsGrib->Message()->JScansPositively();

	switch (anInfo->Grid()->ScanningMode())
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

	if (anInfo->StepSizeOverOneByte()) // Forecast with stepvalues that don't fit in one byte
	{
		itsGrib->Message()->TimeRangeIndicator(10);

		long step = anInfo->Time().Step();
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
		itsGrib->Message()->TypeOfGeneratingProcess(2); // Forecast
	}

	// Level

	itsGrib->Message()->LevelValue(static_cast<long> (anInfo->Level().Value()));

	// Himan levels equal to grib 1

	if (edition == 1)
	{
		itsGrib->Message()->LevelType(anInfo->Level().Type());
	}
	else if (edition == 2)
	{
		itsGrib->Message()->LevelType(itsGrib->Message()->LevelTypeToAnotherEdition(anInfo->Level().Type(),1));
	}

	itsGrib->Message()->Bitmap(true);

	//itsGrib->Message()->BitsPerValue(16);

	itsGrib->Message()->Values(anInfo->Data()->Values(), anInfo->Ni() * anInfo->Nj());

	if (edition == 1)
	{
		//itsGrib->Message()->PackingType("grid_second_order");

	}
	else if (edition == 2)
	{
		itsGrib->Message()->PackingType("grid_jpeg");
	}

	long timeUnit = 1; // hour

	if (anInfo->Time().StepResolution() == kMinuteResolution)
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
		itsGrib->Message()->ResolutionAndComponentFlags(128); // 10000000
	}
	else
	{
		itsGrib->Message()->ResolutionAndComponentFlags(48); // 00110000
	}

	itsGrib->Message()->Write(outputFile, appendToFile);

	string verb = (appendToFile ? "Appended to " : "Wrote ");
	itsLogger->Info(verb + "file '" + outputFile + "'");

	return true;
}

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool readContents, bool readPackedData)
{

	shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

	vector<shared_ptr<himan::info>> infos;

	itsGrib->Open(theInputFile);

	itsLogger->Debug("Reading file '" + theInputFile + "'");

	int foundMessageNo = 0;

	if (options.configuration->SourceProducer().Centre() == kHPMissingInt)
	{
		throw runtime_error(ClassName() + ": Process and centre information for producer " + boost::lexical_cast<string> (options.configuration->SourceProducer().Id()) + " not found from neons");
	}

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

		if (options.configuration->SourceProducer().Process() != process || options.configuration->SourceProducer().Centre() != centre)
		{
			itsLogger->Trace("centre/process do not match: " + boost::lexical_cast<string> (options.configuration->SourceProducer().Process()) + " vs " + boost::lexical_cast<string> (process));
			itsLogger->Trace("centre/process do not match: " + boost::lexical_cast<string> (options.configuration->SourceProducer().Centre()) + " vs " + boost::lexical_cast<string> (centre));
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
			long process = options.configuration->SourceProducer().Process();
		
			p.Name(n->GribParameterName(number, category, discipline, process));

			p.GribParameter(number);
			p.GribDiscipline(discipline);
			p.GribCategory(category);

			if (p.Name() == "T-C" && options.configuration->SourceProducer().Centre() == 7)
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

		HPTimeResolution timeResolution = kHourResolution;

		if (unitOfTimeRange == 0)
		{
			timeResolution = kMinuteResolution;
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

		std::vector<double> ab;

		if (levelType == himan::kHybrid)
		{
		 	long nv = itsGrib->Message()->NV();
		 	long lev = itsGrib->Message()->LevelValue();
			ab = itsGrib->Message()->PV(nv, lev);
		}

		// END VALIDATION OF SEARCH PARAMETERS

		shared_ptr<info> newInfo (new info());
		shared_ptr<grid> newGrid (new grid());

		producer prod(itsGrib->Message()->Centre(), process);

		newGrid->AB(ab);

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

		// Set descriptors

		newInfo->Param(p);
		newInfo->Time(t);
		newInfo->Level(l);

		shared_ptr<unpacked> dm = shared_ptr<unpacked> (new unpacked(ni, nj));

		/*
		 * Read data from grib *
		 */

		size_t len = 0;

#if defined READ_PACKED_DATA && defined HAVE_CUDA

		if (readPackedData && itsGrib->Message()->PackingType() == "grid_simple")
		{
			itsLogger->Trace("Retrieving packed data from grib");
			
			len = itsGrib->Message()->UnpackedValuesLength();

			unsigned char* u = itsGrib->Message()->UnpackedValues();

			double bsf = itsGrib->Message()->BinaryScaleFactor();
			double dsf = itsGrib->Message()->DecimalScaleFactor();
			double rv = itsGrib->Message()->ReferenceValue();
			long bpv = itsGrib->Message()->BitsPerValue();

			size_t len = itsGrib->Message()->Section4Length();

			auto packed = std::make_shared<simple_packed> (bpv, util::ToPower(bsf,2), util::ToPower(-dsf, 10), rv);

			//packed->Resize(len);
			
			packed->Set(u, len);

			newInfo->Grid()->PackedData(packed);
		}
		else
#endif
		if (readContents)
		{
			len = itsGrib->Message()->ValuesLength();

			double* d = itsGrib->Message()->Values();

			dm->Set(d, len);

			free(d);
		}

		newInfo->Grid()->Data(dm);

		infos.push_back(newInfo);

		break ; // We found what we were looking for
	}

	if (infos.size())
	{
		// This will be broken when/if we return multiple infos from this function
		itsLogger->Trace("Data found from message " + boost::lexical_cast<string> (foundMessageNo) + "/" + boost::lexical_cast<string> (itsGrib->MessageCount()));
	}

	return infos;
}
