/**
 * @file grib.cpp
 *
 * @date Nov 20, 2012
 * @author partio
 */

#include "grib.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include "timer_factory.h"
#include "producer.h"
#include "util.h"
#include "grid.h"
#include <boost/filesystem.hpp>

using namespace std;
using namespace himan::plugin;

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

#include "cuda_helper.h"
#include "simple_packed.h"

grib::grib()
{

	itsLogger = logger_factory::Instance()->GetLog("grib");

	itsGrib = make_shared<NFmiGrib> ();
}

shared_ptr<NFmiGrib> grib::Reader()
{
	return itsGrib;
}

bool grib::ToFile(shared_ptr<info> anInfo, string& outputFile, HPFileType fileType, HPFileWriteOption fileWriteOption)
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

bool grib::WriteGrib(shared_ptr<const info> anInfo, string& outputFile, HPFileType fileType, bool appendToFile)
{
	auto aTimer = timer_factory::Instance()->GetTimer();
	aTimer->Start();
	
	long edition = static_cast<long> (fileType);

	// Check levelvalue since that might force us to change file type!
	
	double levelValue = anInfo->Level().Value();

	if (edition == 1 && anInfo->Level().Type() == kHybrid && levelValue > 127)
	{
		itsLogger->Info("Level value is larger than 127, changing file type to GRIB2");
		edition = 2;
	
		outputFile += "2";
	}

	itsGrib->Message().Edition(edition);

	shared_ptr<neons> n;

	if (anInfo->Producer().Centre() == kHPMissingInt)
	{
		n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string, string> producermap = n->NeonsDB().GetGridModelDefinition(static_cast<unsigned long> (anInfo->Producer().Id()));

		if (!producermap["ident_id"].empty() && !producermap["model_id"].empty())
		{
			itsGrib->Message().Centre(boost::lexical_cast<long> (producermap["ident_id"]));
			itsGrib->Message().Process(boost::lexical_cast<long> (producermap["model_id"]));

			if (edition == 1)
			{
				itsGrib->Message().Table2Version(boost::lexical_cast<long> (producermap["no_vers"]));
			}
		}
		else
		{
			string producerId = boost::lexical_cast<string> (anInfo->Producer().Id());
			itsLogger->Warning("Unable to get process and centre information from Neons for producer " + producerId);
			itsLogger->Warning("Setting process to " + producerId + " and centre to 86");
			itsGrib->Message().Centre(86);
			itsGrib->Message().Process(anInfo->Producer().Id());

			if (edition == 1)
			{
				itsLogger->Warning("Setting table2version to 203");
				itsGrib->Message().Table2Version(203);
			}
		}
		
	}
	else
	{
		itsGrib->Message().Centre(anInfo->Producer().Centre());
		itsGrib->Message().Process(anInfo->Producer().Process());

		if (edition == 1)
		{
			itsGrib->Message().Table2Version(anInfo->Producer().TableVersion());
		}
	}

	// Parameter
	
	WriteParameter(anInfo);

	// Area and Grid
	
	WriteAreaAndGrid(anInfo);

	// Time information

	WriteTime(anInfo);
	
	// Level

	itsGrib->Message().LevelValue(static_cast<long> (levelValue));

	// Himan levels equal to grib 1

	if (edition == 1)
	{
		itsGrib->Message().LevelType(anInfo->Level().Type());
	}
	else if (edition == 2)
	{
		itsGrib->Message().LevelType(itsGrib->Message().LevelTypeToAnotherEdition(anInfo->Level().Type(),2));
	}

	// Misc

	if (edition == 2)
	{
		itsGrib->Message().TypeOfGeneratingProcess(2); // Forecast
	}
	
	itsGrib->Message().Bitmap(true);

	//itsGrib->Message().BitsPerValue(16);

	itsGrib->Message().Values(anInfo->Data()->ValuesAsPOD(), static_cast<long> (anInfo->Ni() * anInfo->Nj()));

	if (edition == 2)
	{
		itsGrib->Message().PackingType("grid_jpeg");
	}

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
		itsGrib->Message().ResolutionAndComponentFlags(128); // 10000000
	}
	else
	{
		itsGrib->Message().ResolutionAndComponentFlags(48); // 00110000
	}

	vector<double> AB = anInfo->Grid()->AB();

	if (!AB.empty())
	{
		itsGrib->Message().NV(static_cast<long> (AB.size()));
		itsGrib->Message().PV(AB, AB.size());
	}

	itsGrib->Message().Write(outputFile, appendToFile);

	aTimer->Stop();
	long duration = aTimer->GetTime();

	long bytes = boost::filesystem::file_size(outputFile);

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));
	
	string verb = (appendToFile ? "Appended to " : "Wrote ");
	itsLogger->Info(verb + "file '" + outputFile + "' (" + boost::lexical_cast<string> (speed) + " MB/s)");

	return true;
}

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool readContents, bool readPackedData)
{

	auto n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

	vector<shared_ptr<himan::info>> infos;

	if (!itsGrib->Open(theInputFile))
	{
		itsLogger->Error("Opening file '" + theInputFile + "' failed");
		return infos;
	}

	int foundMessageNo = 0;

	if (options.prod.Centre() == kHPMissingInt)
	{
		itsLogger->Error("Process and centre information for producer " + boost::lexical_cast<string> (options.prod.Id()) + " are undefined");
		return infos;
	}

	auto aTimer = timer_factory::Instance()->GetTimer();
	
	while (itsGrib->NextMessage())
	{
	
		foundMessageNo++;

		/*
		 * One grib file may contain many grib messages. Loop though all messages
		 * and get all that match our search options.
		 *
		 */

		//<!todo Should we actually return all matching messages or only the first one

		long centre = itsGrib->Message().Centre();
		long process = itsGrib->Message().Process();

		if (options.prod.Process() != process || options.prod.Centre() != centre)
		{
			itsLogger->Trace("centre/process do not match: " + boost::lexical_cast<string> (options.prod.Process()) + " vs " + boost::lexical_cast<string> (process));
			itsLogger->Trace("centre/process do not match: " + boost::lexical_cast<string> (options.prod.Centre()) + " vs " + boost::lexical_cast<string> (centre));
			//continue;
		}

		param p;

		long number = itsGrib->Message().ParameterNumber();

		if (itsGrib->Message().Edition() == 1)
		{
			long no_vers = itsGrib->Message().Table2Version();

			long timeRangeIndicator = itsGrib->Message().TimeRangeIndicator();

			string parmName = n->GribParameterName(number, no_vers, timeRangeIndicator);

			if (parmName.empty())
			{
				itsLogger->Warning("Parameter name not found from Neons for no_vers: " +
							boost::lexical_cast<string> (no_vers) + ", number: " +
							boost::lexical_cast<string> (number) + ", timeRangeIndicator: " +
							boost::lexical_cast<string> (timeRangeIndicator));
			}

			p.Name(parmName);
			
			p.GribParameter(number);
			p.GribTableVersion(no_vers);

		}
		else
		{
			long category = itsGrib->Message().ParameterCategory();
			long discipline = itsGrib->Message().ParameterDiscipline();
			long process = options.prod.Process();
		
			p.Name(n->GribParameterName(number, category, discipline, process));

			p.GribParameter(number);
			p.GribDiscipline(discipline);
			p.GribCategory(category);

			if (p.Name() == "T-C" && options.prod.Centre() == 7)
			{
				p.Name("T-K");
			}
		}
		
		// Start timing after the last neons call so we get at least a 
		// somewhat accurate timing result.
		
		aTimer->Start();

		string unit = itsGrib->Message().ParameterUnit();
		
		if (unit == "K")
		{
		   	p.Unit(kK);
		}
		else if (unit == "Pa s-1" || unit == "Pa s**-1" )
		{
		   	p.Unit(kPas);
		}
		else if (unit == "%")
		{
		   	p.Unit(kPrcnt);
		}
		else if (unit == "m s**-1" || unit == "m s-k1")
		{
			p.Unit(kMs);
		}
		else if (unit == "m" || unit == "m of water equivalent")
		{
			p.Unit(kM);
		}
		else if (unit == "mm")
		{
			p.Unit(kMm);
		}
		else if (unit == "Pa")
		{
			p.Unit(kPa);
		}
		else if (unit == "m**2 s**-2")
		{
			p.Unit(kGph);
		}
		else if (unit == "kg kg**-1")
		{
			p.Unit(kKgkg);
		}
		else if (unit == "J m**-2")
		{
			p.Unit(kJm2);
		}
		else if (unit == "kg m**-2")
		{
			p.Unit(kKgm2);
		}
		else
		{
			itsLogger->Trace("Unable to determine himan parameter unit for grib unit " + itsGrib->Message().ParameterUnit());
		}

		if (p != options.param)
		{
			itsLogger->Trace("Parameter does not match: " + options.param.Name() + " (requested) vs " + p.Name() + " (found)");
			continue;
		}

		string dataDate = boost::lexical_cast<string> (itsGrib->Message().DataDate());

		/*
		 * dataTime is HH24MM in long datatype.
		 *
		 * So, for example analysistime 00 is 0, and 06 is 600.
		 *
		 */

		long dt = itsGrib->Message().DataTime();

		string dataTime = boost::lexical_cast<string> (dt);

		if (dt < 1000)
		{
			dataTime = "0" + dataTime;
		}

		long step = itsGrib->Message().NormalizedStep(true, true);
		
		string originDateTimeStr = dataDate + dataTime;

		raw_time originDateTime (originDateTimeStr, "%Y%m%d%H%M");

		forecast_time t (originDateTime, originDateTime);

		long unitOfTimeRange = itsGrib->Message().NormalizedUnitOfTimeRange();

		HPTimeResolution timeResolution = kUnknownTimeResolution;

		switch (unitOfTimeRange)
		{
			case 1:
			case 10:
			case 11:
			case 12:
				timeResolution = kHourResolution;
				break;
				
			case 0:
			case 13:
			case 14:
				timeResolution = kMinuteResolution;
				break;

			default:
				itsLogger->Warning("Unsupported unit of time range: " + boost::lexical_cast<string> (timeResolution));
				break;
		}

		t.StepResolution(timeResolution);

		t.ValidDateTime()->Adjust(timeResolution, static_cast<int> (step));

		if (t != options.time)
		{
			itsLogger->Trace("Times do not match");
			itsLogger->Trace("OriginDateTime: " + options.time.OriginDateTime()->String() + " (requested) vs " + t.OriginDateTime()->String() + " (found)");
			itsLogger->Trace("ValidDateTime: " + options.time.ValidDateTime()->String() + " (requested) vs " + t.ValidDateTime()->String() + " (found)");
			itsLogger->Trace("Step resolution: " + string(HPTimeResolutionToString.at(options.time.StepResolution())) + " (requested) vs " + string(HPTimeResolutionToString.at(t.StepResolution())) + " (found)");

			continue;
		}

		long gribLevel = itsGrib->Message().NormalizedLevelType();

		himan::HPLevelType levelType;

		switch (gribLevel)
		{
		case 1:
			levelType = himan::kGround;
			break;

		case 8:
			levelType = himan::kTopOfAtmosphere;
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
			itsLogger->Error(ClassName() + ": Unsupported level type: " + boost::lexical_cast<string> (gribLevel));
			continue;
			break;

		}

		level l (levelType, static_cast<float> (itsGrib->Message().LevelValue()));

		if (l != options.level)
		{
			itsLogger->Trace("Level does not match: " + 
				string(HPLevelTypeToString.at(options.level.Type())) +
				" " +
				string(boost::lexical_cast<string> (options.level.Value())) +
				" vs " +
				string(HPLevelTypeToString.at(l.Type())) +
				" " +
				string(boost::lexical_cast<string> (l.Value())));
			continue;
		}

		std::vector<double> ab;

		if (levelType == himan::kHybrid)
		{
		 	long nv = itsGrib->Message().NV();
		 	long lev = itsGrib->Message().LevelValue();
			ab = itsGrib->Message().PV(static_cast<size_t> (nv), static_cast<size_t> (lev));
		}

		// END VALIDATION OF SEARCH PARAMETERS

		shared_ptr<info> newInfo (new info());
		grid* newGrid = new grid();

		producer prod(itsGrib->Message().Centre(), process);

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

		size_t ni = static_cast<size_t> (itsGrib->Message().SizeX());
		size_t nj = static_cast<size_t> (itsGrib->Message().SizeY());

		switch (itsGrib->Message().NormalizedGridType())
		{
		case 0:
			newGrid->Projection(kLatLonProjection);
			break;

		case 5:
			newGrid->Projection(kStereographicProjection);

			newGrid->Orientation(itsGrib->Message().GridOrientation());
			newGrid->Di(itsGrib->Message().XLengthInMeters());
			newGrid->Dj(itsGrib->Message().YLengthInMeters());
			break;

		case 10:
			newGrid->Projection(kRotatedLatLonProjection);
			newGrid->SouthPole(himan::point(itsGrib->Message().SouthPoleX(), itsGrib->Message().SouthPoleY()));
			break;

		default:
			throw runtime_error(ClassName() + ": Unsupported projection: " + boost::lexical_cast<string> (itsGrib->Message().NormalizedGridType()));
			break;

		}

		bool iNegative = itsGrib->Message().IScansNegatively();
		bool jPositive = itsGrib->Message().JScansPositively();

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
			newGrid->UVRelativeToGrid(itsGrib->Message().UVRelativeToGrid());
		}

		double X0 = itsGrib->Message().X0();
		double Y0 = itsGrib->Message().Y0();

		// GRIB2 has longitude 0 .. 360, but in neons we have it -180 .. 180
		// NB! ONLY FOR EC and FMI! GFS and GEM geometries are in grib2 format
		//
		// Make conversion to GRIB1 style coordinates, but in the long run we should figure out how to
		// handle grib 1 & grib 2 longitude values in a smart way. (a single geometry
		// can have coordinates in both ways!)
		
		if (itsGrib->Message().Edition() == 2 && (centre == 98 || centre == 86))
		{
			X0 -= 360;
		}

		if (newGrid->Projection() == kStereographicProjection)
		{
			/*
			 * Do not support stereographic projections but in bottom left scanning mode.
			 *
			 * The calculation of grid extremes could be done with f.ex. NFmiAzimuthalArea
			 * but lets not do that unless it's absolutely necessary.
			 */
			
			if (newGrid->ScanningMode() != kBottomLeft)
			{
				itsLogger->Error(ClassName() + ": stereographic projection only supported when scanning mode is bottom left");
				continue;
			}

			const point first(X0, Y0);

			newGrid->BottomLeft(first);

			assert(newGrid->ScanningMode() == kBottomLeft);
			
			std::pair<point, point> coordinates = util::CoordinatesFromFirstGridPoint(first, newGrid->Orientation(), ni, nj, newGrid->Di(), newGrid->Dj());

			newGrid->TopRight(coordinates.second);
		
		}
		else
		{

			himan::point firstPoint(X0, Y0);

			if (centre == 98 && firstPoint.X() == 180)
			{

				/**
				 * Global EC data area is defined as
				 *
				 * latitudeOfFirstGridPointInDegrees = 90;
				 * longitudeOfFirstGridPointInDegrees = 180;
				 * latitudeOfLastGridPointInDegrees = 0;
				 * longitudeOfLastGridPointInDegrees = 180;
				 *
				 * This area makes no sense, normalize the first value to -180.
				 */

				assert(m == kBottomLeft || m == kTopLeft); // Want to make sure we always read from left to right
				
				firstPoint.X(-180.);
			}

			pair<point,point> coordinates = util::CoordinatesFromFirstGridPoint(firstPoint, ni, nj, itsGrib->Message().iDirectionIncrement(),itsGrib->Message().jDirectionIncrement(), m);

			newGrid->BottomLeft(coordinates.first);
			newGrid->TopRight(coordinates.second);
		}

		newInfo->Create(newGrid);

		// Set descriptors

		newInfo->Param(p);
		newInfo->Time(t);
		newInfo->Level(l);

		shared_ptr<unpacked> dm = shared_ptr<unpacked> (new unpacked(ni, nj));

		/*
		 * Read data from grib *
		 */

#if defined GRIB_READ_PACKED_DATA && defined HAVE_CUDA

		if (readPackedData && itsGrib->Message().PackingType() == "grid_simple")
		{
			size_t len = itsGrib->Message().PackedValuesLength();

			unsigned char* data = 0;
			int* unpackedBitmap;

			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&data), len * sizeof(unsigned char)));

			// Get packed values from grib
			
			itsGrib->Message().PackedValues(data);

			itsLogger->Trace("Retrieved " + boost::lexical_cast<string> (len) + " bytes of packed data from grib");
			
			double bsf = itsGrib->Message().BinaryScaleFactor();
			double dsf = itsGrib->Message().DecimalScaleFactor();
			double rv = itsGrib->Message().ReferenceValue();
			long bpv = itsGrib->Message().BitsPerValue();

			auto packed = std::make_shared<simple_packed> (bpv, util::ToPower(bsf,2), util::ToPower(-dsf, 10), rv);

			packed->Set(data, len, static_cast<size_t> (itsGrib->Message().SizeX() * itsGrib->Message().SizeY()));

			if (itsGrib->Message().Bitmap())
			{
				size_t bitmap_len =itsGrib->Message().BytesLength("bitmap");
				size_t bitmap_size = static_cast<size_t> (ceil(static_cast<double> (bitmap_len)/8));

				itsLogger->Trace("Grib has bitmap, length " + boost::lexical_cast<string> (bitmap_len) + " size " + boost::lexical_cast<string> (bitmap_size) + " bytes");

				CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&unpackedBitmap), bitmap_len * sizeof(int)));

				unsigned char* bitmap = new unsigned char[bitmap_size];

				itsGrib->Message().Bytes("bitmap", bitmap);

				UnpackBitmap(bitmap, unpackedBitmap, bitmap_size, bitmap_len);
				
				packed->Bitmap(unpackedBitmap, bitmap_len);

				delete [] bitmap;
			}

			newInfo->Grid()->PackedData(packed);
		}
		else
#endif
		if (readContents)
		{
			size_t len = itsGrib->Message().ValuesLength();

			double* d = itsGrib->Message().Values();

			dm->Set(d, len);

			free(d);

			itsLogger->Trace("Retrieved " + boost::lexical_cast<string> (len * 8) + " bytes of unpacked data from grib");

		}

		newInfo->Grid()->Data(dm);

		infos.push_back(newInfo);

		aTimer->Stop();
		
		break ; // We found what we were looking for
	}

	long duration = aTimer->GetTime();
	long bytes = boost::filesystem::file_size(theInputFile);

	double speed = floor((bytes / 1024. / 1024.) / (duration / 1000.));

	itsLogger->Debug("Read file '" + theInputFile + "' (" + boost::lexical_cast<string> (speed) + " MB/s)");
	
	return infos;
}


#define BitMask1(i)	(1u << i)
#define BitTest(n,i)	!!((n) & BitMask1(i))

void grib::UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len, size_t unpackedLen) const
{
	size_t i, idx = 0, v = 1;

	short j = 0;

	for (i = 0; i < len; i++)
	{
		for (j = 7; j >= 0; j--)
		{
			if (BitTest(bitmap[i], j))
			{
				unpacked[idx] = v++;
			}
			else
			{
				unpacked[idx] = 0;
			}

			if (++idx >= unpackedLen)
			{
				// packed data might not be aligned nicely along byte boundaries --
				// need to break from loop after final element has been processed
				break;
			}
	    }
	}
}

void grib::WriteAreaAndGrid(std::shared_ptr<const info> anInfo)
{
	himan::point firstGridPoint = anInfo->Grid()->FirstGridPoint();
	himan::point lastGridPoint = anInfo->Grid()->LastGridPoint();

	long edition = itsGrib->Message().Edition();
	
	switch (anInfo->Grid()->Projection())
	{
		case kLatLonProjection:
		{
			long gridType = 0; // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(firstGridPoint.X());
			itsGrib->Message().X1(lastGridPoint.X());
			itsGrib->Message().Y0(firstGridPoint.Y());
			itsGrib->Message().Y1(lastGridPoint.Y());

			itsGrib->Message().iDirectionIncrement(anInfo->Di());
			itsGrib->Message().jDirectionIncrement(anInfo->Dj());

			break;
		}

		case kRotatedLatLonProjection:
		{

			long gridType = 10; // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(firstGridPoint.X());
			itsGrib->Message().Y0(firstGridPoint.Y());
			itsGrib->Message().X1(lastGridPoint.X());
			itsGrib->Message().Y1(lastGridPoint.Y());

			itsGrib->Message().SouthPoleX(anInfo->Grid()->SouthPole().X());
			itsGrib->Message().SouthPoleY(anInfo->Grid()->SouthPole().Y());

			itsGrib->Message().iDirectionIncrement(anInfo->Grid()->Di());
			itsGrib->Message().jDirectionIncrement(anInfo->Grid()->Dj());

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().UVRelativeToGrid(anInfo->Grid()->UVRelativeToGrid());

			break;
		}

		case kStereographicProjection:
		{
			long gridType = 5; // Grib 1

			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(anInfo->Grid()->BottomLeft().X());
			itsGrib->Message().Y0(anInfo->Grid()->BottomLeft().Y());

			itsGrib->Message().GridOrientation(anInfo->Grid()->Orientation());

			itsGrib->Message().XLengthInMeters(anInfo->Grid()->Di());
			itsGrib->Message().YLengthInMeters(anInfo->Grid()->Dj());
			break;
		}

		default:
			throw runtime_error(ClassName() + ": invalid projection while writing grib: " + boost::lexical_cast<string> (anInfo->Grid()->Projection()));
			break;
	}

	itsGrib->Message().SizeX(static_cast<long> (anInfo->Ni()));
	itsGrib->Message().SizeY(static_cast<long> (anInfo->Nj()));

	bool iNegative = itsGrib->Message().IScansNegatively();
	bool jPositive = itsGrib->Message().JScansPositively();

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

	itsGrib->Message().IScansNegatively(iNegative);
	itsGrib->Message().JScansPositively(jPositive);
}

void grib::WriteTime(std::shared_ptr<const info> anInfo)
{
	itsGrib->Message().DataDate(boost::lexical_cast<long> (anInfo->Time().OriginDateTime()->String("%Y%m%d")));
	itsGrib->Message().DataTime(boost::lexical_cast<long> (anInfo->Time().OriginDateTime()->String("%H%M")));

	if (itsGrib->Message().Edition() == 1)
	{
		itsGrib->Message().StartStep(anInfo->Time().Step());
		itsGrib->Message().EndStep(anInfo->Time().Step());
	}
	else
	{
		itsGrib->Message().ForecastTime(anInfo->Time().Step());
	}

	/*
	 * Check if this is an aggregated parameter.
	 *
	 * At least when writing grib2, the aggregation of the parameter is defined outside
	 * the actual parameter numbering scheme.
	 *
	 * One thing to note is that when we have mixed time types in the grib, for example
	 * when calculating hourly parameters to harmonie (forecast time in minutes, aggregation
	 * time in hours) we must convert them to the same unit, in harmonies case minute.
	 */

	long unitOfTimeRange = 1; // hour
	
	if (anInfo->Param().Aggregation().Type() != kUnknownAggregationType)
	{
		long timeRangeValue = 1;

		if (anInfo->Param().Aggregation().TimeResolution() == kHourResolution)
		{
			timeRangeValue = 1;

			int timeResolutionValue = anInfo->Param().Aggregation().TimeResolutionValue();

			if (anInfo->Time().StepResolution() == kHourResolution)
			{
				// GRIB1 values !
				// http://www.nco.ncep.noaa.gov/pmb/docs/on388/table4.html
				/*
				if (timeResolutionValue == 1)
				{
					unitOfTimeRange = 1; // hour
				}
				else if (timeResolutionValue == 3)
				{
					unitOfTimeRange = 10; // 3 hours
				}
				else if (timeResolutionValue == 6)
				{
					unitOfTimeRange = 11; // 6 hours
				}
				else if (timeResolutionValue == 12)
				{
					unitOfTimeRange = 12; // 12 hours
				}
				else
				{
					throw runtime_error(ClassName() + ": Invalid unitOfTimeRange: " + boost::lexical_cast<string> (timeResolutionValue));
				}
				*/
			}
			else
			{
				// mixed time types in the grib, must convert
				timeRangeValue = 60 * timeResolutionValue;
				unitOfTimeRange = 0; // minute
				
			}			
		}
		else if (anInfo->Param().Aggregation().TimeResolution() == kMinuteResolution)
		{
			itsLogger->Warning(ClassName() + ": minute resolution for aggregated data, seems fishy");
			
			unitOfTimeRange = 0; // minute
			timeRangeValue = anInfo->Param().Aggregation().TimeResolutionValue();
		}

		itsGrib->Message().LengthOfTimeRange(timeRangeValue);

	}
	else
	{
		if (anInfo->Time().StepResolution() == kMinuteResolution)
		{
			unitOfTimeRange = 0;
		}		
	}
	
	itsGrib->Message().UnitOfTimeRange(unitOfTimeRange);
	
	if (anInfo->StepSizeOverOneByte()) // Forecast with stepvalues that don't fit in one byte
	{
		itsGrib->Message().TimeRangeIndicator(10);

		long step = anInfo->Time().Step();
		long p1 = (step & 0xFF00) >> 8;
		long p2 = step & 0x00FF;

		itsGrib->Message().P1(p1);
		itsGrib->Message().P2(p2);

	}
	else
	{
		itsGrib->Message().TimeRangeIndicator(0); // Force forecast
	}
}

void grib::WriteParameter(std::shared_ptr<const info> anInfo)
{
	/*
	 * For grib1, get param_id from neons since its dependant on the table2version
	 *
	 * For grib2, assume the plugin has set the correct numbers since they are "static".
	 */

	if (itsGrib->Message().Edition() == 1)
	{
		shared_ptr<neons> n;
		
		long parm_id = anInfo->Param().GribIndicatorOfParameter();

		if (parm_id == kHPMissingInt)
		{
			if (!n)
			{
				n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));
			}

			parm_id = n->NeonsDB().GetGridParameterId(itsGrib->Message().Table2Version(), anInfo->Param().Name());

			if (parm_id == -1)
			{
				itsLogger->Warning("Parameter " + anInfo->Param().Name() + " does not have mapping for code table " + boost::lexical_cast<string> (anInfo->Producer().TableVersion()) + " in neons");
			}
		}

		itsGrib->Message().ParameterNumber(parm_id);
	}
	else
	{
		itsGrib->Message().ParameterNumber(anInfo->Param().GribParameter());
		itsGrib->Message().ParameterCategory(anInfo->Param().GribCategory());
		itsGrib->Message().ParameterDiscipline(anInfo->Param().GribDiscipline()) ;

		if (anInfo->Param().Aggregation().Type() != kUnknownAggregationType)
		{
			itsGrib->Message().ProductDefinitionTemplateNumber(8);

			long type;

			switch (anInfo->Param().Aggregation().Type())
			{
				default:
				case kAverage:
					type=0;
					break;
				case kAccumulation:
					type=1;
					break;
				case kMaximum:
					type=2;
					break;
				case kMinimum:
					type=3;
					break;
			}

			itsGrib->Message().TypeOfStatisticalProcessing(type);
		}
	}
}
