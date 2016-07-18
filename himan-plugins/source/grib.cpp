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
#include "latitude_longitude_grid.h"
#include "stereographic_grid.h"
#include "reduced_gaussian_grid.h"
#include <boost/filesystem.hpp>

using namespace std;
using namespace himan::plugin;

#include "neons.h"
#include "radon.h"
#include "cache.h"

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

bool grib::ToFile(info& anInfo, string& outputFile, bool appendToFile)
{
	// Write only that data which is currently set at descriptors

	auto aTimer = timer_factory::Instance()->GetTimer();
	aTimer->Start();

	if (anInfo.Grid()->Class() == kIrregularGrid && anInfo.Grid()->Type() != kReducedGaussian)
	{
		itsLogger->Error("Unable to write irregular grid of type " + HPGridTypeToString.at(anInfo.Grid()->Type()) + " to grib");
		return false;
	}
	
	if (!itsWriteOptions.write_empty_grid)
	{
		if (anInfo.Data().MissingCount() == anInfo.Data().Size())
		{
			itsLogger->Debug("Not writing empty grid");
			return true;
		}
	}

	long edition = static_cast<long> (itsWriteOptions.configuration->OutputFileType());

	// Check levelvalue and forecast type since those might force us to change to grib2!
	
	double levelValue = anInfo.Level().Value();
	HPForecastType forecastType = anInfo.ForecastType().Type();

	if (edition == 1 && ((anInfo.Level().Type() == kHybrid && levelValue > 127) || (forecastType == kEpsControl || forecastType == kEpsPerturbation)))
	{
		itsLogger->Debug("File type forced to GRIB2 (level value: " + boost::lexical_cast<string> (levelValue) + ", forecast type: " + HPForecastTypeToString.at(forecastType) + ")");
		edition = 2;
		if (itsWriteOptions.configuration->FileCompression() == kNoCompression && itsWriteOptions.configuration->FileWriteOption() != kSingleFile)
		{	
			outputFile += "2";
		}
		
		if (itsWriteOptions.configuration->FileCompression() == kGZIP)
		{
			outputFile.insert(outputFile.end()-3, '2');
		}
		else if (itsWriteOptions.configuration->FileCompression() == kBZIP2)
		{
			outputFile.insert(outputFile.end()-4, '2');
		}
		else if (itsWriteOptions.configuration->FileCompression() != kNoCompression)
		{
			itsLogger->Error("Unable to write to compressed grib. Unknown file compression: " + HPFileCompressionToString.at(itsWriteOptions.configuration->FileCompression()));
			return false;
		}
	}

	itsGrib->Message().Edition(edition);

	if (anInfo.Producer().Centre() == kHPMissingInt)
	{
		auto n = GET_PLUGIN(neons);

		map<string, string> producermap = n->NeonsDB().GetGridModelDefinition(static_cast<unsigned long> (anInfo.Producer().Id()));

		if (!producermap["ident_id"].empty() && !producermap["model_id"].empty())
		{
			itsGrib->Message().Centre(boost::lexical_cast<long> (producermap["ident_id"]));
			itsGrib->Message().Process(boost::lexical_cast<long> (producermap["model_id"]));
		}
		else
		{
			string producerId = boost::lexical_cast<string> (anInfo.Producer().Id());
			itsLogger->Warning("Unable to get process and centre information from Neons for producer " + producerId);
			itsLogger->Warning("Setting process to " + producerId + " and centre to 86");
			itsGrib->Message().Centre(86);
			itsGrib->Message().Process(anInfo.Producer().Id());
		}
		
	}
	else
	{
		itsGrib->Message().Centre(anInfo.Producer().Centre());
		itsGrib->Message().Process(anInfo.Producer().Process());
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
		itsGrib->Message().LevelType(anInfo.Level().Type());
	}
	else if (edition == 2)
	{
		itsGrib->Message().LevelType(itsGrib->Message().LevelTypeToAnotherEdition(anInfo.Level().Type(),2));
	}

	// Forecast type

	itsGrib->Message().ForecastType(anInfo.ForecastType().Type());

	if (static_cast<int> (anInfo.ForecastType().Type()) > 2)
	{
		itsGrib->Message().ForecastTypeValue(static_cast<long> (anInfo.ForecastType().Value()));
	}

	if (itsWriteOptions.use_bitmap && anInfo.Data().MissingCount() > 0)
	{
		itsGrib->Message().MissingValue(anInfo.Data().MissingValue());
		itsGrib->Message().Bitmap(true);
	}

	//itsGrib->Message().BitsPerValue(16);

#if defined GRIB_WRITE_PACKED_DATA and defined HAVE_CUDA

	if (anInfo.Grid()->IsPackedData() && anInfo.Grid()->PackedData().ClassName() == "simple_packed")
	{
		itsLogger->Trace("Writing packed data");
		simple_packed* s = reinterpret_cast<simple_packed*> (anInfo.Grid()->PackedData());

		itsGrib->Message().ReferenceValue(s->coefficients.referenceValue);
		itsGrib->Message().BinaryScaleFactor(s->coefficients.binaryScaleFactor);
		itsGrib->Message().DecimalScaleFactor(s->coefficients.decimalScaleFactor);
		itsGrib->Message().BitsPerValue(s->coefficients.bitsPerValue);

		itsLogger->Trace("bits per value: " + boost::lexical_cast<string> (itsGrib->Message().BitsPerValue()));
		itsLogger->Trace("decimal scale factor: " + boost::lexical_cast<string> (itsGrib->Message().DecimalScaleFactor()));
		itsLogger->Trace("binary scale factor: " + boost::lexical_cast<string> (itsGrib->Message().BinaryScaleFactor()));
		itsLogger->Trace("reference value: " + boost::lexical_cast<string> (itsGrib->Message().ReferenceValue()));

		itsGrib->Message().PackedValues(s->data, anInfo.Data().Size(), 0, 0);
	}
	else
#endif
	{
		itsLogger->Trace("Writing unpacked data");

#ifdef DEBUG
		// Check that data is not NaN, otherwise grib_api will go to 
		// an eternal loop

		auto data = anInfo.Data().Values();
		bool foundNanValue = false;

		for (size_t i = 0; i < data.size(); i++)
		{
			double d = data[i];

			if (!isfinite(d))
			{
				foundNanValue = true;
				break;
			}
		}

		assert(!foundNanValue);
#endif

		itsGrib->Message().Values(anInfo.Data().ValuesAsPOD(), static_cast<long> (anInfo.Data().Size()));
	}

	if (edition == 2 && itsWriteOptions.packing_type == kJpegPacking)
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

	if (anInfo.Grid()->Type() == kReducedGaussian)
	{
		itsGrib->Message().ResolutionAndComponentFlags(0);
	}
	else
	{
		// TODO: check if parameter really is rotated uv
		if (edition == 1)
		{
			itsGrib->Message().ResolutionAndComponentFlags(128); // 10000000
		}
		else
		{
			itsGrib->Message().ResolutionAndComponentFlags(48); // 00110000
		}
	}

	vector<double> AB = anInfo.Grid()->AB();

	if (!AB.empty())
	{
		itsGrib->Message().NV(static_cast<long> (AB.size()));
		itsGrib->Message().PV(AB, AB.size());
	}

	if ((itsWriteOptions.configuration->FileCompression() == kGZIP || itsWriteOptions.configuration->FileCompression() == kBZIP2) && appendToFile)
	{
		itsLogger->Warning("Unable to append to a compressed file");
		appendToFile = false;
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

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool readContents, bool readPackedData, bool forceCaching) const
{

	shared_ptr<neons> n;
	shared_ptr<radon> r;

	if (options.configuration->DatabaseType() == kNeons || options.configuration->DatabaseType() == kNeonsAndRadon)
	{
		n = GET_PLUGIN(neons);
	}
	
	if (options.configuration->DatabaseType() == kRadon || options.configuration->DatabaseType() == kNeonsAndRadon)
	{
		r = GET_PLUGIN(radon);
	}

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
	aTimer->Start();

	while (itsGrib->NextMessage())
	{
	
		foundMessageNo++;
		bool dataIsValid = true;

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

			string parmName = "";

			if (options.configuration->DatabaseType() == kNeons || options.configuration->DatabaseType() == kNeonsAndRadon)
			{
				parmName = n->GribParameterName(number, no_vers, timeRangeIndicator);
			}

			if (parmName.empty() && (options.configuration->DatabaseType() == kRadon || options.configuration->DatabaseType() == kNeonsAndRadon))
			{
				auto parminfo = r->RadonDB().GetParameterFromGrib1(options.prod.Id(), no_vers, number, timeRangeIndicator,
				itsGrib->Message().NormalizedLevelType(), itsGrib->Message().LevelValue());

				if (parminfo.size())
				{
					parmName = parminfo["name"];
				}
			}

			if (parmName.empty())
			{
				itsLogger->Warning("Parameter name not found from Neons for no_vers: " +
							boost::lexical_cast<string> (no_vers) + ", number: " +
							boost::lexical_cast<string> (number) + ", timeRangeIndicator: " +
							boost::lexical_cast<string> (timeRangeIndicator));
			}
			else
			{
				p.Name(parmName);
			}
			
			p.GribParameter(number);
			p.GribTableVersion(no_vers);

		}
		else
		{
			long category = itsGrib->Message().ParameterCategory();
			long discipline = itsGrib->Message().ParameterDiscipline();
			long process = options.prod.Process();
	
			string parmName = "";

			if (options.configuration->DatabaseType() == kNeons || options.configuration->DatabaseType() == kNeonsAndRadon)
			{
				parmName = n->GribParameterName(number, category, discipline, process);
			}

			if (parmName.empty() && (options.configuration->DatabaseType() == kRadon || options.configuration->DatabaseType() == kNeonsAndRadon))
			{
				auto parminfo = r->RadonDB().GetParameterFromGrib2(options.prod.Id(), discipline, category, number,
				itsGrib->Message().NormalizedLevelType(), itsGrib->Message().LevelValue());

				if (parminfo.size())
				{
					parmName = parminfo["name"];
				}
			}

			if (parmName.empty())
			{
				itsLogger->Warning("Parameter name not found from database for discipline: " +
							boost::lexical_cast<string> (discipline) + ", category: " +
							boost::lexical_cast<string> (category) + ", number: " +
							boost::lexical_cast<string> (number));
			}
			else
			{
				p.Name(parmName);
			}

			p.GribParameter(number);
			p.GribDiscipline(discipline);
			p.GribCategory(category);

			if (p.Name() == "T-C" && options.prod.Centre() == 7)
			{
				p.Name("T-K");
			}
		}
		
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
		else if (unit == "hPa")
		{
			p.Unit(kHPa);
		}
		else
		{
			itsLogger->Trace("Unable to determine himan parameter unit for grib unit " + itsGrib->Message().ParameterUnit());
		}

		if (p != options.param)
		{
			itsLogger->Trace("Parameter does not match: " + options.param.Name() + " (requested) vs " + p.Name() + " (found)");
			
			if (forceCaching)
			{
				dataIsValid = false;
			}
			else
			{
				continue;
			}
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

		t.ValidDateTime().Adjust(timeResolution, static_cast<int> (step));
		
		if (t != options.time)
		{
			forecast_time optsTime(options.time);

			itsLogger->Trace("Times do not match");

			if (optsTime.OriginDateTime() != t.OriginDateTime())
			{
				itsLogger->Trace("OriginDateTime: " + optsTime.OriginDateTime().String() + " (requested) vs " + t.OriginDateTime().String() + " (found)");
			}

			if (optsTime.ValidDateTime() != t.ValidDateTime())
			{
				itsLogger->Trace("ValidDateTime: " + optsTime.ValidDateTime().String() + " (requested) vs " + t.ValidDateTime().String() + " (found)");
			}

			if (optsTime.StepResolution() != t.StepResolution())
			{
				itsLogger->Trace("Step resolution: " + string(HPTimeResolutionToString.at(optsTime.StepResolution())) + " (requested) vs " + string(HPTimeResolutionToString.at(t.StepResolution())) + " (found)");
			}

			if (forceCaching)
			{
				dataIsValid = false;
			}
			else
			{
				continue;
			}
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
			itsLogger->Trace("Level does not match");
			
			if (options.level.Type() != l.Type())
			{
				itsLogger->Trace("Type: " + string(HPLevelTypeToString.at(options.level.Type())) + " (requested) vs " + string(HPLevelTypeToString.at(l.Type())) + " (found)");
			}
			
			if (options.level.Value() != l.Value())
			{
				itsLogger->Trace("Value: " + string(boost::lexical_cast<string> (options.level.Value())) + " (requested) vs " + string(boost::lexical_cast<string> (l.Value())) + " (found)");
			}
			if (forceCaching)
			{
				dataIsValid = false;
			}
			else
			{
				continue;
			}
		}

		forecast_type ty(static_cast<HPForecastType> (itsGrib->Message().ForecastType()), itsGrib->Message().ForecastTypeValue());
		
		if (options.ftype.Type() != ty.Type() || options.ftype.Value() != ty.Value())
		{
			itsLogger->Trace("Forecast type does not match");
			
			if (options.ftype.Type() != ty.Type())
			{
				itsLogger->Trace("Type: " + string(HPForecastTypeToString.at(options.ftype.Type())) + " (requested) vs " + string(HPForecastTypeToString.at(ty.Type())) + " (found)");
			}
			
			if (options.ftype.Value() != ty.Value())
			{
				itsLogger->Trace("Value: " + string(boost::lexical_cast<string> (options.ftype.Value())) + " (requested) vs " + string(boost::lexical_cast<string> (ty.Value())) + " (found)");
			}
			
			if (forceCaching)
			{
				dataIsValid = false;
			}
			else
			{
				continue;
			}
		}
		
		// END VALIDATION OF SEARCH PARAMETERS

		shared_ptr<info> newInfo (new info());

		producer prod(centre, process);

		if (options.configuration->DatabaseType() == kNeons || options.configuration->DatabaseType() == kNeonsAndRadon)
		{
			// No easy way to get fmi producer id from from Neons based on centre/ident.
			// Use given producer id instead.

			prod.Id(options.prod.Id());
		}			
			
		if (prod.Id() == kHPMissingInt && (options.configuration->DatabaseType() == kRadon || options.configuration->DatabaseType() == kNeonsAndRadon))
		{
			// Do a double check and fetch the fmi producer id from database.

			long typeId = 1; // deterministic forecast, default
			long msgType = itsGrib->Message().ForecastType();
			
			if (msgType == 2) {
				typeId = 2; // ANALYSIS
			}
			else if (msgType == 3 || msgType == 4)
			{
				typeId = 3; // ENSEMBLE
			}

			auto prodInfo = r->RadonDB().GetProducerFromGrib(centre, process, typeId);
			
			if (!prodInfo.empty())
			{
				prod.Id(boost::lexical_cast<int> (prodInfo["id"]));
			}
		}

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

		vector<forecast_type> theForecastTypes;
		
		theForecastTypes.push_back(ty);
				
		newInfo->ForecastTypes(theForecastTypes);
		
		/*
		 * Get area information from grib.
		 */

		unique_ptr<grid> newGrid = ReadAreaAndGrid();

		assert(newGrid);
		
		std::vector<double> ab;

		if (levelType == himan::kHybrid)
		{
		 	long nv = itsGrib->Message().NV();
		 	long lev = itsGrib->Message().LevelValue();
			
			if (nv > 0)
			{
				ab = itsGrib->Message().PV(static_cast<size_t> (nv), static_cast<size_t> (lev));
			}
		}
		
		newGrid->AB(ab);

		newInfo->Create(newGrid.get());

		// Set descriptors

		newInfo->Param(p);
		newInfo->Time(t);
		newInfo->Level(l);
		newInfo->ForecastType(ty);

		/*
		 * Read data from grib *
		 */
		
		auto& dm = newInfo->Grid()->Data();

#if defined GRIB_READ_PACKED_DATA && defined HAVE_CUDA

		if (readPackedData && itsGrib->Message().PackingType() == "grid_simple")
		{
			// Get coefficient information

			double bsf = itsGrib->Message().BinaryScaleFactor();
			double dsf = itsGrib->Message().DecimalScaleFactor();
			double rv = itsGrib->Message().ReferenceValue();
			long bpv = itsGrib->Message().BitsPerValue();

			auto packed = unique_ptr<simple_packed> (new simple_packed(bpv, util::ToPower(bsf,2), util::ToPower(-dsf, 10), rv));
			
			// Get packed values from grib
			
			size_t len = itsGrib->Message().PackedValuesLength();
			unsigned char* data = 0;
			int* unpackedBitmap = 0;
	
			if (len > 0)
			{
				CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&data), len * sizeof(unsigned char)));
			
				itsGrib->Message().PackedValues(data);

				itsLogger->Trace("Retrieved " + boost::lexical_cast<string> (len) + " bytes of packed data from grib");
			}
			else
			{
				itsLogger->Warning("Grid is constant or empty");
			}

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

			packed->Set(data, len, static_cast<size_t> (itsGrib->Message().SizeX() * itsGrib->Message().SizeY()));

			newInfo->Grid()->PackedData(move(packed));

		}
		else
#endif
		if (readContents)
		{
			size_t len = itsGrib->Message().ValuesLength();

			double* d = itsGrib->Message().Values();

			dm.Set(d, len);

			free(d);

			itsLogger->Trace("Retrieved " + boost::lexical_cast<string> (len * 8) + " bytes of unpacked data from grib");

		}

		newInfo->Grid()->Data(dm);

		if (!dataIsValid && forceCaching)
		{
			// Put this data to cache now if the data has regular grid and the grid is equal
			// to wanted grid. This is a requirement since can't interpolate area here.
			
			if (options.configuration->UseCache() && 
					dynamic_pointer_cast<const plugin_configuration> (options.configuration)->Info()->Grid()->Class() == kRegularGrid &&
					dynamic_pointer_cast<const plugin_configuration> (options.configuration)->Info()->Grid()->Class() == newInfo->Grid()->Class() &&
					*dynamic_pointer_cast<const plugin_configuration> (options.configuration)->Info()->Grid() == *newInfo->Grid())
			{
				itsLogger->Trace("Force cache insert");

				auto c = GET_PLUGIN(cache);

				c->Insert(*newInfo);
			}

			continue;
		}
		
		infos.push_back(newInfo);
		newInfo->First();
		
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

unique_ptr<himan::grid> grib::ReadAreaAndGrid() const
{
		
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

	double X0 = itsGrib->Message().X0();
	double Y0 = itsGrib->Message().Y0();
	
	// GRIB2 has longitude 0 .. 360, but in neons we have it -180 .. 180
	// NB! ONLY FOR EC and FMI! GFS and GEM geometries are in grib2 format
	//
	// Make conversion to GRIB1 style coordinates, but in the long run we should figure out how to
	// handle grib 1 & grib 2 longitude values in a smart way. (a single geometry
	// can have coordinates in both ways!)

	long centre = itsGrib->Message().Centre();

	if (itsGrib->Message().Edition() == 2 && (centre == 98 || centre == 86) && X0 != 0)
	{
		X0 -= 360;
		if (X0 < -180) X0 += 360;
	}

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
		 * Normalize the first value to -180.
		 */

		assert(m == kBottomLeft || m == kTopLeft); // Want to make sure we always read from left to right

		firstPoint.X(-180.);
	}
		
	unique_ptr<grid> newGrid;
	
	switch (itsGrib->Message().NormalizedGridType())
	{
		case 0:
		{
			newGrid = unique_ptr<latitude_longitude_grid> (new latitude_longitude_grid);
			latitude_longitude_grid* const rg = dynamic_cast<latitude_longitude_grid*> (newGrid.get());

			size_t ni = static_cast<size_t> (itsGrib->Message().SizeX());
			size_t nj = static_cast<size_t> (itsGrib->Message().SizeY());
			
			rg->Ni(ni);
			rg->Nj(nj);

			rg->ScanningMode(m);
			
			pair<point,point> coordinates = util::CoordinatesFromFirstGridPoint(firstPoint, ni, nj, itsGrib->Message().iDirectionIncrement(),itsGrib->Message().jDirectionIncrement(), m);

			rg->BottomLeft(coordinates.first);
			rg->TopRight(coordinates.second);

			rg->Data(matrix<double> (ni, nj, 1, kFloatMissing));

			break;
		}
		case 4:
		{
			newGrid = unique_ptr<reduced_gaussian_grid> (new reduced_gaussian_grid);
			reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*> (newGrid.get());

			gg->N(static_cast<size_t> (itsGrib->Message().GetLongKey("N")));
			gg->NumberOfLongitudesAlongParallels(itsGrib->Message().PL());
			gg->Nj(static_cast<size_t> (itsGrib->Message().SizeY()));
			gg->ScanningMode(m);
			
			assert(m == kTopLeft);
			
			//pair<point,point> coordinates = util::CoordinatesFromFirstGridPoint(firstPoint, ni, nj, itsGrib->Message().iDirectionIncrement(),itsGrib->Message().jDirectionIncrement(), m);

			gg->TopLeft(firstPoint);
			
			point lastPoint(itsGrib->Message().X1(), itsGrib->Message().Y1());
			gg->BottomRight(lastPoint);

			break;
		}
		case 5:
		{
			newGrid = unique_ptr<stereographic_grid> (new stereographic_grid);
			stereographic_grid* const rg = dynamic_cast<stereographic_grid*> (newGrid.get());

			size_t ni = static_cast<size_t> (itsGrib->Message().SizeX());
			size_t nj = static_cast<size_t> (itsGrib->Message().SizeY());
			
			rg->Ni(ni);
			rg->Nj(nj);

			rg->Orientation(itsGrib->Message().GridOrientation());
			rg->Di(itsGrib->Message().XLengthInMeters());
			rg->Dj(itsGrib->Message().YLengthInMeters());
			
			/*
			 * Do not support stereographic projections but in bottom left scanning mode.
			 *
			 * The calculation of grid extremes could be done with f.ex. NFmiAzimuthalArea
			 * but lets not do that unless it's absolutely necessary.
			 */

			if (m != kBottomLeft)
			{
				throw runtime_error("Stereographic projection only supported when scanning mode is bottom left");
			}

			rg->ScanningMode(m);

			const point first(X0, Y0);

			rg->BottomLeft(first);

			std::pair<point, point> coordinates = util::CoordinatesFromFirstGridPoint(first, rg->Orientation(), ni, nj, rg->Di(), rg->Dj());

			rg->TopRight(coordinates.second);
			rg->Data(matrix<double> (ni, nj, 1, kFloatMissing));

			break;
		}
		
		case 10:
		{
			newGrid = unique_ptr<rotated_latitude_longitude_grid> (new rotated_latitude_longitude_grid);
			rotated_latitude_longitude_grid* const rg = dynamic_cast<rotated_latitude_longitude_grid*> (newGrid.get());

			size_t ni = static_cast<size_t> (itsGrib->Message().SizeX());
			size_t nj = static_cast<size_t> (itsGrib->Message().SizeY());

			rg->Ni(ni);
			rg->Nj(nj);

			rg->SouthPole(himan::point(itsGrib->Message().SouthPoleX(), itsGrib->Message().SouthPoleY()));
			rg->UVRelativeToGrid(itsGrib->Message().UVRelativeToGrid());

			rg->ScanningMode(m);
			
			pair<point,point> coordinates = util::CoordinatesFromFirstGridPoint(firstPoint, ni, nj, itsGrib->Message().iDirectionIncrement(),itsGrib->Message().jDirectionIncrement(), m);

			rg->BottomLeft(coordinates.first);
			rg->TopRight(coordinates.second);

			rg->Data(matrix<double> (ni, nj, 1, kFloatMissing));

			break;
		}
		default:
			throw runtime_error(ClassName() + ": Unsupported grid type: " + boost::lexical_cast<string> (itsGrib->Message().NormalizedGridType()));
			break;

	}

	return newGrid;
}

void grib::WriteAreaAndGrid(info& anInfo)
{
	long edition = itsGrib->Message().Edition();
	HPScanningMode scmode = kUnknownScanningMode;

	switch (anInfo.Grid()->Type())
	{
		case kLatitudeLongitude:
		{
			latitude_longitude_grid* const rg = dynamic_cast<latitude_longitude_grid*> (anInfo.Grid());

			himan::point firstGridPoint = rg->FirstPoint();
			himan::point lastGridPoint = rg->LastPoint();

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

			itsGrib->Message().iDirectionIncrement(rg->Di());
			itsGrib->Message().jDirectionIncrement(rg->Dj());

			itsGrib->Message().SizeX(static_cast<long> (rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long> (rg->Nj()));
	
			scmode = rg->ScanningMode();
			
			break;
		}

		case kRotatedLatitudeLongitude:
		{
			rotated_latitude_longitude_grid* const rg = dynamic_cast<rotated_latitude_longitude_grid*> (anInfo.Grid());

			himan::point firstGridPoint = rg->FirstPoint();
			himan::point lastGridPoint = rg->LastPoint();
			
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

			itsGrib->Message().SouthPoleX(rg->SouthPole().X());
			itsGrib->Message().SouthPoleY(rg->SouthPole().Y());

			itsGrib->Message().iDirectionIncrement(rg->Di());
			itsGrib->Message().jDirectionIncrement(rg->Dj());

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().UVRelativeToGrid(rg->UVRelativeToGrid());

			itsGrib->Message().SizeX(static_cast<long> (rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long> (rg->Nj()));
	
			scmode = rg->ScanningMode();
			
			break;
		}

		case kStereographic:
		{
			stereographic_grid* const rg = dynamic_cast<stereographic_grid*> (anInfo.Grid());

			long gridType = 5; // Grib 1
			
			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			itsGrib->Message().GridType(gridType);

			itsGrib->Message().X0(rg->BottomLeft().X());
			itsGrib->Message().Y0(rg->BottomLeft().Y());

			itsGrib->Message().GridOrientation(rg->Orientation());

			itsGrib->Message().XLengthInMeters(rg->Di());
			itsGrib->Message().YLengthInMeters(rg->Dj());
			
			itsGrib->Message().SizeX(static_cast<long> (rg->Ni()));
			itsGrib->Message().SizeY(static_cast<long> (rg->Nj()));
			
			scmode = rg->ScanningMode();
			
			break;
		}

		case kReducedGaussian:
		{
			reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*> (anInfo.Grid());

			//himan::point firstGridPoint = gg->FirstGridPoint();
			//himan::point lastGridPoint = gg->LastGridPoint();

			long gridType = 4; // Grib 1
			
			if (edition == 2)
			{
				gridType = itsGrib->Message().GridTypeToAnotherEdition(gridType, 2);
			}

			assert(gg->ScanningMode() == kTopLeft);

			itsGrib->Message().GridType(gridType);
			
			itsGrib->Message().X0(gg->BottomLeft().X());
			itsGrib->Message().Y0(gg->BottomLeft().Y());
			itsGrib->Message().X1(gg->TopRight().X());
			itsGrib->Message().Y1(gg->TopRight().Y());
				
			//itsGrib->Message().SizeX(static_cast<long> (65535));
			//itsGrib->Message().SizeY(static_cast<long> (gg->Nj()));
			itsGrib->Message().SetLongKey("iDirectionIncrement", 65535);
			itsGrib->Message().SetLongKey("numberOfPointsAlongAParallel", 65535);

			itsGrib->Message().SetLongKey("N", static_cast<long> (gg->N()));

			itsGrib->Message().PL(gg->NumberOfLongitudesAlongParallels());
			
			scmode = gg->ScanningMode();
			
			break;
		}
		default:
			throw runtime_error(ClassName() + ": invalid projection while writing grib: " + boost::lexical_cast<string> (anInfo.Grid()->Type()));
			break;
	}

	bool iNegative, jPositive;

	switch (scmode)
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
			throw runtime_error(ClassName() + ": Unknown scanning mode when writing grib");
			break;
	}

	itsGrib->Message().IScansNegatively(iNegative);
	itsGrib->Message().JScansPositively(jPositive);
}

void grib::WriteTime(info& anInfo)
{
	itsGrib->Message().DataDate(boost::lexical_cast<long> (anInfo.Time().OriginDateTime().String("%Y%m%d")));
	itsGrib->Message().DataTime(boost::lexical_cast<long> (anInfo.Time().OriginDateTime().String("%H%M")));

	double divisor = 1;
	long unitOfTimeRange = 1;

	if (anInfo.Producer().Id() == 210)
	{
		// Harmonie
		unitOfTimeRange = 13; // 15 minutes
		divisor = 15;
	}
	else if (anInfo.Time().Step() > 255) // Forecast with stepvalues that don't fit in one byte
	{
		long step = anInfo.Time().Step();

		if (step % 3 == 0 && step / 3 < 255)
		{
			unitOfTimeRange = 13; // 3 hours
			divisor = 3;
		}
		else if (step % 6 == 0 && step / 6 < 255)
		{
			unitOfTimeRange = 11; // 6 hours
			divisor = 6;
		}
		else if (step % 12 == 0 && step / 12 < 255)
		{
			unitOfTimeRange = 12; // 12 hours
			divisor = 12;
		}
		else
		{
			itsLogger->Fatal("Step too large, unable to continue");
			exit(1);
		}
	}

	long period = itsWriteOptions.configuration->ForecastStep();

	if (anInfo.Param().Aggregation().TimeResolution() != kUnknownTimeResolution)
	{
		period = anInfo.Param().Aggregation().TimeResolutionValue();

		// Time range and aggregation need to share a common time unit
		if (anInfo.Param().Aggregation().TimeResolution() == kHourResolution && (unitOfTimeRange == 0 || unitOfTimeRange == 13))
		{
			period *= 60;
		}
	}

	if (itsGrib->Message().Edition() == 1)
	{
		itsGrib->Message().UnitOfTimeRange(unitOfTimeRange);

		long p1 = static_cast<long> ((anInfo.Time().Step() - period) / divisor);

		switch (anInfo.Param().Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				// Forecast product valid for reference time + P1 (P1 > 0)
				itsGrib->Message().TimeRangeIndicator(0);
				itsGrib->Message().P1(static_cast<int> (anInfo.Time().Step() / divisor));
				break;
			case kAverage:
				// Average (reference time + P1 to reference time + P2)
				itsGrib->Message().TimeRangeIndicator(3);

				if (p1 < 0)
				{
					itsLogger->Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}
			
				itsGrib->Message().P1(p1);
				itsGrib->Message().P2(static_cast<long> (anInfo.Time().Step() / divisor));
				break;
			case kAccumulation:
				// Accumulation (reference time + P1 to reference time + P2) product considered valid at reference time + P2
				itsGrib->Message().TimeRangeIndicator(4);			

				if (p1 < 0)
				{
					itsLogger->Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}
				itsGrib->Message().P1(p1);
				itsGrib->Message().P2(static_cast<long> (anInfo.Time().Step() / divisor));
				break;
			case kDifference:
				// Difference (reference time + P2 minus reference time + P1) product considered valid at reference time + P2
				itsGrib->Message().TimeRangeIndicator(5);

				if (p1 < 0)
				{
					itsLogger->Warning("Forcing starting step from negative value to zero");
					p1 = 0;
				}

				itsGrib->Message().P1(p1);
				itsGrib->Message().P2(static_cast<long> (anInfo.Time().Step() / divisor));
				break;
		}

		assert(itsGrib->Message().TimeRangeIndicator() != 10);
	}
	else
	{
		if (unitOfTimeRange == 13)
		{
			unitOfTimeRange = 254;
		}

		itsGrib->Message().UnitOfTimeRange(unitOfTimeRange);

		// Statistical processing is set in WriteParameter()
		switch (anInfo.Param().Aggregation().Type())
		{
			default:
			case kUnknownAggregationType:
				itsGrib->Message().ForecastTime(anInfo.Time().Step());
				break;
			case kAverage:
			case kAccumulation:
			case kDifference:
				itsGrib->Message().SetLongKey("indicatorOfUnitForTimeRange", unitOfTimeRange);
				itsGrib->Message().ForecastTime(static_cast<long> ((anInfo.Time().Step() - period) / divisor)); // start step
				itsGrib->Message().LengthOfTimeRange(static_cast<long> (itsWriteOptions.configuration->ForecastStep() / divisor)); // step length
				break;
		}
	}
}

void grib::WriteParameter(info& anInfo)
{

	if (itsGrib->Message().Edition() == 1)
	{
		if (anInfo.Param().GribTableVersion() != kHPMissingInt && anInfo.Param().GribIndicatorOfParameter() != kHPMissingInt)
		{
			// In radon table version is a parameter property, not a 
			// producer property
		
			itsGrib->Message().Table2Version(anInfo.Param().GribTableVersion());
			itsGrib->Message().ParameterNumber(anInfo.Param().GribIndicatorOfParameter());
		}
		else
		{
			long parm_id = anInfo.Param().GribIndicatorOfParameter();

			if (itsWriteOptions.configuration->DatabaseType() == kNeons || itsWriteOptions.configuration->DatabaseType() == kNeonsAndRadon)
			{
				// In neons table version is a producer property

				long tableVersion = anInfo.Producer().TableVersion();

				auto n = GET_PLUGIN(neons);
				
				if (tableVersion == kHPMissingInt)
				{
					auto prodinfo = n->NeonsDB().GetProducerDefinition(anInfo.Producer().Id());
					
					if (prodinfo.empty())
					{
						itsLogger->Warning("Producer information not found from neons for producer " + boost::lexical_cast<string> (anInfo.Producer().Id()));
						itsLogger->Warning("Setting table2version to 203");	
						tableVersion = 203;
					}
					else
					{
						tableVersion = boost::lexical_cast<long> (prodinfo["no_vers"]);
					}
				}

				if (parm_id == kHPMissingInt)
				{
					parm_id = n->NeonsDB().GetGridParameterId(tableVersion, anInfo.Param().Name());
				}
				
				if (parm_id == -1)
				{
					itsLogger->Warning("Parameter " + anInfo.Param().Name() + " does not have mapping for code table " + boost::lexical_cast<string> (anInfo.Producer().TableVersion()) + " in neons");
					itsLogger->Warning("Setting parameter to 1");
					parm_id = 1;
				}

				itsGrib->Message().ParameterNumber(parm_id);
				itsGrib->Message().Table2Version(tableVersion);
			}
			
			if (itsWriteOptions.configuration->DatabaseType() == kRadon || (itsWriteOptions.configuration->DatabaseType() == kNeonsAndRadon && parm_id == kHPMissingInt))
			{
				auto r = GET_PLUGIN(radon);
				
				auto paramInfo = r->RadonDB().GetParameterFromDatabaseName(anInfo.Producer().Id(), anInfo.Param().Name());
				
				if (paramInfo.empty())
				{
					itsLogger->Warning("Parameter " + anInfo.Param().Name() + " does not have mapping for producer " + boost::lexical_cast<string> (anInfo.Producer().Id()) + " in radon");
					itsLogger->Warning("Setting table2version to 203");
					itsGrib->Message().Table2Version(203);
				}
				else
				{
					itsGrib->Message().ParameterNumber(boost::lexical_cast<long> (paramInfo["grib1_number"]));
					itsGrib->Message().Table2Version(boost::lexical_cast<long> (paramInfo["grib1_table_version"]));
				}
			}
		}		
	}
	else if (itsGrib->Message().Edition() == 2)
	{
		itsGrib->Message().ParameterNumber(anInfo.Param().GribParameter());
		itsGrib->Message().ParameterCategory(anInfo.Param().GribCategory());
		itsGrib->Message().ParameterDiscipline(anInfo.Param().GribDiscipline()) ;

		if (anInfo.Param().Aggregation().Type() != kUnknownAggregationType)
		{
			itsGrib->Message().ProductDefinitionTemplateNumber(8);

			long type;

			switch (anInfo.Param().Aggregation().Type())
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
