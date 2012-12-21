/*
 * grib.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "grib.h"
#include "logger_factory.h"

using namespace std;
using namespace hilpee::plugin;

grib::grib()
{

	itsLogger = logger_factory::Instance()->GetLog("grib");

	itsGrib = shared_ptr<NFmiGrib> (new NFmiGrib());
}

shared_ptr<NFmiGrib> grib::Reader()
{
	return itsGrib;
}

bool grib::ToFile(shared_ptr<info> theInfo, const string& theOutputFile, HPFileType theFileType, bool theActiveOnly)
{

	// Write only that data which is currently set at descriptors

	if (theActiveOnly)
	{

		/* Section 0 */

		itsGrib->Message()->Edition(static_cast<int> (theFileType));

		if (theFileType == kGRIB2)
		{
			itsGrib->Message()->ParameterDiscipline(theInfo->Param()->GribDiscipline()) ;
		}

		/* Section 1 */

		itsGrib->Message()->Centre(86);

		if (theFileType == kGRIB2)
		{
			// Origin time
			itsGrib->Message()->Year(theInfo->Time()->OriginDateTime()->String("%Y"));
			itsGrib->Message()->Month(theInfo->Time()->OriginDateTime()->String("%m"));
			itsGrib->Message()->Day(theInfo->Time()->OriginDateTime()->String("%d"));
			itsGrib->Message()->Hour(theInfo->Time()->OriginDateTime()->String("%H"));
			itsGrib->Message()->Minute(theInfo->Time()->OriginDateTime()->String("%M"));
			itsGrib->Message()->Second("0");
		}

		itsGrib->Message()->StartStep(theInfo->Time()->Step());
		itsGrib->Message()->EndStep(theInfo->Time()->Step());

		/* Section 4 */

		if (theFileType == kGRIB2)
		{
			itsGrib->Message()->ParameterCategory(theInfo->Param()->GribCategory()) ;
			itsGrib->Message()->ParameterNumber(theInfo->Param()->GribParameter()) ;
		}

		// TODO: need to normalize these, now they are grib2

		switch (theInfo->Projection())
		{
			case kLatLonProjection:
				itsGrib->Message()->GridType(0);
				itsGrib->Message()->X0(theInfo->BottomLeftLongitude());
				itsGrib->Message()->Y0(theInfo->BottomLeftLatitude());
				itsGrib->Message()->X1(theInfo->TopRightLongitude());
				itsGrib->Message()->Y1(theInfo->TopRightLatitude());
				break;

			case kStereographicProjection:
				itsGrib->Message()->GridType(20);
				itsGrib->Message()->X0(theInfo->BottomLeftLongitude());
				itsGrib->Message()->Y0(theInfo->BottomLeftLatitude());

				// missing iDirectionIncrementInMeters
				itsGrib->Message()->GridOrientation(theInfo->Orientation());
				break;

			default:
				throw runtime_error(ClassName() + ": invalid projection while writing grib: " + boost::lexical_cast<string> (theInfo->Projection()));
				break;
		}

		itsGrib->Message()->SizeX(theInfo->Ni());
		itsGrib->Message()->SizeY(theInfo->Nj());

		itsGrib->Message()->Bitmap(true);

		itsGrib->Message()->Values(theInfo->Data()->Values(), theInfo->Ni() * theInfo->Nj());

		itsGrib->Message()->PackingType("grid_jpeg");
		itsGrib->Message()->Write(theOutputFile);

	}

	return true;

}

vector<shared_ptr<hilpee::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool theReadContents)
{

	vector<shared_ptr<hilpee::info>> theInfos;

	itsGrib->Open(theInputFile);

	itsLogger->Trace("Reading file '" + theInputFile + "'");

	while (itsGrib->NextMessage())
	{

		/*
		 * One grib file may contain many grib messages. Loop though all messages
		 * and get all that match our search options.
		 *
		 * \todo Should we actually return all matching messages or only the first one
		 */

		unsigned long producer = itsGrib->Message()->Process();

		if (options.configuration.SourceProducer() != producer)
		{
			itsLogger->Trace("Producer does not match: " + boost::lexical_cast<string> (options.configuration.SourceProducer()) + " vs " + boost::lexical_cast<string> (producer));
			continue;
		}

		shared_ptr<param> p (new param());

		if (itsGrib->Message()->Edition() == 1)
		{
			/*
			if (itsGrib->Message()->ParameterNumber() != options.param.GribParameter())
				continue;

			p->GribParameter(itsGrib->Message()->ParameterNumber());
			*/
			throw runtime_error(ClassName() + ": grib 1 not supported yet");
		}
		else
		{

			long number = itsGrib->Message()->ParameterNumber();
			long category = itsGrib->Message()->ParameterCategory();
			long discipline = itsGrib->Message()->ParameterDiscipline();

			// Need to get name and unit of parameter

#ifdef NEONS
			FFFFFFFFFFFFFFFUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU!!!!!!!!!!!!!!!!!!!
#else

			assert(discipline == 0);

			if (number == 1 && category == 3)
			{
				p->Name("P-HPA");
				p->Unit(kPa);
			}
			else if (number == 0 && category == 0)
			{
				p->Name("T-K");
				p->Unit(kK);
			}
			else if (number == 8 && category == 2)
			{
				p->Name("VV-PAS");
				p->Unit(kPas);
			}
			else
			{
				throw runtime_error(ClassName() + ": I do not recognize this parameter (and I can't connect to neons)");
			}
#endif

			// Name is our primary identifier -- not univ_id or grib param id

			if (*p != options.param)
			{
				itsLogger->Trace("Parameter does not match: " + options.param.Name() + " vs " + p->Name());
				continue;
			}

			p->GribParameter(number);
			p->GribDiscipline(discipline);
			p->GribCategory(category);
		}

		string dataDate = boost::lexical_cast<string> (itsGrib->Message()->DataDate());
		string dataTime = boost::lexical_cast<string> (itsGrib->Message()->DataTime());

		long step = itsGrib->Message()->EndStep();

		// grib_api stores times as long, so when origin hour is 00
		// it gets stored as 0 which boost does not understand
		// (since it's a mix of 12 hour and 24 hour times)

		if (step < 10)
		{
			dataTime = "0" + dataTime;
		}

		string originDateTimeStr = dataDate + dataTime;

		raw_time originDateTime (originDateTimeStr, "%Y%m%d%H");

		std::shared_ptr<forecast_time> t (new forecast_time(originDateTime, originDateTime));
		t->ValidDateTime()->Adjust("hours", step);

		if (*t != options.time)
		{
			itsLogger->Trace("Times do not match");
			itsLogger->Trace("OriginDateTime: " + options.time.OriginDateTime()->String() + " vs " + t->OriginDateTime()->String());
			itsLogger->Trace("ValidDateTime: " + options.time.ValidDateTime()->String() + " vs " + t->ValidDateTime()->String());
			continue;
		}

		long gribLevel = itsGrib->Message()->NormalizedLevelType();

		hilpee::HPLevelType levelType;

		switch (gribLevel)
		{
			case 1:
				levelType = hilpee::kGround;
				break;

			case 100:
				levelType = hilpee::kPressure;
				break;

			case 102:
				levelType = hilpee::kMeanSea;
				break;

			case 105:
				levelType = hilpee::kHeight;
				break;

			case 109:
				levelType = hilpee::kHybrid;
				break;

			default:
				throw runtime_error(ClassName() + ": Unsupported level type: " + boost::lexical_cast<string> (gribLevel));

		}

		shared_ptr<level> l (new level(levelType, itsGrib->Message()->LevelValue()));

		if (*l != options.level)
		{
			itsLogger->Trace("Level does not match");
			continue;
		}

		// END VALIDATION OF SEARCH PARAMETERS

		shared_ptr<info> newInfo (new info());

		newInfo->Producer(producer);

		vector<shared_ptr<param>> theParams;

		theParams.push_back(p);

		newInfo->Params(theParams);

		if (newInfo->Producer() == 96)
		{
			//newInfo->Producer(53);    // VALIAIKAINEN (miten ratkaistaan tama oikeasti:
		}

		// fmi-tuottaja-id != grib-process-id ainakin grib2 kohdalla

		vector<shared_ptr<forecast_time>> theTimes;

		theTimes.push_back(t);

		newInfo->Times(theTimes);

		vector<shared_ptr<level> > theLevels;

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
				break;

			default:
				throw runtime_error(ClassName() + ": Unsupported projection: " + boost::lexical_cast<string> (itsGrib->Message()->NormalizedGridType()));
				break;

		}

		newInfo->BottomLeftLatitude(itsGrib->Message()->Y0());
		newInfo->BottomLeftLongitude(itsGrib->Message()->X0());

		// Assume +x+y

		newInfo->TopRightLatitude(itsGrib->Message()->Y1());
		newInfo->TopRightLongitude(itsGrib->Message()->X1());


		size_t ni = itsGrib->Message()->SizeX();
		size_t nj = itsGrib->Message()->SizeY();


		/*
		 * Read data from grib *
		 */

		size_t len = 0;
		double* d = 0;

		if (theReadContents)
		{
			len = itsGrib->Message()->ValuesLength();

			// d = new double[len];

			d = itsGrib->Message()->Values();
			//theNewInfo->DataLength(itsGrib->Message()->ValuesLength());
			//theNewInfo->DataValues(itsGrib->Message()->Values());
			itsLogger->Debug("Read data from file '" + theInputFile + "'");
		}

		newInfo->Create();

		// Set descriptors

		newInfo->Param(p);
		newInfo->Time(t);
		newInfo->Level(l);

		newInfo->ResetLocation();

		size_t i = 0;

		while (newInfo->NextLocation())
		{

			if (!newInfo->Value(d[i]))
			{
				throw runtime_error(ClassName() + ": Unable to set value to querydata");
			}

			i++;
		}

		shared_ptr<d_matrix_t> dm = shared_ptr<d_matrix_t> (new d_matrix_t(ni, nj));

		dm->Data(d, len);

		newInfo->Data(dm);

		theInfos.push_back(newInfo);

		if (d)
		{
			delete d;
		}
	}


	return theInfos;
}
