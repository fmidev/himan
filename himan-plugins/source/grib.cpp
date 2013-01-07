/*
 * grib.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "grib.h"
#include "logger_factory.h"

using namespace std;
using namespace himan::plugin;

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

    // Write only that data which is currently set at descriptors

    if (theActiveOnly)
    {

        /* Section 0 */

        itsGrib->Message()->Edition(static_cast<int> (theFileType));

        if (theFileType == kGRIB2)
        {
            itsGrib->Message()->ParameterDiscipline(theInfo->Param().GribDiscipline()) ;
        }

        /* Section 1 */

        itsGrib->Message()->Centre(theInfo->Producer().Centre());
        itsGrib->Message()->Process(theInfo->Producer().Process());

        if (theFileType == kGRIB2)
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

        if (theFileType == kGRIB2)
        {
            itsGrib->Message()->ParameterCategory(theInfo->Param().GribCategory()) ;
            itsGrib->Message()->ParameterNumber(theInfo->Param().GribParameter()) ;
        }

        // TODO: need to normalize these, now they are grib2

        switch (theInfo->Projection())
        {
        case kLatLonProjection:
        {
            itsGrib->Message()->GridType(0);

            string scanningMode = "+x-y"; // GFS

            double latitudeOfFirstGridPointInDegrees, longitudeOfFirstGridPointInDegrees;
            double latitudeOfLastGridPointInDegrees, longitudeOfLastGridPointInDegrees;

            if (scanningMode == "+x-y")
            {
                latitudeOfFirstGridPointInDegrees = theInfo->TopRightLatitude();
                longitudeOfFirstGridPointInDegrees = theInfo->BottomLeftLongitude();

                latitudeOfLastGridPointInDegrees = theInfo->BottomLeftLatitude();
                longitudeOfLastGridPointInDegrees = theInfo->TopRightLongitude();
            }
            else
            {
                throw runtime_error(ClassName() + ": unsupported scanning mode: " + scanningMode);
            }

            itsGrib->Message()->X0(longitudeOfFirstGridPointInDegrees);
            itsGrib->Message()->Y0(latitudeOfFirstGridPointInDegrees);
            itsGrib->Message()->X1(longitudeOfLastGridPointInDegrees);
            itsGrib->Message()->Y1(latitudeOfLastGridPointInDegrees);
            break;
        }
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

        // Level

        if (theFileType == kGRIB2)
        {
            itsGrib->Message()->LevelType(theInfo->Level().Type());
            itsGrib->Message()->LevelValue(static_cast<long> (theInfo->Level().Value()));
        }

        itsGrib->Message()->Bitmap(true);

        itsGrib->Message()->Values(theInfo->Data()->Values(), theInfo->Ni() * theInfo->Nj());

        itsGrib->Message()->PackingType("grid_jpeg");
        itsGrib->Message()->Write(theOutputFile);

    }

    return true;

}

vector<shared_ptr<himan::info>> grib::FromFile(const string& theInputFile, const search_options& options, bool theReadContents)
{

    vector<shared_ptr<himan::info>> theInfos;

    itsGrib->Open(theInputFile);

    itsLogger->Trace("Reading file '" + theInputFile + "'");

    int foundMessageNo = -1;

    while (itsGrib->NextMessage())
    {

        foundMessageNo++;

        /*
         * One grib file may contain many grib messages. Loop though all messages
         * and get all that match our search options.
         *
         */

        //<!todo Should we actually return all matching messages or only the first one

        long process = itsGrib->Message()->Process();

        //<!todo How to best match neons producer id to grib centre/process

        if (options.configuration->SourceProducer() != process)
        {
            itsLogger->Trace("Producer does not match: " + boost::lexical_cast<string> (options.configuration->SourceProducer()) + " vs " + boost::lexical_cast<string> (process));
            // continue;
        }

        param p;

        //<! todo GRIB1 support

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
            throw runtime_error("FFFFFFFFFFFFFFFUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU!!!!!!!!!!!!!!!!!!!");
#else

            assert(discipline == 0);

            if (number == 1 && category == 3)
            {
                p.Name("P-HPA");
                p.Unit(kPa);
            }
            else if (number == 0 && category == 0)
            {
                p.Name("T-K");
                p.Unit(kK);
            }
            else if (number == 8 && category == 2)
            {
                p.Name("VV-PAS");
                p.Unit(kPas);
            }
            else
            {
                throw runtime_error(ClassName() + ": I do not recognize this parameter (and I can't connect to neons)");
            }

#endif

            // Name is our primary identifier -- not univ_id or grib param id

            if (p != options.param)
            {
                itsLogger->Trace("Parameter does not match: " + options.param.Name() + " vs " + p.Name());
                continue;
            }

            p.GribParameter(number);
            p.GribDiscipline(discipline);
            p.GribCategory(category);
        }

        string dataDate = boost::lexical_cast<string> (itsGrib->Message()->DataDate());

        long dt = itsGrib->Message()->DataTime();

        // grib_api stores times as long, so when origin hour is 00
        // it gets stored as 0 which boost does not understand
        // (since it's a mix of 12 hour and 24 hour times)

        string dataTime = boost::lexical_cast<string> (dt);

        if (dt < 10)
        {
            dataTime = "0" + dataTime;
        }

        long step = itsGrib->Message()->ForecastTime();

        string originDateTimeStr = dataDate + dataTime;

        raw_time originDateTime (originDateTimeStr, "%Y%m%d%H");

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

        dm->Data(d, len);

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
        itsLogger->Trace("Data Found from message " + boost::lexical_cast<string> (foundMessageNo) + "/" + boost::lexical_cast<string> (itsGrib->MessageCount()));
    }

    return theInfos;
}
