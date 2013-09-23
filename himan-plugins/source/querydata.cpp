/**
 * @file querydata.cpp
 *
 * @date Nov 27, 2012
 * @author: partio
 */


#include "querydata.h"
#include "logger_factory.h"
#include <fstream>
#include <NFmiQueryData.h>
#include <NFmiTimeList.h>
#include <NFmiLatLonArea.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiGrid.h>
#include <NFmiQueryDataUtil.h>

using namespace std;
using namespace himan::plugin;

querydata::querydata()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("querydata"));
}

bool querydata::ToFile(shared_ptr<info> theInfo, const string& theOutputFile, HPFileWriteOption fileWriteOption)
{

    ofstream out(theOutputFile.c_str());

    /*
     * Create required descriptors
     */

	bool activeOnly = true;

	if (fileWriteOption == kSingleFile)
	{
		activeOnly = false;
	}

    NFmiParamDescriptor pdesc = CreateParamDescriptor(theInfo, activeOnly);
    NFmiTimeDescriptor tdesc = CreateTimeDescriptor(theInfo, activeOnly);
    NFmiHPlaceDescriptor hdesc = CreateHPlaceDescriptor(theInfo);
    NFmiVPlaceDescriptor vdesc = CreateVPlaceDescriptor(theInfo, activeOnly);

    assert(pdesc.Size());
    assert(tdesc.Size());
    assert(hdesc.Size());
    assert(vdesc.Size());

    if (pdesc.Size() == 0)
    {
        itsLogger->Error("No valid parameters found");
        return false;
    }
    else if (tdesc.Size() == 0)
    {
        itsLogger->Error("No valid times found");
        return false;
    }
    else if (hdesc.Size() == 0)
    {
        itsLogger->Error("No valid times found");
        return false;
    }
    else if (vdesc.Size() == 0)
    {
        itsLogger->Error("No valid times found");
        return false;
    }

    NFmiFastQueryInfo qi(pdesc, tdesc, hdesc, vdesc);

    unique_ptr<NFmiQueryData> qdata (NFmiQueryDataUtil::CreateEmptyData(qi));

    NFmiFastQueryInfo qinfo = qdata.get();

    qinfo.SetProducer(NFmiProducer(theInfo->Producer().Id(), theInfo->Producer().Name()));

    /*
     * At the same time check that we have only constant-sized grids
     */

#ifndef NDEBUG
    size_t size = 0;
    string firstOriginTime;

    bool first = true;
#endif

    if (fileWriteOption == kNeons || fileWriteOption == kMultipleFiles)
    {
        // Should user Param(arg) Level(arg) Time(arg) here !
        qinfo.FirstParam();
        qinfo.FirstLevel();
        qinfo.FirstTime();

        theInfo->ResetLocation();
        qinfo.ResetLocation();

        assert(theInfo->Data()->Size() == qinfo.Size());

        while (theInfo->NextLocation() && qinfo.NextLocation())
        {
            qinfo.FloatValue(static_cast<float> (theInfo->Value()));
        }
    }
    else
    {
    	/*
    	 * info-class (like grib) support many different grid types, the "worst" case being that
    	 * all info-class grid elements are different from each other (different producer,grid,area,time etc).
    	 * querydata on the other does not support this, so we should check that all elements are equal
    	 * before writing querydata.
    	 */

        theInfo->ResetTime();
        qinfo.ResetTime();

        while (theInfo->NextTime() && qinfo.NextTime())
        {

#ifndef NDEBUG

        	if (first)
        	{
        		firstOriginTime = theInfo->Time().OriginDateTime()->String();
        	}
        	else
        	{
        		assert(firstOriginTime == theInfo->Time().OriginDateTime()->String());
        	}

#endif
        	theInfo->ResetLevel();
            qinfo.ResetLevel();

            while (theInfo->NextLevel() && qinfo.NextLevel())
            {
                theInfo->ResetParam();
                qinfo.ResetParam();

                while (theInfo->NextParam() && qinfo.NextParam())
                {
                    theInfo->ResetLocation();
                    qinfo.ResetLocation();

#ifndef NDEBUG

                    if (first)
                    {
                        first = false;
                        size = theInfo->Data()->Size();
                    }
                    else
                    {
                        assert(theInfo->Data()->Size() == size);
                    }

#endif

                    while (theInfo->NextLocation() && qinfo.NextLocation())
                    {
                        qinfo.FloatValue(static_cast<float> (theInfo->Value()));
                    }
                }
            }
        }
    }

    out << *qdata;

    itsLogger->Info("Wrote file '" + theOutputFile + "'");

    return true;

}

NFmiTimeDescriptor querydata::CreateTimeDescriptor(shared_ptr<info> info, bool theActiveOnly)
{

    /*
     * Create time descriptor
     */

    NFmiTimeList tlist;

    if (theActiveOnly)
    {
        tlist.Add(new NFmiMetTime(boost::lexical_cast<long> (info->Time().ValidDateTime()->String("%Y%m%d")),
                                  boost::lexical_cast<long> (info->Time().ValidDateTime()->String("%H%M"))));
    }
    else
    {
        info->ResetTime();

        while (info->NextTime())
        {
            tlist.Add(new NFmiMetTime(boost::lexical_cast<long> (info->Time().ValidDateTime()->String("%Y%m%d")),
                                      boost::lexical_cast<long> (info->Time().ValidDateTime()->String("%H%M"))));
        }

    }

    return NFmiTimeDescriptor(tlist.FirstTime(), tlist);

}


NFmiParamDescriptor querydata::CreateParamDescriptor(shared_ptr<info> info, bool theActiveOnly)
{


    /*
     * Create parameter descriptor
     */

    NFmiParamBag pbag;

    if (theActiveOnly)
    {
        assert(info->Param().UnivId());

        pbag.Add(NFmiDataIdent(NFmiParam(info->Param().UnivId(), info->Param().Name())));

    }
    else
    {
        info->ResetParam();

        while (info->NextParam())
        {

            assert(info->Param().UnivId());

            pbag.Add(NFmiDataIdent(NFmiParam(info->Param().UnivId(), info->Param().Name())));

        }
    }

    return NFmiParamDescriptor(pbag);


}

NFmiHPlaceDescriptor querydata::CreateHPlaceDescriptor(shared_ptr<info> info)
{

    NFmiArea* theArea = 0;

    // Assume all grids in the info have equal projections

    switch (info->Grid()->Projection())
    {
    case kLatLonProjection:
    {
        theArea = new NFmiLatLonArea(NFmiPoint(info->Grid()->BottomLeft().X(), info->Grid()->BottomLeft().Y()),
                                     NFmiPoint(info->Grid()->TopRight().X(), info->Grid()->TopRight().Y()));

        break;
    }

    case kRotatedLatLonProjection:
    {
        theArea = new NFmiRotatedLatLonArea(NFmiPoint(info->Grid()->BottomLeft().X(), info->Grid()->BottomLeft().Y()),
                                            NFmiPoint(info->Grid()->TopRight().X(), info->Grid()->TopRight().Y()),
                                            NFmiPoint(info->Grid()->SouthPole().X(), info->Grid()->SouthPole().Y()),
                                            NFmiPoint(0.,0.), // default values
                                            NFmiPoint(1.,1.), // default values
                                            true);

        break;
    }

    case kStereographicProjection:
    {
		theArea = new NFmiStereographicArea(NFmiPoint(info->Grid()->BottomLeft().X(), info->Grid()->BottomLeft().Y()),
											info->Grid()->Di() * static_cast<double> ((info->Grid()->Ni()-1)),
											info->Grid()->Dj() * static_cast<double> ((info->Grid()->Nj()-1)),
											info->Grid()->Orientation());

        break;

    }

    default:
        itsLogger->Error("No supported projection found");
        return NFmiHPlaceDescriptor();
        break;
    }

    NFmiGrid theGrid (theArea, info->Grid()->Ni(), info->Grid()->Nj());

    delete theArea;

    return NFmiHPlaceDescriptor(theGrid);

}

NFmiVPlaceDescriptor querydata::CreateVPlaceDescriptor(shared_ptr<info> info, bool theActiveOnly)
{

    NFmiLevelBag lbag;

    if (theActiveOnly)
    {
        lbag.AddLevel(NFmiLevel(info->Level().Type(), "Hipihipi", info->Level().Value()));
    }
    else
    {

        info->ResetLevel();

        while (info->NextLevel())
        {
            lbag.AddLevel(NFmiLevel(info->Level().Type(), "Hipihipi", info->Level().Value()));
        }
    }

    return NFmiVPlaceDescriptor(lbag);

}

shared_ptr<himan::info> querydata::FromFile(const string& inputFile, const search_options& options, bool readContents)
{
	itsLogger->Fatal("Function FromFile() not implemented yet");
	exit(1);
}
