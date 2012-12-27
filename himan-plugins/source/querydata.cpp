/*
 * querydata.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: partio
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

bool querydata::ToFile(shared_ptr<info> theInfo, const string& theOutputFile, bool theActiveOnly)
{

	ofstream out(theOutputFile.c_str());

	/*
	 * Create required descriptors
	 */

	NFmiParamDescriptor pdesc = CreateParamDescriptor(theInfo, theActiveOnly);
	NFmiTimeDescriptor tdesc = CreateTimeDescriptor(theInfo, theActiveOnly);
	NFmiHPlaceDescriptor hdesc = CreateHPlaceDescriptor(theInfo);
	NFmiVPlaceDescriptor vdesc = CreateVPlaceDescriptor(theInfo, theActiveOnly);

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

	if (!qdata.get())
	{
		throw runtime_error(ClassName() + ": Failed to create querydata");
	}

	if (!qdata->Init())
	{
		throw runtime_error(ClassName() + ": Failed to init querydata");
	}


	NFmiFastQueryInfo qinfo = (qdata.get());

	qinfo.SetProducer(NFmiProducer(theInfo->Producer(), "Himan"));

	/*
	 * At the same time check that we have only constant-sized grids
	 */

#ifndef NDEBUG
	size_t size = 0;

	bool first = true;
#endif

	if (theActiveOnly)
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
		theInfo->ResetTime();
		qinfo.ResetTime();

		while (theInfo->NextTime() && qinfo.NextTime())
		{
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
		tlist.Add(new NFmiMetTime(boost::lexical_cast<long> (info->Time()->ValidDateTime()->String("%Y%m%d")),
		                          boost::lexical_cast<long> (info->Time()->ValidDateTime()->String("%H%M"))));
	}
	else
	{
		info->ResetTime();

		while (info->NextTime())
		{
			tlist.Add(new NFmiMetTime(boost::lexical_cast<long> (info->Time()->ValidDateTime()->String("%Y%m%d")),
			                          boost::lexical_cast<long> (info->Time()->ValidDateTime()->String("%H%M"))));
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
		assert(info->Param()->UnivId());

		pbag.Add(NFmiDataIdent(NFmiParam(info->Param()->UnivId(), info->Param()->Name())));

	}
	else
	{
		info->ResetParam();

		while (info->NextParam())
		{

			assert(info->Param()->UnivId());

			pbag.Add(NFmiDataIdent(NFmiParam(info->Param()->UnivId(), info->Param()->Name())));

		}
	}

	return NFmiParamDescriptor(pbag);


}

NFmiHPlaceDescriptor querydata::CreateHPlaceDescriptor(shared_ptr<info> info)
{

	NFmiArea* theArea = 0;

	switch (info->Projection())
	{
		case kLatLonProjection:
			{
				theArea = new NFmiLatLonArea(NFmiPoint(info->BottomLeftLongitude(), info->BottomLeftLatitude()),
				                             NFmiPoint(info->TopRightLongitude(), info->TopRightLatitude()));

				break;
			}

		case kRotatedLatLonProjection:
			{
				theArea = new NFmiRotatedLatLonArea(NFmiPoint(info->BottomLeftLongitude(), info->BottomLeftLatitude()),
				                                    NFmiPoint(info->TopRightLongitude(), info->TopRightLatitude()),
				                                    NFmiPoint(0., -30.) // south pole location
				                                   );
				break;
			}

		case kStereographicProjection:
			{
				theArea = new NFmiStereographicArea(NFmiPoint(info->BottomLeftLongitude(), info->BottomLeftLatitude()),
				                                    NFmiPoint(info->TopRightLongitude(), info->TopRightLatitude()),
				                                    info->Orientation());
				break;

			}

		default:
			itsLogger->Warning("No supported projection found");
			return NFmiHPlaceDescriptor();
			break;
	}

	NFmiGrid theGrid (theArea, info->Ni(), info->Nj());

	delete theArea;

	return NFmiHPlaceDescriptor(theGrid);

}

NFmiVPlaceDescriptor querydata::CreateVPlaceDescriptor(shared_ptr<info> info, bool theActiveOnly)
{

	NFmiLevelBag lbag;

	if (theActiveOnly)
	{
		lbag.AddLevel(NFmiLevel(info->Level()->Type(), "Hipihipi", info->Level()->Value()));
	}
	else
	{

		info->ResetLevel();

		while (info->NextLevel())
		{
			lbag.AddLevel(NFmiLevel(info->Level()->Type(), "Hipihipi", info->Level()->Value()));
		}
	}

	return NFmiVPlaceDescriptor(lbag);

}
