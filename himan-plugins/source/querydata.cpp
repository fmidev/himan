/**
 * @file querydata.cpp
 *
 * @date Nov 27, 2012
 * @author: partio
 */


#include "querydata.h"
#include "logger_factory.h"
#include <fstream>

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#endif

#include <NFmiQueryData.h>
#include <NFmiTimeList.h>
#include <NFmiLatLonArea.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiQueryDataUtil.h>

#ifdef __clang__

#pragma clang diagnostic pop

#endif

using namespace std;
using namespace himan::plugin;

querydata::querydata()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("querydata"));
}

bool querydata::ToFile(shared_ptr<info> theInfo, const string& theOutputFile, HPFileWriteOption fileWriteOption)
{
	ofstream out(theOutputFile.c_str());

	bool activeOnly = true;

	if (fileWriteOption == kSingleFile)
	{
		activeOnly = false;
	}

	auto qdata = CreateQueryData(theInfo, activeOnly);

	if (!qdata)
	{
		return false;
	}

	out << *qdata;

	itsLogger->Info("Wrote file '" + theOutputFile + "'");

	return true;
	
}

shared_ptr<NFmiQueryData> querydata::CreateQueryData(shared_ptr<info> theInfo, bool activeOnly)
{
	
	/*
	 * Create required descriptors
	 */

	NFmiParamDescriptor pdesc = CreateParamDescriptor(theInfo, activeOnly);
	NFmiTimeDescriptor tdesc = CreateTimeDescriptor(theInfo, activeOnly);
	NFmiHPlaceDescriptor hdesc = CreateHPlaceDescriptor(theInfo, activeOnly);
	NFmiVPlaceDescriptor vdesc = CreateVPlaceDescriptor(theInfo, activeOnly);

	assert(pdesc.Size());
	assert(tdesc.Size());
	assert(hdesc.Size());
	assert(vdesc.Size());

	shared_ptr<NFmiQueryData> qdata;
	
	if (pdesc.Size() == 0)
	{
		itsLogger->Error("No valid parameters found");
		return qdata;
	}
	else if (tdesc.Size() == 0)
	{
		itsLogger->Error("No valid times found");
		return qdata;
	}
	else if (hdesc.Size() == 0)
	{
		itsLogger->Error("No valid times found");
		return qdata;
	}
	else if (vdesc.Size() == 0)
	{
		itsLogger->Error("No valid times found");
		return qdata;
	}

	NFmiFastQueryInfo qi(pdesc, tdesc, hdesc, vdesc);

	qdata = make_shared<NFmiQueryData> (*NFmiQueryDataUtil::CreateEmptyData(qi));

	NFmiFastQueryInfo qinfo = qdata.get();

	qinfo.SetProducer(NFmiProducer(static_cast<unsigned long> (theInfo->Producer().Id()), theInfo->Producer().Name()));

	/*
	 * At the same time check that we have only constant-sized grids
	 */

#ifndef NDEBUG
	size_t size = 0;
	string firstOriginTime;

	bool first = true;
#endif

	if (activeOnly)
	{
		assert(theInfo->Grid()->ScanningMode() == kBottomLeft);

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
					
					if (theInfo->Dimensions()->IsMissing(theInfo->TimeIndex(), theInfo->LevelIndex(), theInfo->ParamIndex()))
					{
						// No data in info
						
						continue;
					}

					assert(theInfo->Grid()->ScanningMode() == kBottomLeft);

					qinfo.ResetLocation();
					theInfo->ResetLocation();
					
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


	return qdata;

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

NFmiHPlaceDescriptor querydata::CreateHPlaceDescriptor(shared_ptr<info> info, bool activeOnly)
{

	/*
	 * If whole info is converted to querydata, we need to check that if info contains
	 * more than one element, the grids of each element must be equal!
	 */
	
	if (!activeOnly && info->SizeTimes() * info->SizeParams() * info->SizeLevels() > 1)
	{
		info->ResetTime();
		shared_ptr<grid> firstGrid;

		while (info->NextTime())
		{

			info->ResetLevel();

			while (info->NextLevel())
			{

				info->ResetParam();

				while (info->NextParam())
				{

					if (!firstGrid)
					{
						firstGrid = info->Grid();
						continue;
					}

					if (info->Dimensions()->IsMissing(info->TimeIndex(), info->LevelIndex(), info->ParamIndex()))
					{
						continue;
					}

					if (*firstGrid != *info->Grid())
					{
						itsLogger->Error("All grids in info are not equal, unable to write querydata");
						exit(1); // not sure what to do here!
					}

					assert(info->Grid()->ScanningMode() == kBottomLeft);
				}
			}
		}
	}

	NFmiArea* theArea = 0;

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
		lbag.AddLevel(NFmiLevel(info->Level().Type(), "Hipihipi", static_cast<float> (info->Level().Value())));
	}
	else
	{

		info->ResetLevel();

		while (info->NextLevel())
		{
			lbag.AddLevel(NFmiLevel(info->Level().Type(), "Hipihipi", static_cast<float> (info->Level().Value())));
		}
	}

	return NFmiVPlaceDescriptor(lbag);

}

shared_ptr<himan::info> querydata::FromFile(const string& inputFile, const search_options& options, bool readContents)
{
	throw runtime_error(ClassName() + ": Function FromFile() not implemented yet");
}

shared_ptr<himan::info> querydata::CreateInfo(shared_ptr<NFmiQueryData> theData) const
{
	auto newInfo = make_shared<info> ();
	auto newGrid = make_shared<grid> ();

	NFmiQueryInfo* qi = theData->Info();

	newInfo->Producer(producer(230, 86, 230, "HIMAN"));
	
	// Times

	vector<forecast_time> theTimes;

	raw_time originTime(string(qi->OriginTime().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");
	
	for (qi->ResetTime(); qi->NextTime(); )
	{
		raw_time ct(string(qi->Time().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");

		forecast_time t(originTime, ct);
		theTimes.push_back(t);
	}

	newInfo->Times(theTimes);

	// Levels

	vector<level> theLevels;

	for (qi->ResetLevel(); qi->NextLevel(); )
	{
		HPLevelType lt = kUnknownLevel;

		switch (qi->Level()->LevelType())
		{
			case kFmiHybridLevel:
				lt = kHybrid;
				break;

			case kFmiHeight:
				lt = kHeight;
				break;

			case kFmiPressure:
				lt = kPressure;
				break;

			case kFmiNoLevelType:
			case kFmiAnyLevelType:
				break;
				
			default:
				throw runtime_error("Unknown level type in querydata: " + boost::lexical_cast<string> (qi->Level()->LevelType()));
				break;
		}

		level l(lt, qi->Level()->LevelValue());

		theLevels.push_back(l);
	}

	newInfo->Levels(theLevels);

	// Parameters

	vector<himan::param> theParams;

	for (qi->ResetParam(); qi->NextParam(); )
	{
		param p (string(qi->Param().GetParamName()), qi->Param().GetParamIdent());
		theParams.push_back(p);
	}

	newInfo->Params(theParams);

	// Grid

	newGrid->ScanningMode(kBottomLeft);
	newGrid->UVRelativeToGrid(false);

	switch (qi->Area()->ClassId())
	{
		case kNFmiLatLonArea:
			newGrid->Projection(kLatLonProjection);
			break;

		case kNFmiRotatedLatLonArea:
			newGrid->Projection(kRotatedLatLonProjection);
			newGrid->SouthPole(reinterpret_cast<const NFmiRotatedLatLonArea*> (qi->Area())->SouthernPole());
			break;

		case kNFmiStereographicArea:
			newGrid->Projection(kStereographicProjection);
			newGrid->Orientation(reinterpret_cast<const NFmiStereographicArea*> (qi->Area())->Orientation());
			//newGrid->Di(reinterpret_cast<const NFmiStereographicArea*> (qi->Area())->);
			//newGrid->Dj(itsGrib->Message()->YLengthInMeters());

			break;
	}

	size_t ni = qi->Grid()->XNumber();
	size_t nj = qi->Grid()->YNumber();
	
	newGrid->BottomLeft(qi->Area()->BottomLeftLatLon());
	newGrid->TopRight(qi->Area()->TopRightLatLon());

	newInfo->Create(newGrid);

	// Copy data

	for (newInfo->ResetTime(), qi->ResetTime(); newInfo->NextTime() && qi->NextTime();)
	{
		for (newInfo->ResetLevel(), qi->ResetLevel(); newInfo->NextLevel() && qi->NextLevel();)
		{
			for (newInfo->ResetParam(), qi->ResetParam(); newInfo->NextParam() && qi->NextParam();)
			{
				shared_ptr<unpacked> dm = shared_ptr<unpacked> (new unpacked(ni, nj));

				size_t i;
				
				for (qi->ResetLocation(), i = 0; qi->NextLocation() && i < ni*nj; i++)
				{
					dm->Set(i, static_cast<double> (qi->FloatValue()));
				}

				newInfo->Grid()->Data(dm);
			}
		}
	}
	return newInfo;

}
