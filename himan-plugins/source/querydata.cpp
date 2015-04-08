/**
 * @file querydata.cpp
 *
 * @date Nov 27, 2012
 * @author: partio
 */


#include "querydata.h"
#include "logger_factory.h"
#include <fstream>
#include "regular_grid.h"
#include "irregular_grid.h"
#include <boost/foreach.hpp>

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

#ifdef __clang__

#pragma clang diagnostic pop

#endif

using namespace std;
using namespace himan::plugin;

querydata::querydata()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("querydata"));
}

bool querydata::ToFile(info& theInfo, string& theOutputFile, HPFileWriteOption fileWriteOption)
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

shared_ptr<NFmiQueryData> querydata::CreateQueryData(info& theInfo, bool activeOnly)
{

	/*
	 * Create required descriptors
	 */

	shared_ptr<NFmiQueryData> qdata;
	
	NFmiParamDescriptor pdesc = CreateParamDescriptor(theInfo, activeOnly);
	NFmiTimeDescriptor tdesc = CreateTimeDescriptor(theInfo, activeOnly);
	NFmiHPlaceDescriptor hdesc = CreateHPlaceDescriptor(theInfo, activeOnly);
	NFmiVPlaceDescriptor vdesc = CreateVPlaceDescriptor(theInfo, activeOnly);

	assert(pdesc.Size());
	assert(tdesc.Size());
	assert(hdesc.Size());
	assert(vdesc.Size());

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
		itsLogger->Error("No valid grids found");
		return qdata;
	}
	else if (vdesc.Size() == 0)
	{
		itsLogger->Error("No valid levels found");
		return qdata;
	}

	NFmiFastQueryInfo qi(pdesc, tdesc, hdesc, vdesc);

	qdata = make_shared<NFmiQueryData> (new NFmiQueryData(qi));
    qdata->Init();
	qdata->Info()->SetProducer(NFmiProducer(static_cast<unsigned long> (theInfo.Producer().Id()), theInfo.Producer().Name()));

	NFmiFastQueryInfo qinfo = qdata.get();

	/*
	 * At the same time check that we have only constant-sized grids
	 */

	if (activeOnly)
	{

		qinfo.FirstParam();
		qinfo.FirstLevel();
		qinfo.FirstTime();

		CopyData(theInfo, qinfo);
	}
	else
	{
		/*
		 * info-class (like grib) support many different grid types, the "worst" case being that
		 * all info-class grid elements are different from each other (different producer,grid,area,time etc).
		 * querydata on the other does not support this, so we should check that all elements are equal
		 * before writing querydata.
		 */

		theInfo.ResetTime();
		qinfo.ResetTime();

		while (theInfo.NextTime() && qinfo.NextTime())
		{

			theInfo.ResetLevel();
			qinfo.ResetLevel();

			while (theInfo.NextLevel() && qinfo.NextLevel())
			{
				theInfo.ResetParam();
				qinfo.ResetParam();

				while (theInfo.NextParam() && qinfo.NextParam())
				{
					
					if (!theInfo.Grid())
					{
						// No data in info (sparse info class)
						
						continue;
					}

					CopyData(theInfo, qinfo);

				}
			}
		}
	}

	qdata->LatLonCache();
	return qdata;

}


bool querydata::CopyData(info& theInfo, NFmiFastQueryInfo& qinfo) const
{
	bool swapped = false;
	
	HPScanningMode originalMode = kUnknownScanningMode;
	
	if (theInfo.Grid()->Type() == kRegularGrid)
	{
		originalMode = dynamic_cast<regular_grid*> (theInfo.Grid())->ScanningMode();

		if (originalMode == kTopLeft)
		{
			// For newbase we have swap data to kBottomLeft.

			swapped = true;

			dynamic_cast<regular_grid*> (theInfo.Grid())->Swap(kBottomLeft);

		}
		else if (originalMode != kBottomLeft)
		{
			itsLogger->Fatal("Invalid scannignmode: " + string(HPScanningModeToString.at(originalMode)));
			exit(1);
		}
	}
	
	assert(theInfo.Data().Size() == qinfo.Size());

	theInfo.ResetLocation();
	qinfo.ResetLocation();

	while (theInfo.NextLocation() && qinfo.NextLocation())
	{
		qinfo.FloatValue(static_cast<float> (theInfo.Value()));
	}

	if (swapped)
	{
		dynamic_cast<regular_grid*> (theInfo.Grid())->Swap(originalMode);
	}

	return true;
}

NFmiTimeDescriptor querydata::CreateTimeDescriptor(info& info, bool theActiveOnly)
{

	/*
	 * Create time descriptor
	 */

	NFmiTimeList tlist;

	if (theActiveOnly)
	{
		tlist.Add(new NFmiMetTime(boost::lexical_cast<long> (info.Time().ValidDateTime().String("%Y%m%d")),
								  boost::lexical_cast<long> (info.Time().ValidDateTime().String("%H%M"))));
	}
	else
	{
		info.ResetTime();

		raw_time firstOriginTime;
		
		while (info.NextTime())
		{
			if (firstOriginTime.Empty())
			{
				firstOriginTime = info.Time().OriginDateTime();
			}
			else
			{
				if (firstOriginTime != info.Time().OriginDateTime())
				{
					itsLogger->Error("Origintime is not the same for all grids in info");
					return NFmiTimeDescriptor();
				}
			}
			
			tlist.Add(new NFmiMetTime(boost::lexical_cast<long> (info.Time().ValidDateTime().String("%Y%m%d")),
									  boost::lexical_cast<long> (info.Time().ValidDateTime().String("%H%M"))));
		}

	}

	return NFmiTimeDescriptor(tlist.FirstTime(), tlist);

}


NFmiParamDescriptor querydata::CreateParamDescriptor(info& info, bool theActiveOnly)
{

	/*
	 * Create parameter descriptor
	 */

	NFmiParamBag pbag;

	if (theActiveOnly)
	{
		assert(info.Param().UnivId());

		pbag.Add(NFmiDataIdent(NFmiParam(info.Param().UnivId(), info.Param().Name())));

	}
	else
	{
		info.ResetParam();

		while (info.NextParam())
		{

			assert(info.Param().UnivId());

			pbag.Add(NFmiDataIdent(NFmiParam(info.Param().UnivId(), info.Param().Name())));

		}
	}

	return NFmiParamDescriptor(pbag);

}


NFmiHPlaceDescriptor querydata::CreatePoint(info& info) const
{
	const irregular_grid* g = dynamic_cast<irregular_grid*> (info.Grid());
	NFmiLocationBag bag;
	
	BOOST_FOREACH(const station& s, g->Stations())
	{
		NFmiStation stat(s.Id(), s.Name(), s.X(), s.Y());
		bag.AddLocation(stat);
	}
	
	return NFmiHPlaceDescriptor(bag);
}

NFmiHPlaceDescriptor querydata::CreateGrid(info& info) const
{
	/*
	 * If whole info is converted to querydata, we need to check that if info contains
	 * more than one element, the grids of each element must be equal!
	 *
	 * TODO: interpolate to same grid if they are different ???
	 */


	NFmiArea* theArea = 0;

	const regular_grid* g = dynamic_cast<const regular_grid*> (info.Grid());
	
	switch (g->Projection())
	{
		case kLatLonProjection:
		{
			theArea = new NFmiLatLonArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
										 NFmiPoint(g->TopRight().X(), g->TopRight().Y()));

			break;
		}

		case kRotatedLatLonProjection:
		{
			theArea = new NFmiRotatedLatLonArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
												NFmiPoint(g->TopRight().X(), g->TopRight().Y()),
												NFmiPoint(g->SouthPole().X(), g->SouthPole().Y()),
												NFmiPoint(0.,0.), // default values
												NFmiPoint(1.,1.), // default values
												true);

			break;
		}

		case kStereographicProjection:
		{
			theArea = new NFmiStereographicArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
												g->Di() * static_cast<double> ((g->Ni()-1)),
												g->Dj() * static_cast<double> ((g->Nj()-1)),
												g->Orientation());

			break;

		}

		default:
			itsLogger->Error("No supported projection found");
			return NFmiHPlaceDescriptor();
			break;
	}

	NFmiGrid theGrid (theArea, g->Ni(), g->Nj());

	delete theArea;

	return NFmiHPlaceDescriptor(theGrid);
	
}

NFmiHPlaceDescriptor querydata::CreateHPlaceDescriptor(info& info, bool activeOnly)
{
	if (!activeOnly && info.SizeTimes() * info.SizeParams() * info.SizeLevels() > 1)
	{
		info.ResetTime();
		const grid* firstGrid = 0;

		while (info.NextTime())
		{
			info.ResetLevel();

			while (info.NextLevel())
			{
				info.ResetParam();

				while (info.NextParam())
				{
					grid* g = info.Grid();

					if (!g)
					{
						continue;
					}

					if (!firstGrid)
					{
						if (firstGrid->Type() == kRegularGrid)
						{
							firstGrid = dynamic_cast<regular_grid*> (g);
						}
						else
						{
							firstGrid = dynamic_cast<irregular_grid*> (g);	
						}
						
						continue;
					}

					if (firstGrid->Type() != g->Type())
					{
						itsLogger->Error("All grids in info are not equal, unable to write querydata");
						return NFmiHPlaceDescriptor();
					}
					
					if (firstGrid->Type() == kRegularGrid)
					{
						const regular_grid* fg_ = dynamic_cast<const regular_grid*> (firstGrid);
						regular_grid* g_ = dynamic_cast<regular_grid*> (info.Grid());
					
						if (*fg_ != *g_)
						{
							itsLogger->Error("All grids in info are not equal, unable to write querydata");
							return NFmiHPlaceDescriptor();
						}
					}
					else
					{
						const irregular_grid* fg_ = dynamic_cast<const irregular_grid*> (firstGrid);
						irregular_grid* g_ = dynamic_cast<irregular_grid*> (info.Grid());
					
						if (*fg_ != *g_)
						{
							itsLogger->Error("All grids in info are not equal, unable to write querydata");
							return NFmiHPlaceDescriptor();
						}
					}
				}
			}
		}
	}


	if (info.Grid()->Type() == kRegularGrid)
	{
		return CreateGrid(info);
	}
	else
	{
		return CreatePoint(info);
	}			

}

NFmiVPlaceDescriptor querydata::CreateVPlaceDescriptor(info& info, bool theActiveOnly)
{

	NFmiLevelBag lbag;

	if (theActiveOnly)
	{
		lbag.AddLevel(NFmiLevel(info.Level().Type(), HPLevelTypeToString.at(info.Level().Type()), static_cast<float> (info.Level().Value())));
	}
	else
	{

		info.ResetLevel();

		while (info.NextLevel())
		{
			lbag.AddLevel(NFmiLevel(info.Level().Type(), HPLevelTypeToString.at(info.Level().Type()), static_cast<float> (info.Level().Value())));
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
	regular_grid newGrid;

	NFmiQueryInfo qinfo = theData.get();

	producer p (230, 86, 230, "HIMAN");
	p.TableVersion(203);
	
	newInfo->Producer(p);
	
	// Times

	vector<forecast_time> theTimes;

	raw_time originTime(string(qinfo.OriginTime().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");
	
	for (qinfo.ResetTime(); qinfo.NextTime(); )
	{
		raw_time ct(string(qinfo.Time().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");

		forecast_time t(originTime, ct);
		theTimes.push_back(t);
	}

	newInfo->Times(theTimes);

	// Levels

	vector<level> theLevels;

	for (qinfo.ResetLevel(); qinfo.NextLevel(); )
	{
		HPLevelType lt = kUnknownLevel;

		switch (qinfo.Level()->LevelType())
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
				throw runtime_error("Unknown level type in querydata: " + boost::lexical_cast<string> (qinfo.Level()->LevelType()));
				break;
		}

		level l(lt, qinfo.Level()->LevelValue());

		theLevels.push_back(l);
	}

	newInfo->Levels(theLevels);

	// Parameters

	vector<himan::param> theParams;

	for (qinfo.ResetParam(); qinfo.NextParam(); )
	{
		param p (string(qinfo.Param().GetParamName()), qinfo.Param().GetParamIdent());
		theParams.push_back(p);
	}

	newInfo->Params(theParams);

	vector<forecast_type> ftypes;
	ftypes.push_back(forecast_type(kDeterministic));
	
	newInfo->ForecastTypes(ftypes);
	
	// Grid

	newGrid.ScanningMode(kBottomLeft);
	newGrid.UVRelativeToGrid(false);

	switch (qinfo.Area()->ClassId())
	{
		case kNFmiLatLonArea:
			newGrid.Projection(kLatLonProjection);
			break;

		case kNFmiRotatedLatLonArea:
		{
			newGrid.Projection(kRotatedLatLonProjection);
			NFmiPoint southPole = reinterpret_cast<const NFmiRotatedLatLonArea*> (qinfo.Area())->SouthernPole();
			newGrid.SouthPole(point(southPole.X(), southPole.Y()));
		}
			break;

		case kNFmiStereographicArea:
			newGrid.Projection(kStereographicProjection);
			newGrid.Orientation(reinterpret_cast<const NFmiStereographicArea*> (qinfo.Area())->Orientation());
			//newGrid.Di(reinterpret_cast<const NFmiStereographicArea*> (qinfo.Area())->);
			//newGrid.Dj(itsGrib->Message()->YLengthInMeters());

			break;
	}

	size_t ni = qinfo.Grid()->XNumber();
	size_t nj = qinfo.Grid()->YNumber();
	
	newGrid.BottomLeft(point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()));
	newGrid.TopRight(point(qinfo.Area()->TopRightLatLon().X(), qinfo.Area()->TopRightLatLon().Y()));

	newInfo->Create(&newGrid);

	// Copy data

	newInfo->FirstForecastType();
	
	for (newInfo->ResetTime(), qinfo.ResetTime(); newInfo->NextTime() && qinfo.NextTime();)
	{
		assert(newInfo->TimeIndex() == qinfo.TimeIndex());

		for (newInfo->ResetLevel(), qinfo.ResetLevel(); newInfo->NextLevel() && qinfo.NextLevel();)
		{
			assert(newInfo->LevelIndex() == qinfo.LevelIndex());

			for (newInfo->ResetParam(), qinfo.ResetParam(); newInfo->NextParam() && qinfo.NextParam();)
			{
				assert(newInfo->ParamIndex() == qinfo.ParamIndex());

				unpacked dm(ni, nj, 1, kFloatMissing);

				size_t i;
				
				for (qinfo.ResetLocation(), i = 0; qinfo.NextLocation() && i < ni*nj; i++)
				{
					dm.Set(i, static_cast<double> (qinfo.FloatValue()));
				}

				newInfo->Grid()->Data(dm);
			}
		}
	}

	return newInfo;

}
