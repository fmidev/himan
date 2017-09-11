/**
 * @file querydata.cpp
 *
 */

#include "querydata.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include "point_list.h"
#include "stereographic_grid.h"
#include "ogr_spatialref.h"
#include <fstream>

#include "plugin_factory.h"
#include "radon.h"

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#endif

#include <NFmiGdalArea.h>
#include <NFmiLatLonArea.h>
#include <NFmiQueryData.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiTimeList.h>
#include "NFmiFastQueryInfo.h"

#ifdef __clang__

#pragma clang diagnostic pop

#endif

using namespace std;
using namespace himan::plugin;

querydata::querydata() : itsUseDatabase(true)
{
	itsLogger = logger("querydata");
}
bool querydata::ToFile(info& theInfo, string& theOutputFile)
{
	ofstream out(theOutputFile.c_str());

	bool activeOnly = true;

	if (itsWriteOptions.configuration->FileWriteOption() == kSingleFile)
	{
		activeOnly = false;
	}

	auto qdata = CreateQueryData(theInfo, activeOnly, true);

	if (!qdata)
	{
		return false;
	}

	out << *qdata;

	itsLogger.Info("Wrote file '" + theOutputFile + "'");

	return true;
}

shared_ptr<NFmiQueryData> querydata::CreateQueryData(const info& originalInfo, bool activeOnly, bool applyScaleAndBase)
{
	auto localInfo(originalInfo);

	/*
	 * Create required descriptors
	 */

	shared_ptr<NFmiQueryData> qdata;

	NFmiParamDescriptor pdesc = CreateParamDescriptor(localInfo, activeOnly);
	NFmiTimeDescriptor tdesc = CreateTimeDescriptor(localInfo, activeOnly);
	NFmiHPlaceDescriptor hdesc = CreateHPlaceDescriptor(localInfo, activeOnly);
	NFmiVPlaceDescriptor vdesc = CreateVPlaceDescriptor(localInfo, activeOnly);

	assert(pdesc.Size());
	assert(tdesc.Size());
	assert(hdesc.Size());
	assert(vdesc.Size());

	if (pdesc.Size() == 0)
	{
		itsLogger.Error("No valid parameters found");
		return qdata;
	}
	else if (tdesc.Size() == 0)
	{
		itsLogger.Error("No valid times found");
		return qdata;
	}
	else if (hdesc.Size() == 0)
	{
		itsLogger.Error("No valid grids found");
		return qdata;
	}
	else if (vdesc.Size() == 0)
	{
		itsLogger.Error("No valid levels found");
		return qdata;
	}

	NFmiFastQueryInfo qi(pdesc, tdesc, hdesc, vdesc);

	qdata = make_shared<NFmiQueryData>(new NFmiQueryData(qi));
	qdata->Init();
	qdata->Info()->SetProducer(
	    NFmiProducer(static_cast<unsigned long>(localInfo.Producer().Id()), localInfo.Producer().Name()));

	NFmiFastQueryInfo qinfo = qdata.get();

	/*
	 * At the same time check that we have only constant-sized grids
	 */

	if (activeOnly)
	{
		qinfo.FirstParam();
		qinfo.FirstLevel();
		qinfo.FirstTime();

		CopyData(localInfo, qinfo, applyScaleAndBase);
	}
	else
	{
		/*
		 * info-class (like grib) support many different grid types, the "worst" case being that
		 * all info-class grid elements are different from each other (different producer,grid,area,time etc).
		 * querydata on the other does not support this, so we should check that all elements are equal
		 * before writing querydata.
		 */

		localInfo.ResetTime();
		qinfo.ResetTime();

		while (localInfo.NextTime() && qinfo.NextTime())
		{
			localInfo.ResetLevel();
			qinfo.ResetLevel();

			while (localInfo.NextLevel() && qinfo.NextLevel())
			{
				localInfo.ResetParam();
				qinfo.ResetParam();

				while (localInfo.NextParam() && qinfo.NextParam())
				{
					if (!localInfo.Grid())
					{
						// No data in info (sparse info class)

						continue;
					}

					CopyData(localInfo, qinfo, applyScaleAndBase);
				}
			}
		}
	}

	qdata->LatLonCache();

	return qdata;
}

bool querydata::CopyData(info& theInfo, NFmiFastQueryInfo& qinfo, bool applyScaleAndBase) const
{
	assert(theInfo.Data().Size() == qinfo.SizeLocations());

	// convert missing value to kFloatMissing
	theInfo.Grid()->Data().MissingValue(kFloatMissing);
	theInfo.ResetLocation();
	qinfo.ResetLocation();

	double scale = 1, base = 0;

	if (applyScaleAndBase)
	{
		scale = theInfo.Param().Scale();
		base = theInfo.Param().Base();
	}

	if (theInfo.Grid()->Class() == kRegularGrid && theInfo.Grid()->ScanningMode() != kBottomLeft)
	{
		assert(theInfo.Grid()->ScanningMode() == kTopLeft);

		size_t nj = theInfo.Data().SizeY();
		size_t ni = theInfo.Data().SizeX();

		int y = static_cast<int>(nj) - 1;

		do
		{
			size_t x = 0;

			do
			{
				qinfo.NextLocation();
				qinfo.FloatValue(static_cast<float>(theInfo.Data().At(x, y) * scale + base));
				x++;
			} while (x < ni);

			y--;
		} while (y != -1);
	}
	else
	{
		// Grid is irregular OR source & dest are both kBottomLeft
		while (theInfo.NextLocation() && qinfo.NextLocation())
		{
			qinfo.FloatValue(static_cast<float>(theInfo.Value() * scale + base));
		}
	}

	// return to original missing value
	theInfo.Grid()->Data().MissingValue(MissingDouble());

	return true;
}

NFmiTimeDescriptor querydata::CreateTimeDescriptor(info& info, bool theActiveOnly)
{
	/*
	 * Create time descriptor
	 */

	NFmiTimeList tlist;

	NFmiMetTime originTime;

	if (theActiveOnly)
	{
		originTime = NFmiMetTime(boost::lexical_cast<long>(info.Time().ValidDateTime().String("%Y%m%d")),
		                         boost::lexical_cast<long>(info.Time().OriginDateTime().String("%H%M")));

		tlist.Add(new NFmiMetTime(boost::lexical_cast<long>(info.Time().ValidDateTime().String("%Y%m%d")),
		                          boost::lexical_cast<long>(info.Time().ValidDateTime().String("%H%M"))));
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
				originTime = NFmiMetTime(boost::lexical_cast<long>(firstOriginTime.String("%Y%m%d")),
				                         boost::lexical_cast<long>(firstOriginTime.String("%H%M")));
			}
			else
			{
				if (firstOriginTime != info.Time().OriginDateTime())
				{
					itsLogger.Error("Origintime is not the same for all grids in info");
					return NFmiTimeDescriptor();
				}
			}

			tlist.Add(new NFmiMetTime(boost::lexical_cast<long>(info.Time().ValidDateTime().String("%Y%m%d")),
			                          boost::lexical_cast<long>(info.Time().ValidDateTime().String("%H%M"))));
		}
	}

	return NFmiTimeDescriptor(originTime, tlist);
}

void AddToParamBag(himan::info& info, NFmiParamBag& pbag, bool readParamInfoFromDatabase)
{
	using namespace himan;

	if (info.Param().UnivId() == static_cast<long>(kHPMissingInt) && readParamInfoFromDatabase)
	{
		auto r = GET_PLUGIN(radon);

		auto levelInfo =
		    r->RadonDB().GetLevelFromDatabaseName(boost::to_upper_copy(HPLevelTypeToString.at(info.Level().Type())));

		param p = info.Param();

		if (!levelInfo.empty() && !levelInfo["id"].empty())
		{
			auto parmInfo = r->RadonDB().GetParameterFromDatabaseName(info.Producer().Id(), info.Param().Name(),
			                                                          stoi(levelInfo["id"]), info.Level().Value());

			if (!parmInfo.empty() && !parmInfo["univ_id"].empty())
			{
				p.UnivId(stol(parmInfo["univ_id"]));
				p.Scale(stod(parmInfo["scale"]));
				p.Base(stod(parmInfo["base"]));
				p.InterpolationMethod(kBiLinear);
			}
		}

		info.SetParam(p);
	}

	NFmiParam nbParam(info.Param().UnivId(), info.Param().Name(), ::kFloatMissing, ::kFloatMissing,
	                  static_cast<float>(info.Param().Scale()), static_cast<float>(info.Param().Base()), "%.1f",
	                  ::kLinearly);

	pbag.Add(NFmiDataIdent(nbParam));
}

NFmiParamDescriptor querydata::CreateParamDescriptor(info& info, bool theActiveOnly)
{
	/*
	 * Create parameter descriptor
	 */

	NFmiParamBag pbag;

	if (theActiveOnly)
	{
		AddToParamBag(info, pbag, itsUseDatabase);
	}
	else
	{
		info.ResetParam();

		while (info.NextParam())
		{
			AddToParamBag(info, pbag, itsUseDatabase);
		}
	}

	return NFmiParamDescriptor(pbag);
}

NFmiHPlaceDescriptor querydata::CreatePoint(info& info) const
{
	const point_list* g = dynamic_cast<point_list*>(info.Grid());
	NFmiLocationBag bag;

	for (const station& s : g->Stations())
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

	switch (info.Grid()->Type())
	{
		case kLatitudeLongitude:
		{
			latitude_longitude_grid* const g = dynamic_cast<latitude_longitude_grid*>(info.Grid());

			theArea = new NFmiLatLonArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
			                             NFmiPoint(g->TopRight().X(), g->TopRight().Y()));

			break;
		}

		case kRotatedLatitudeLongitude:
		{
			rotated_latitude_longitude_grid* const g = dynamic_cast<rotated_latitude_longitude_grid*>(info.Grid());

			theArea = new NFmiRotatedLatLonArea(
			    NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()), NFmiPoint(g->TopRight().X(), g->TopRight().Y()),
			    NFmiPoint(g->SouthPole().X(), g->SouthPole().Y()), NFmiPoint(0., 0.),  // default values
			    NFmiPoint(1., 1.),                                                     // default values
			    true);

			break;
		}

		case kStereographic:
		{
			stereographic_grid* const g = dynamic_cast<stereographic_grid*>(info.Grid());

			theArea = new NFmiStereographicArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
			                                    g->Di() * static_cast<double>((g->Ni() - 1)),
			                                    g->Dj() * static_cast<double>((g->Nj() - 1)), g->Orientation());

			break;
		}

		case kLambertConformalConic:
		{
			lambert_conformal_grid* const g = dynamic_cast<lambert_conformal_grid*>(info.Grid());

			std::stringstream ss;
			ss << "GEOGCS[\"MEPS\","
			   << " DATUM[\"unknown\","
			   << "     SPHEROID[\"Sphere\",6367470,0]],"
			   << " PRIMEM[\"Greenwich\",0],"
			   << " UNIT[\"degree\",0.0174532925199433]]";

			theArea =
			    new NFmiGdalArea(ss.str(), g->SpatialReference(), 0, 0, g->Di() * (static_cast<double>(g->Ni()) - 1),
			                     g->Dj() * (static_cast<double>(g->Nj()) - 1));
			break;
		}

		default:
			itsLogger.Error("No supported projection found");
			return NFmiHPlaceDescriptor();
			break;
	}

	assert(theArea);

	NFmiGrid theGrid(theArea, info.Grid()->Ni(), info.Grid()->Nj());

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
						firstGrid = g;
						continue;
					}

					if (firstGrid->Type() != g->Type())
					{
						itsLogger.Error("All grids in info are not equal, unable to write querydata");
						return NFmiHPlaceDescriptor();
					}

					if (firstGrid->Class() == kRegularGrid)
					{
						if (*firstGrid != *g)
						{
							itsLogger.Error("All grids in info are not equal, unable to write querydata");
							return NFmiHPlaceDescriptor();
						}
					}
					else
					{
						const point_list* fg_ = dynamic_cast<const point_list*>(firstGrid);
						point_list* g_ = dynamic_cast<point_list*>(info.Grid());

						if (*fg_ != *g_)
						{
							itsLogger.Error("All grids in info are not equal, unable to write querydata");
							return NFmiHPlaceDescriptor();
						}
					}
				}
			}
		}
	}

	if (info.Grid()->Class() == kRegularGrid)
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
		lbag.AddLevel(NFmiLevel(info.Level().Type(), HPLevelTypeToString.at(info.Level().Type()),
		                        static_cast<float>(info.Level().Value())));
	}
	else
	{
		info.ResetLevel();

		while (info.NextLevel())
		{
			lbag.AddLevel(NFmiLevel(info.Level().Type(), HPLevelTypeToString.at(info.Level().Type()),
			                        static_cast<float>(info.Level().Value())));
		}
	}

	return NFmiVPlaceDescriptor(lbag);
}

shared_ptr<himan::info> querydata::FromFile(const string& inputFile, const search_options& options) const
{
	throw runtime_error(ClassName() + ": Function FromFile() not implemented yet");
}

shared_ptr<himan::info> querydata::CreateInfo(shared_ptr<NFmiQueryData> theData) const
{
	auto newInfo = make_shared<info>();
	grid* newGrid = 0;

	NFmiQueryInfo qinfo = theData.get();

	producer p(230, 86, 230, "HIMAN");
	p.TableVersion(203);

	newInfo->Producer(p);

	// Times

	vector<forecast_time> theTimes;

	raw_time originTime(string(qinfo.OriginTime().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");

	for (qinfo.ResetTime(); qinfo.NextTime();)
	{
		raw_time ct(string(qinfo.Time().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");

		forecast_time t(originTime, ct);
		theTimes.push_back(t);
	}

	newInfo->Times(theTimes);

	// Levels

	vector<level> theLevels;

	for (qinfo.ResetLevel(); qinfo.NextLevel();)
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
				throw runtime_error("Unknown level type in querydata: " + to_string(qinfo.Level()->LevelType()));
				break;
		}

		level l(lt, qinfo.Level()->LevelValue());

		theLevels.push_back(l);
	}

	newInfo->Levels(theLevels);

	// Parameters

	vector<himan::param> theParams;

	for (qinfo.ResetParam(); qinfo.NextParam();)
	{
		param p(string(qinfo.Param().GetParamName()), qinfo.Param().GetParamIdent());
		theParams.push_back(p);
	}

	newInfo->Params(theParams);

	vector<forecast_type> ftypes;
	ftypes.push_back(forecast_type(kDeterministic));

	newInfo->ForecastTypes(ftypes);

	// Grid

	const size_t ni = qinfo.Grid()->XNumber();
	const size_t nj = qinfo.Grid()->YNumber();

	switch (qinfo.Area()->ClassId())
	{
		case kNFmiLatLonArea:
		{
			newGrid = new latitude_longitude_grid;
			latitude_longitude_grid* const ll = dynamic_cast<latitude_longitude_grid*>(newGrid);
			ll->BottomLeft(point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()));
			ll->TopRight(point(qinfo.Area()->TopRightLatLon().X(), qinfo.Area()->TopRightLatLon().Y()));
			ll->Ni(ni);
			ll->Nj(nj);
		}
		break;

		case kNFmiRotatedLatLonArea:
		{
			newGrid = new rotated_latitude_longitude_grid;
			rotated_latitude_longitude_grid* const rll = dynamic_cast<rotated_latitude_longitude_grid*>(newGrid);
			NFmiPoint southPole = reinterpret_cast<const NFmiRotatedLatLonArea*>(qinfo.Area())->SouthernPole();
			rll->SouthPole(point(southPole.X(), southPole.Y()));
			rll->UVRelativeToGrid(false);
			rll->BottomLeft(point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()));
			rll->TopRight(point(qinfo.Area()->TopRightLatLon().X(), qinfo.Area()->TopRightLatLon().Y()));
			rll->Ni(ni);
			rll->Nj(nj);
		}
		break;

		case kNFmiStereographicArea:
		{
			newGrid = new stereographic_grid;
			stereographic_grid* const s = dynamic_cast<stereographic_grid*>(newGrid);
			s->Orientation(reinterpret_cast<const NFmiStereographicArea*>(qinfo.Area())->Orientation());
			s->BottomLeft(point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()));
			s->TopRight(point(qinfo.Area()->TopRightLatLon().X(), qinfo.Area()->TopRightLatLon().Y()));
			s->Ni(ni);
			s->Nj(nj);
		}
		break;

		default:
			itsLogger.Fatal("Invalid projection");
			abort();
	}

	newGrid->ScanningMode(kBottomLeft);

	newInfo->Create(newGrid);

	delete newGrid;

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

				matrix<double> dm(ni, nj, 1, static_cast<double>(32700.f));
				size_t i;

				for (qinfo.ResetLocation(), i = 0; qinfo.NextLocation() && i < ni * nj; i++)
				{
					dm.Set(i, static_cast<double>(qinfo.FloatValue()));
				}

				// convert kFloatMissing to nan
				dm.MissingValue(MissingDouble());
				newInfo->Grid()->Data(dm);
			}
		}
	}

	return newInfo;
}

bool querydata::UseDatabase() const { return itsUseDatabase; }
void querydata::UseDatabase(bool theUseDatabase) { itsUseDatabase = theUseDatabase; }
