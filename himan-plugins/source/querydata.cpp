/**
 * @file querydata.cpp
 *
 */

#include "querydata.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include "ogr_spatialref.h"
#include "point_list.h"
#include "stereographic_grid.h"
#include <fstream>

#include "plugin_factory.h"
#include "radon.h"

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#endif

#include "NFmiFastQueryInfo.h"
#include <NFmiGdalArea.h>
#include <NFmiLatLonArea.h>
#include <NFmiQueryData.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiTimeList.h>

#ifdef __clang__

#pragma clang diagnostic pop

#endif

using namespace std;
using namespace himan::plugin;

querydata::querydata()
{
	itsLogger = logger("querydata");
}

bool querydata::ToFile(info<double>& theInfo, string& theOutputFile)
{
	return ToFile<double>(theInfo, theOutputFile);
}

template <typename T>
bool querydata::ToFile(info<T>& theInfo, string& theOutputFile)
{
	ofstream out(theOutputFile.c_str());

	bool activeOnly = true;

	if (itsWriteOptions.configuration->FileWriteOption() == kSingleFile)
	{
		activeOnly = false;
	}

	auto qdata = CreateQueryData<double>(theInfo, activeOnly, true);

	if (!qdata)
	{
		return false;
	}

	out << *qdata;

	itsLogger.Info("Wrote file '" + theOutputFile + "'");

	return true;
}

template bool querydata::ToFile<double>(info<double>&, string&);

template <typename T>
shared_ptr<NFmiQueryData> querydata::CreateQueryData(const info<T>& originalInfo, bool activeOnly,
                                                     bool applyScaleAndBase)
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

	ASSERT(pdesc.Size());
	ASSERT(tdesc.Size());
	ASSERT(hdesc.Size());
	ASSERT(vdesc.Size());

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

		localInfo.template Reset<forecast_time>();
		qinfo.ResetTime();

		while (localInfo.template Next<forecast_time>() && qinfo.NextTime())
		{
			localInfo.template Reset<level>();
			qinfo.ResetLevel();

			while (localInfo.template Next<level>() && qinfo.NextLevel())
			{
				localInfo.template Reset<param>();
				qinfo.ResetParam();

				while (localInfo.template Next<param>() && qinfo.NextParam())
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

template shared_ptr<NFmiQueryData> querydata::CreateQueryData<double>(const info<double>&, bool, bool);

bool querydata::CopyData(info<double>& theInfo, NFmiFastQueryInfo& qinfo, bool applyScaleAndBase) const
{
	ASSERT(theInfo.Data().Size() == qinfo.SizeLocations());

	// convert missing value to kFloatMissing
	theInfo.Data().MissingValue(kFloatMissing);
	theInfo.ResetLocation();
	qinfo.ResetLocation();

	double scale = 1, base = 0;

	if (applyScaleAndBase)
	{
		scale = theInfo.Param().Scale();
		base = theInfo.Param().Base();
	}

	if (theInfo.Grid()->Class() == kRegularGrid &&
	    dynamic_pointer_cast<regular_grid>(theInfo.Grid())->ScanningMode() != kBottomLeft)
	{
		ASSERT(std::dynamic_pointer_cast<regular_grid>(theInfo.Grid())->ScanningMode() == kTopLeft);

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
	theInfo.Data().MissingValue(MissingDouble());

	return true;
}

NFmiTimeDescriptor querydata::CreateTimeDescriptor(info<double>& info, bool theActiveOnly)
{
	/*
	 * Create time descriptor
	 */

	NFmiTimeList tlist;

	NFmiMetTime originTime;

	if (theActiveOnly)
	{
		originTime = NFmiMetTime(stol(info.Time().ValidDateTime().String("%Y%m%d")),
		                         stol(info.Time().OriginDateTime().String("%H%M")));

		tlist.Add(new NFmiMetTime(stol(info.Time().ValidDateTime().String("%Y%m%d")),
		                          stol(info.Time().ValidDateTime().String("%H%M"))));
	}
	else
	{
		info.Reset<forecast_time>();

		raw_time firstOriginTime;

		while (info.Next<forecast_time>())
		{
			if (firstOriginTime.Empty())
			{
				firstOriginTime = info.Time().OriginDateTime();
				originTime = NFmiMetTime(stol(firstOriginTime.String("%Y%m%d")), stol(firstOriginTime.String("%H%M")));
			}
			else
			{
				if (firstOriginTime != info.Time().OriginDateTime())
				{
					itsLogger.Error("Origintime is not the same for all grids in info");
					return NFmiTimeDescriptor();
				}
			}

			tlist.Add(new NFmiMetTime(stol(info.Time().ValidDateTime().String("%Y%m%d")),
			                          stol(info.Time().ValidDateTime().String("%H%M"))));
		}
	}

	return NFmiTimeDescriptor(originTime, tlist);
}

void AddToParamBag(himan::info<double>& info, NFmiParamBag& pbag)
{
	string precision;

	if (info.Param().Precision() != himan::kHPMissingInt)
	{
		precision = "%." + to_string(info.Param().Precision()) + "f";
	}
	else
	{
		precision = "%.1f";
	}

	NFmiParam nbParam(info.Param().UnivId(), info.Param().Name(), ::kFloatMissing, ::kFloatMissing,
	                  static_cast<float>(info.Param().Scale()), static_cast<float>(info.Param().Base()), precision,
	                  ::kLinearly);

	pbag.Add(NFmiDataIdent(nbParam));
}

NFmiParamDescriptor querydata::CreateParamDescriptor(info<double>& info, bool theActiveOnly)
{
	/*
	 * Create parameter descriptor
	 */

	NFmiParamBag pbag;

	if (theActiveOnly)
	{
		AddToParamBag(info, pbag);
	}
	else
	{
		info.Reset<param>();

		while (info.Next<param>())
		{
			AddToParamBag(info, pbag);
		}
	}

	return NFmiParamDescriptor(pbag);
}

NFmiHPlaceDescriptor querydata::CreatePoint(info<double>& info) const
{
	const auto g = std::dynamic_pointer_cast<point_list>(info.Grid());
	NFmiLocationBag bag;

	for (const station& s : g->Stations())
	{
		NFmiStation stat(s.Id(), s.Name(), s.X(), s.Y());
		bag.AddLocation(stat);
	}

	return NFmiHPlaceDescriptor(bag);
}

NFmiHPlaceDescriptor querydata::CreateGrid(info<double>& info) const
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
			auto g = std::dynamic_pointer_cast<latitude_longitude_grid>(info.Grid());

			theArea = new NFmiLatLonArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
			                             NFmiPoint(g->TopRight().X(), g->TopRight().Y()));

			break;
		}

		case kRotatedLatitudeLongitude:
		{
			auto g = std::dynamic_pointer_cast<rotated_latitude_longitude_grid>(info.Grid());

			theArea = new NFmiRotatedLatLonArea(
			    NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()), NFmiPoint(g->TopRight().X(), g->TopRight().Y()),
			    NFmiPoint(g->SouthPole().X(), g->SouthPole().Y()), NFmiPoint(0., 0.),  // default values
			    NFmiPoint(1., 1.),                                                     // default values
			    true);

			break;
		}

		case kStereographic:
		{
			auto g = std::dynamic_pointer_cast<stereographic_grid>(info.Grid());

			theArea = new NFmiStereographicArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
			                                    g->Di() * static_cast<double>((g->Ni() - 1)),
			                                    g->Dj() * static_cast<double>((g->Nj() - 1)), g->Orientation());
			break;
		}

		case kLambertConformalConic:
		{
			auto g = std::dynamic_pointer_cast<lambert_conformal_grid>(info.Grid());

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

	ASSERT(theArea);

#ifdef DEBUG
	OGRSpatialReference crs;
	const auto wkt = theArea->WKT();
	if (crs.SetFromUserInput(wkt.c_str()) == OGRERR_NONE)
	{
		char* proj4 = 0;
		crs.exportToProj4(&proj4);
		itsLogger.Trace(string(proj4));
		OGRFree(proj4);
	}
#endif

	NFmiGrid theGrid(theArea, std::dynamic_pointer_cast<regular_grid>(info.Grid())->Ni(),
	                 std::dynamic_pointer_cast<regular_grid>(info.Grid())->Nj());

	delete theArea;

	return NFmiHPlaceDescriptor(theGrid);
}

NFmiHPlaceDescriptor querydata::CreateHPlaceDescriptor(info<double>& info, bool activeOnly)
{
	if (!activeOnly && info.Size<forecast_time>() * info.Size<param>() * info.Size<level>() > 1)
	{
		info.Reset<forecast_time>();
		std::shared_ptr<grid> firstGrid = nullptr;

		while (info.Next<forecast_time>())
		{
			info.Reset<level>();

			while (info.Next<level>())
			{
				info.Reset<param>();

				while (info.Next<param>())
				{
					auto g = info.Grid();

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
						const auto fg_ = std::dynamic_pointer_cast<point_list>(firstGrid);
						auto g_ = std::dynamic_pointer_cast<point_list>(info.Grid());

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

NFmiVPlaceDescriptor querydata::CreateVPlaceDescriptor(info<double>& info, bool theActiveOnly)
{
	NFmiLevelBag lbag;

	if (theActiveOnly)
	{
		lbag.AddLevel(NFmiLevel(info.Level().Type(), HPLevelTypeToString.at(info.Level().Type()),
		                        static_cast<float>(info.Level().Value())));
	}
	else
	{
		info.Reset<level>();

		while (info.Next<level>())
		{
			lbag.AddLevel(NFmiLevel(info.Level().Type(), HPLevelTypeToString.at(info.Level().Type()),
			                        static_cast<float>(info.Level().Value())));
		}
	}

	return NFmiVPlaceDescriptor(lbag);
}

template <typename T>
shared_ptr<himan::info<T>> querydata::CreateInfo(shared_ptr<NFmiQueryData> theData) const
{
	auto newInfo = make_shared<info<T>>();
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

	newInfo->template Set<forecast_time>(theTimes);

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

	newInfo->template Set<level>(theLevels);

	// Parameters

	vector<himan::param> theParams;

	for (qinfo.ResetParam(); qinfo.NextParam();)
	{
		param par(string(qinfo.Param().GetParamName()), qinfo.Param().GetParamIdent());
		theParams.push_back(par);
	}

	newInfo->template Set<param>(theParams);

	vector<forecast_type> ftypes;
	ftypes.push_back(forecast_type(kDeterministic));

	newInfo->template Set<forecast_type>(ftypes);

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
			s->Ni(ni);
			s->Nj(nj);
		}
		break;

		default:
			itsLogger.Fatal("Invalid projection");
			himan::Abort();
	}

	dynamic_cast<regular_grid*>(newGrid)->ScanningMode(kBottomLeft);
	newGrid->EarthShape(earth_shape<double>(6371220));

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(newGrid->Clone());

	newInfo->Create(b);

	delete newGrid;

	// Copy data

	newInfo->template First<forecast_type>();

	for (newInfo->template Reset<forecast_time>(), qinfo.ResetTime();
	     newInfo->template Next<forecast_time>() && qinfo.NextTime();)
	{
		ASSERT(newInfo->template Index<forecast_time>() == qinfo.TimeIndex());

		for (newInfo->template Reset<level>(), qinfo.ResetLevel();
		     newInfo->template Next<level>() && qinfo.NextLevel();)
		{
			ASSERT(newInfo->template Index<level>() == qinfo.LevelIndex());

			for (newInfo->template Reset<param>(), qinfo.ResetParam();
			     newInfo->template Next<param>() && qinfo.NextParam();)
			{
				ASSERT(newInfo->template Index<param>() == qinfo.ParamIndex());

				matrix<T> dm(ni, nj, 1, static_cast<double>(32700.f));
				size_t i;

				for (qinfo.ResetLocation(), i = 0; qinfo.NextLocation() && i < ni * nj; i++)
				{
					dm.Set(i, static_cast<T>(qinfo.FloatValue()));
				}

				// convert kFloatMissing to nan
				dm.MissingValue(MissingDouble());
				b = newInfo->Base();
				b->data = move(dm);
			}
		}
	}

	return newInfo;
}

template shared_ptr<himan::info<double>> querydata::CreateInfo<double>(shared_ptr<NFmiQueryData>) const;
