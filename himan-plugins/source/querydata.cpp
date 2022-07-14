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
#include "util.h"
#include <boost/filesystem.hpp>
#include <fstream>

#include "plugin_factory.h"
#include "radon.h"

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#endif

#include "newbase/NFmiFastQueryInfo.h"
#include <newbase/NFmiGdalArea.h>
#include <newbase/NFmiLatLonArea.h>
#include <newbase/NFmiQueryData.h>
#include <newbase/NFmiRotatedLatLonArea.h>
#include <newbase/NFmiStereographicArea.h>
#include <newbase/NFmiTimeList.h>

#ifdef __clang__

#pragma clang diagnostic pop

#endif

using namespace std;
using namespace himan::plugin;

template <typename T>
bool CopyData(himan::info<T>& theInfo, NFmiFastQueryInfo& qinfo, bool applyScaleAndBase)
{
	ASSERT(theInfo.Data().Size() == qinfo.SizeLocations());

	theInfo.ResetLocation();
	qinfo.ResetLocation();

	float scale = 1, base = 0;

	if (applyScaleAndBase)
	{
		scale = static_cast<float>(theInfo.Param().Scale());
		base = static_cast<float>(theInfo.Param().Base());
	}

	if (theInfo.Grid()->Class() == himan::kRegularGrid &&
	    dynamic_pointer_cast<himan::regular_grid>(theInfo.Grid())->ScanningMode() != himan::kBottomLeft)
	{
		ASSERT(std::dynamic_pointer_cast<himan::regular_grid>(theInfo.Grid())->ScanningMode() == himan::kTopLeft);

		size_t nj = theInfo.Data().SizeY();
		size_t ni = theInfo.Data().SizeX();

		int y = static_cast<int>(nj) - 1;

		do
		{
			size_t x = 0;

			do
			{
				qinfo.NextLocation();
				const T val = theInfo.Data().At(x, y);

				if (himan::IsValid(val))
				{
					qinfo.FloatValue(static_cast<float>(val) * scale + base);
				}
				else
				{
					qinfo.FloatValue(kFloatMissing);
				}
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
			const float val = static_cast<float>(theInfo.Value());

			if (val != kFloatMissing)
			{
				qinfo.FloatValue(val * scale + base);
			}
		}
	}

	// return to original missing value
	theInfo.Data().MissingValue(himan::MissingValue<T>());

	return true;
}

template <typename T>
NFmiTimeDescriptor CreateTimeDescriptor(himan::info<T>& info, bool theActiveOnly)
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
		info.template Reset<himan::forecast_time>();

		himan::raw_time firstOriginTime;

		while (info.template Next<himan::forecast_time>())
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
					throw std::runtime_error("Origintime is not the same for all grids in info");
				}
			}

			tlist.Add(new NFmiMetTime(stol(info.Time().ValidDateTime().String("%Y%m%d")),
			                          stol(info.Time().ValidDateTime().String("%H%M"))));
		}
	}

	return NFmiTimeDescriptor(originTime, tlist);
}

template <typename T>
void AddToParamBag(himan::info<T>& info, NFmiParamBag& pbag)
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

template <typename T>
NFmiHPlaceDescriptor CreateGrid(himan::info<T>& info)
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
		case himan::kLatitudeLongitude:
		{
			auto g = std::dynamic_pointer_cast<himan::latitude_longitude_grid>(info.Grid());

			theArea = new NFmiLatLonArea(NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()),
			                             NFmiPoint(g->TopRight().X(), g->TopRight().Y()));

			break;
		}

		case himan::kRotatedLatitudeLongitude:
		{
			auto g = std::dynamic_pointer_cast<himan::rotated_latitude_longitude_grid>(info.Grid());
			theArea = new NFmiRotatedLatLonArea(
			    NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()), NFmiPoint(g->TopRight().X(), g->TopRight().Y()),
			    NFmiPoint(g->SouthPole().X(), g->SouthPole().Y()), NFmiPoint(0., 0.),  // default values
			    NFmiPoint(1., 1.),                                                     // default values
			    false);

			break;
		}

		case himan::kStereographic:
		{
			auto g = std::dynamic_pointer_cast<himan::stereographic_grid>(info.Grid());

			theArea = new NFmiStereographicArea(
			    NFmiPoint(g->BottomLeft().X(), g->BottomLeft().Y()), g->Di() * static_cast<double>((g->Ni() - 1)),
			    g->Dj() * static_cast<double>((g->Nj() - 1)), g->Orientation(), NFmiPoint(0.f, 0.f),
			    NFmiPoint(1.f, 1.f), g->LatitudeOfCenter(), g->LatitudeOfOrigin());
			break;
		}

		case himan::kLambertConformalConic:
		case himan::kTransverseMercator:
		{
			auto g = std::dynamic_pointer_cast<himan::regular_grid>(info.Grid());

			std::stringstream ss;
			ss << "GEOGCS[\"MEPS\","
			   << " DATUM[\"unknown\","
			   << "     SPHEROID[\"Sphere\",6367470,0]],"
			   << " PRIMEM[\"Greenwich\",0],"
			   << " UNIT[\"degree\",0.0174532925199433]]";

			theArea =
			    new NFmiGdalArea(ss.str(), *g->SpatialReference(), 0, 0, g->Di() * (static_cast<double>(g->Ni()) - 1),
			                     g->Dj() * (static_cast<double>(g->Nj()) - 1));
			break;
		}

		default:
			himan::logger logr("querydata");
			logr.Fatal(fmt::format("'{}' projection not supported", himan::HPGridTypeToString.at(info.Grid()->Type())));
			himan::Abort();
	}

	ASSERT(theArea);

	NFmiGrid theGrid(theArea, std::dynamic_pointer_cast<himan::regular_grid>(info.Grid())->Ni(),
	                 std::dynamic_pointer_cast<himan::regular_grid>(info.Grid())->Nj());

	delete theArea;

	return NFmiHPlaceDescriptor(theGrid);
}

template <typename T>
NFmiParamDescriptor CreateParamDescriptor(himan::info<T>& info, bool theActiveOnly)
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
		info.template Reset<himan::param>();

		while (info.template Next<himan::param>())
		{
			AddToParamBag(info, pbag);
		}
	}

	return NFmiParamDescriptor(pbag);
}

template <typename T>
NFmiHPlaceDescriptor CreatePoint(himan::info<T>& info)
{
	const auto g = std::dynamic_pointer_cast<himan::point_list>(info.Grid());
	NFmiLocationBag bag;

	for (const himan::station& s : g->Stations())
	{
		NFmiStation stat(s.Id(), s.Name(), s.X(), s.Y());
		bag.AddLocation(stat);
	}

	return NFmiHPlaceDescriptor(bag);
}

template <typename T>
NFmiHPlaceDescriptor CreateHPlaceDescriptor(himan::info<T>& info, bool activeOnly)
{
	himan::logger log("querydata");
	if (!activeOnly && info.template Size<himan::forecast_time>() * info.template Size<himan::param>() *
	                           info.template Size<himan::level>() >
	                       1)
	{
		info.template Reset<himan::forecast_time>();
		std::shared_ptr<himan::grid> firstGrid = nullptr;

		while (info.template Next<himan::forecast_time>())
		{
			info.template Reset<himan::level>();

			while (info.template Next<himan::level>())
			{
				info.template Reset<himan::param>();

				while (info.template Next<himan::param>())
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
						log.Error("All grids in info are not equal, unable to write querydata");
						return NFmiHPlaceDescriptor();
					}

					if (firstGrid->Class() == himan::kRegularGrid)
					{
						if (*firstGrid != *g)
						{
							log.Error("All grids in info are not equal, unable to write querydata");
							return NFmiHPlaceDescriptor();
						}
					}
					else
					{
						const auto fg_ = std::dynamic_pointer_cast<himan::point_list>(firstGrid);
						auto g_ = std::dynamic_pointer_cast<himan::point_list>(info.Grid());

						if (*fg_ != *g_)
						{
							log.Error("All grids in info are not equal, unable to write querydata");
							return NFmiHPlaceDescriptor();
						}
					}
				}
			}
		}
	}

	if (info.Grid()->Class() == himan::kRegularGrid)
	{
		return CreateGrid<T>(info);
	}
	else
	{
		return CreatePoint<T>(info);
	}
}

template <typename T>
NFmiVPlaceDescriptor CreateVPlaceDescriptor(himan::info<T>& info, bool theActiveOnly)
{
	NFmiLevelBag lbag;

	if (theActiveOnly)
	{
		lbag.AddLevel(NFmiLevel(info.Level().Type(), himan::HPLevelTypeToString.at(info.Level().Type()),
		                        static_cast<float>(info.Level().Value())));
	}
	else
	{
		info.template Reset<himan::level>();

		while (info.template Next<himan::level>())
		{
			lbag.AddLevel(NFmiLevel(info.template Level().Type(),
			                        himan::HPLevelTypeToString.at(info.template Level().Type()),
			                        static_cast<float>(info.template Level().Value())));
		}
	}

	return NFmiVPlaceDescriptor(lbag);
}

querydata::querydata()
{
	itsLogger = logger("querydata");
}

pair<himan::HPWriteStatus, himan::file_information> querydata::ToFile(info<double>& theInfo)
{
	return ToFile<double>(theInfo);
}

template <typename T>
pair<himan::HPWriteStatus, himan::file_information> querydata::ToFile(info<T>& theInfo)
{
	file_information finfo;
	finfo.file_location = util::MakeFileName(theInfo, *itsWriteOptions.configuration);
	finfo.file_type = kQueryData;
	finfo.storage_type = itsWriteOptions.configuration->WriteStorageType();

	boost::filesystem::path pathname(finfo.file_location);

	if (!pathname.parent_path().empty() && !boost::filesystem::is_directory(pathname.parent_path()))
	{
		boost::filesystem::create_directories(pathname.parent_path());
	}

	ofstream out(finfo.file_location.c_str());

	bool activeOnly = true;

	if (itsWriteOptions.configuration->WriteMode() == kAllGridsToAFile)
	{
		activeOnly = false;
	}
	else if (itsWriteOptions.configuration->WriteMode() == kFewGridsToAFile)
	{
		itsLogger.Fatal("Write option 'few' is not supported with querydata");
		himan::Abort();
	}

	auto qdata = CreateQueryData<T>(theInfo, activeOnly, true);

	if (!qdata)
	{
		throw kInvalidWriteOptions;
	}

	out << *qdata;

	itsLogger.Info("Wrote file '" + finfo.file_location + "'");

	return make_pair(HPWriteStatus::kFinished, finfo);
}

template pair<himan::HPWriteStatus, himan::file_information> querydata::ToFile<double>(info<double>&);
template pair<himan::HPWriteStatus, himan::file_information> querydata::ToFile<float>(info<float>&);
template pair<himan::HPWriteStatus, himan::file_information> querydata::ToFile<short>(info<short>&);
template pair<himan::HPWriteStatus, himan::file_information> querydata::ToFile<unsigned char>(info<unsigned char>&);

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

		CopyData<T>(localInfo, qinfo, applyScaleAndBase);
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

					CopyData<T>(localInfo, qinfo, applyScaleAndBase);
				}
			}
		}
	}

	qdata->LatLonCache();

	return qdata;
}

template shared_ptr<NFmiQueryData> querydata::CreateQueryData<double>(const info<double>&, bool, bool);
template shared_ptr<NFmiQueryData> querydata::CreateQueryData<float>(const info<float>&, bool, bool);
template shared_ptr<NFmiQueryData> querydata::CreateQueryData<short>(const info<short>&, bool, bool);
template shared_ptr<NFmiQueryData> querydata::CreateQueryData<unsigned char>(const info<unsigned char>&, bool, bool);

template <typename T>
shared_ptr<himan::info<T>> querydata::CreateInfo(shared_ptr<NFmiQueryData> theData) const
{
	grid* newGrid = 0;

	NFmiQueryInfo qinfo = theData.get();

	// Times

	vector<forecast_time> times;

	raw_time originTime(string(qinfo.OriginTime().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");

	for (qinfo.ResetTime(); qinfo.NextTime();)
	{
		raw_time ct(string(qinfo.Time().ToStr(kYYYYMMDDHHMM)), "%Y%m%d%H%M");

		times.emplace_back(originTime, ct);
	}

	// Levels

	vector<level> levels;

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

			case kFmiPressureLevel:
				lt = kPressure;
				break;

			case kFmiNoLevelType:
			case kFmiAnyLevelType:
				break;

			default:
				throw runtime_error("Unknown level type in querydata: " + to_string(qinfo.Level()->LevelType()));
				break;
		}

		levels.emplace_back(lt, qinfo.Level()->LevelValue());
	}

	// Parameters

	vector<himan::param> params;

	for (qinfo.ResetParam(); qinfo.NextParam();)
	{
		params.emplace_back(string(qinfo.Param().GetParamName()), qinfo.Param().GetParamIdent());
	}

	vector<forecast_type> ftypes{forecast_type(kDeterministic)};

	auto newInfo = make_shared<info<T>>(ftypes, times, levels, params);
	producer p(230, 86, 230, "HIMAN");
	p.TableVersion(203);

	newInfo->Producer(p);

	// Grid

	const size_t ni = qinfo.Grid()->XNumber();
	const size_t nj = qinfo.Grid()->YNumber();

	switch (qinfo.Area()->ClassId())
	{
		case kNFmiLatLonArea:
			newGrid = new latitude_longitude_grid(
			    kBottomLeft, point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()),
			    point(qinfo.Area()->TopRightLatLon().X(), qinfo.Area()->TopRightLatLon().Y()), ni, nj,
			    earth_shape<double>(6371220.));
			break;

		case kNFmiRotatedLatLonArea:
		{
			const NFmiPoint southPole = reinterpret_cast<const NFmiRotatedLatLonArea*>(qinfo.Area())->SouthernPole();

			newGrid = new rotated_latitude_longitude_grid(
			    kBottomLeft, point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()),
			    point(qinfo.Area()->TopRightLatLon().X(), qinfo.Area()->TopRightLatLon().Y()), ni, nj,
			    earth_shape<double>(6371220.), point(southPole.X(), southPole.Y()));
		}
		break;

		case kNFmiStereographicArea:
		{
			const double di = qinfo.Area()->WorldXYWidth() / static_cast<double>(qinfo.Grid()->XNumber());
			const double dj = qinfo.Area()->WorldXYHeight() / static_cast<double>(qinfo.Grid()->YNumber());

			newGrid = new stereographic_grid(
			    kBottomLeft, point(qinfo.Area()->BottomLeftLatLon().X(), qinfo.Area()->BottomLeftLatLon().Y()), ni, nj,
			    di, dj, reinterpret_cast<const NFmiStereographicArea*>(qinfo.Area())->Orientation(),
			    earth_shape<double>(6371220.));
		}
		break;

		default:
			itsLogger.Fatal("Invalid projection");
			himan::Abort();
	}

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

				matrix<T> dm(ni, nj, 1, MissingValue<T>());
				size_t i;

				for (qinfo.ResetLocation(), i = 0; qinfo.NextLocation() && i < ni * nj; i++)
				{
					const float val = qinfo.FloatValue();

					if (val == kFloatMissing)
					{
						dm.Set(i, MissingValue<T>());
					}
					else
					{
						dm.Set(i, static_cast<T>(val));
					}
				}

				b = newInfo->Base();
				b->data = move(dm);
			}
		}
	}

	return newInfo;
}

template shared_ptr<himan::info<double>> querydata::CreateInfo<double>(shared_ptr<NFmiQueryData>) const;
template shared_ptr<himan::info<float>> querydata::CreateInfo<float>(shared_ptr<NFmiQueryData>) const;
template shared_ptr<himan::info<short>> querydata::CreateInfo<short>(shared_ptr<NFmiQueryData>) const;
template shared_ptr<himan::info<unsigned char>> querydata::CreateInfo<unsigned char>(shared_ptr<NFmiQueryData>) const;
