#include "auto_taf.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "logger.h"
#include "plugin_factory.h"
#include "radon.h"
#include "util.h"
#include <algorithm>

using namespace std;
using namespace himan;
using namespace himan::plugin;

// thresholds for cloud values
const double cloud_treshold = 0.05;
const double few = .130;  // 1/8
const double sct = .370;  // 3/8
const double bkn = .620;  // 5/8
const double ovc = .900;  // 7/8

struct cloud_layer
{
	cloud_layer() : base(MissingDouble()), amount(MissingDouble()), top(MissingDouble()){};
	cloud_layer(double base, double amount, double top) : base(base), amount(amount), top(top){};

	double base;
	double amount;
	double top;
};

void CloudMin(double& base1, double& top1)
{
	double min = 0.5;
	if (base1 < min)
	{
		base1 = min;
		if (top1 <= base1)
		{
			top1 = base1 + 0.5;
		}
	}
}

double RoundedBase(double base)
{
	double rbase;
	if (base > 10000.0)
	{
		rbase = 1000.0 * floor((base / 1000.0) + 0.5);
	}
	else if (base > 2500.0)
	{
		rbase = 500.0 * floor((base / 500.0) + 0.5);
	}
	else
	{
		rbase = 100.0 * floor((base / 100.0) + 0.5);
	}
	return rbase;
}

// return the height of the LowestLayer cloud layer with cloudiness above given threshold in range [0,end)
double LowestLayer(const vector<cloud_layer>& c_l, double threshold, size_t& end)
{
	for (size_t i = 0; i < end; ++i)
	{
		if (c_l[i].amount > threshold)
		{
			end = i;
			return c_l[i].base;
		}
	}
	return MissingDouble();
}

auto_taf::auto_taf() { itsLogger = logger("auto_taf"); }
void auto_taf::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param FEW("FEW1-FT");
	param SCT("SCT1-FT");
	param BKN("BKN1-FT");
	param OVC("OVC1-FT");
	param CB("CB-FT");
	param CBN("CB-PRCNT");

	CBN.Unit(kPrcnt);

	SetParams({FEW, SCT, BKN, OVC, CB, CBN});

	Start();
}

void auto_taf::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
#ifdef DEBUG
	auto pxy = myTargetInfo->Grid()->XY(point(24.975133948461, 60.363380893204));
	auto pdebug =
	    myTargetInfo->Data().Index(static_cast<size_t>(round(pxy.X())), static_cast<size_t>(round(pxy.Y())), 0);
	cout << "index of debug point " << pdebug << '\n';

	vector<pair<double, double>> cloud_profile;
#endif
	// Required source parameters
	const params Nparam{param("N-0TO1"), param("N-PRCNT")};
	const param TCU_CB("CBTCU-FL");
	const param C2("CL-2-FT");
	const param LCL("LCL-HPA");

	// Get producer meta information
	producer prod = itsConfiguration->SourceProducer(0);
	auto r = GET_PLUGIN(radon);
	size_t firstLevel = stoi(r->RadonDB().GetProducerMetaData(prod.Id(), "first hybrid level number"));
	size_t lastLevel = stoi(r->RadonDB().GetProducerMetaData(prod.Id(), "last hybrid level number"));
	// Finished get producer meta information

	auto myThreadedLogger = logger("auto_taf_pluginThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()));

	// find height of cb base and convert to feet
	level HL500 = level(kHeightLayer, 500, 0);

	info_t LCL500 = Fetch(forecastTime, HL500, LCL, forecastType, false);
	if (!LCL500)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()));
		return;
	}

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	auto bounds = minmax_element(VEC(LCL500).begin(), VEC(LCL500).end());

	auto levelsMax = h->LevelForHeight(myTargetInfo->Producer(), *(bounds.second));
	auto levelsMin = h->LevelForHeight(myTargetInfo->Producer(), *(bounds.first));

	h->HeightUnit(kM);

	auto cbbase = h->VerticalHeight(param("P-HPA"), levelsMin.second.Value(), levelsMax.first.Value(), VEC(LCL500));

	info_t TC = Fetch(forecastTime, level(kHeight, 0.0), TCU_CB, forecastType, false);
	info_t Ceiling2 = Fetch(forecastTime, level(kHeight, 0.0), C2, forecastType, false);

	// search for lowest cloud layer
	auto few_lowest = h->VerticalHeight(param("N-0TO1"), 0.0, 10000.0, few, 1);
	auto sct_lowest = h->VerticalHeight(param("N-0TO1"), 0.0, 10000.0, sct, 1);

	for_each(few_lowest.begin(), few_lowest.end(), [](double& val) { val /= 0.3048; });
	for_each(sct_lowest.begin(), sct_lowest.end(), [](double& val) { val /= 0.3048; });

	if (!TC || !Ceiling2)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()));
		return;
	}

	for (auto&& tup : zip_range(cbbase, VEC(TC)))
	{
		double& _CB = tup.get<0>();
		const double& _TC = tup.get<1>();
		if (abs(_TC) > 50.0)
		{
			_CB = _CB / 0.3048;
		}
		else
		{
			_CB = MissingDouble();
		}
	}
	// end find height of cb base

	// mark cloud layers
	size_t grd_size = VEC(TC).size();

	vector<vector<double>> top = vector<vector<double>>(grd_size, vector<double>());
	vector<vector<double>> base = vector<vector<double>>(grd_size, vector<double>());
	vector<vector<double>> N_max = vector<vector<double>>(grd_size, vector<double>());

	size_t max_num_cl = 4;  // maximum number of cloud layers

	for (size_t j = lastLevel - 1; j > firstLevel; --j)
	{
		info_t N = Fetch(forecastTime, level(kHybrid, static_cast<double>(j)), Nparam, forecastType, false);
		info_t Height = Fetch(forecastTime, level(kHybrid, static_cast<double>(j)), param("HL-M"), forecastType, false);

		if (!N || !Height)
		{
			myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " + to_string(j));
			continue;
		}

		for (size_t k = 0; k < grd_size; ++k)
		{
			const double& _N = N->Data().At(k);
			const double& _Height = Height->Data().At(k);

			if (_N < cloud_treshold)
			{
				if (base[k].size() > top[k].size())
				{
					double newtop = _Height / 0.3048;
					top[k].push_back(newtop);
				}
			}
			else
			{
				if (base[k].size() == top[k].size())
				{
					double newbase = _Height / 0.3048;
					base[k].push_back(newbase);
					N_max[k].push_back(_N);
				}
				else if (_N > N_max[k].back())
				{
					N_max[k].back() = _N;
				}
			}
#ifdef DEBUG
			if (k == pdebug)
			{
				cloud_profile.emplace_back(_Height / 0.3048, _N);
			}
#endif
			// update the maximum number of cloud layers in the whole grid
			max_num_cl = max(max_num_cl, base[k].size());
		}
	}
	// end mark cloud layers

	// gather cloud data in cloud layer struct
	vector<vector<cloud_layer>> c_l = vector<vector<cloud_layer>>(grd_size, vector<cloud_layer>(max_num_cl));

	for (auto&& tup : zip_range(c_l, N_max, base, top))
	{
		vector<cloud_layer>& _c_l = tup.get<0>();
		vector<double>& _N_max = tup.get<1>();
		vector<double>& _base = tup.get<2>();
		vector<double>& _top = tup.get<3>();
		for (size_t j = 0; j < _base.size(); ++j)
		{
			_c_l[j].base = _base[j];
			_c_l[j].top = _top[j];
			_c_l[j].amount = _N_max[j];
		}
	}

	// check LowestLayer base is above ground
	for (auto&& tup : zip_range(c_l, base))
	{
		vector<cloud_layer>& _c_l = tup.get<0>();
		vector<double>& _base = tup.get<1>();

		if (_base.size() > 0)
		{
			CloudMin(_c_l[0].base, _c_l[0].top);
		}
	}

	// change the corridors below the ceiling2 from the LowestLayer bkn layer to the sct layers
	for (size_t k = 0; k < grd_size; ++k)
	{
		for (size_t j = 0; j < base[k].size(); ++j)
		{
			if (c_l[k][j].base < Ceiling2->Data().At(k) && c_l[k][j].amount >= bkn)
			{
				if (c_l[k][j].top > Ceiling2->Data().At(k))
				{
					c_l[k][j].base = Ceiling2->Data().At(k);
				}
				else
				{
					c_l[k][j].amount = sct;
				}
			}
		}
	}

	// cut off odd cloud layers
	for (size_t k = 0; k < grd_size; ++k)
	{
		if (base[k].size() == 0) continue;

		// is there a cload layer in height class
		vector<bool> height_class(4, false);
		vector<double> height_bounds = {200.0, 500.0, 1000.0, 1400.0};
		vector<size_t> layers_to_remove;

		for (size_t j = 0; j < base[k].size(); ++j)
		{
			// low cloud height class <200ft
			if (!height_class[0] && c_l[k][j].base < height_bounds[0])
			{
				height_class[0] = true;
				continue;
			}
			else if (c_l[k][j].base < height_bounds[0])
			{
				layers_to_remove.push_back(j);
				continue;
			}

			// low cloud height class 200ft-500ft
			if (!height_class[1] && c_l[k][j].base < height_bounds[1] && c_l[k][j].base >= height_bounds[0])
			{
				height_class[1] = true;
				continue;
			}
			else if (c_l[k][j].base < height_bounds[1] && c_l[k][j].base >= height_bounds[0])
			{
				layers_to_remove.push_back(j);
				continue;
			}

			// low cloud height class 500ft-1000ft
			if (!height_class[2] && c_l[k][j].base < height_bounds[2] && c_l[k][j].base >= height_bounds[1])
			{
				height_class[2] = true;
				continue;
			}
			else if (c_l[k][j].base < height_bounds[2] && c_l[k][j].base >= height_bounds[1])
			{
				layers_to_remove.push_back(j);
				continue;
			}

			// low cloud height class 1000ft-1500ft
			if (!height_class[3] && c_l[k][j].base < height_bounds[3] && c_l[k][j].base >= height_bounds[2])
			{
				height_class[3] = true;
				continue;
			}
			else if (c_l[k][j].base < height_bounds[3] && c_l[k][j].base >= height_bounds[2])
			{
				layers_to_remove.push_back(j);
				continue;
			}

			// heights above 1500ft
			if (j != 0 && RoundedBase(c_l[k][j].base) - RoundedBase(c_l[k][j - 1].base) < 500.0)
			{
				layers_to_remove.push_back(j);
			}
		}
		while (layers_to_remove.size() > 0)
		{
			c_l[k].erase(c_l[k].begin() + layers_to_remove.back());
			layers_to_remove.pop_back();
		}
	}

	// write cloud data to output parameters
	vector<double> ovcbase(grd_size, MissingDouble());
	vector<double> bknbase(grd_size, MissingDouble());
	vector<double> sctbase(grd_size, MissingDouble());
	vector<double> fewbase(grd_size, MissingDouble());

	vector<double> cbN(grd_size, MissingDouble());

	for (size_t k = 0; k < grd_size; ++k)
	{
		size_t n = base[k].size();
		size_t m = 4;

		if (n == 0)
		{
			// No cloud layers for this grid point
			continue;
		}

		--n;

		if (abs(TC->Data().At(k)) > 50.0)
		{
			// find nearest cloud layer above or equal cbbase
			auto nearest_cl =
			    lower_bound(c_l[k].begin(), c_l[k].end(), cloud_layer(cbbase[k], MissingDouble(), MissingDouble()),
			                [](const cloud_layer& a, const cloud_layer& b) { return a.base < b.base; });

			// no cloud layers above initial cbbase, use highest cloud layer
			if (nearest_cl == c_l[k].end())
			{
				cbbase[k] = (--nearest_cl)->base;
			}
			// nearest cloud layer is lowest cloud layer
			else if (nearest_cl == c_l[k].begin())
			{
				// pass
			}
			// select closest of layers above and below cbbase
			else if (nearest_cl->base - cbbase[k] < cbbase[k] - (--nearest_cl)->base)
			{
				++nearest_cl;
			}

			if (nearest_cl == c_l[k].begin() && cbbase[k] - nearest_cl->base > 500.0)
			{
				// create cblayer as new cloud layer
				c_l[k].insert(nearest_cl + 1, cloud_layer(cbbase[k], nearest_cl->amount, MissingDouble()));
				if (nearest_cl->amount >= ovc)
				{
					cbN[k] = nearest_cl->amount * 100.0;
					nearest_cl->amount = bkn;
					m = 1;
				}
				else
				{
					cbN[k] = (nearest_cl)->amount * 100.0;
					m = 5;
				}
			}
			else
			{
				cbbase[k] = nearest_cl->base;
				cbN[k] = nearest_cl->amount * 100.0;  // cloud amount in %
			}

			if (IsMissing(cbN[k])) cbbase[k] = MissingDouble();
		}

		ovcbase[k] = LowestLayer(c_l[k], ovc, m);
		if (cbbase[k] > ovcbase[k]) cbbase[k] = MissingDouble();
		if (ovcbase[k] == cbbase[k]) ovcbase[k] = MissingDouble();
		if (m == 0) continue;

		bknbase[k] = LowestLayer(c_l[k], bkn, m = min(m, size_t(3)));
		if (bknbase[k] >= cbbase[k]) bknbase[k] = MissingDouble();
		if (m == 0) continue;

		sctbase[k] = LowestLayer(c_l[k], sct, m = min(m, size_t(2)));
		if (sctbase[k] >= cbbase[k]) sctbase[k] = MissingDouble();
		if (m == 0) continue;

		fewbase[k] = LowestLayer(c_l[k], cloud_treshold, m = min(m, size_t(1)));
		if (fewbase[k] >= cbbase[k]) fewbase[k] = MissingDouble();
		if (fewbase[k] < sctbase[k] && few_lowest[k] > sctbase[k]) sctbase[k] = few_lowest[k];
		if (fewbase[k] < bknbase[k] && sct_lowest[k] > bknbase[k]) bknbase[k] = sct_lowest[k];
		if (fewbase[k] < ovcbase[k] && sct_lowest[k] > ovcbase[k]) ovcbase[k] = sct_lowest[k];
	}

#ifdef DEBUG
	for (size_t k = 0; k < grd_size; ++k)
	{
		// Check against illegal outcomes for TAF
		assert(!(fewbase[k] > sctbase[k]));
		assert(!(sctbase[k] > bknbase[k]));
		assert(!(bknbase[k] > ovcbase[k]));
		assert(!(cbbase[k] >= ovcbase[k]));
		assert(!(bknbase[k] >= cbbase[k]));
		assert(!(sctbase[k] >= cbbase[k]));
		assert(!(fewbase[k] >= cbbase[k]));
		assert(!(bknbase[k] < TC->Data().At(k)));
		assert(!(ovcbase[k] < TC->Data().At(k)));

		if (k == pdebug)
		{
			auto p = myTargetInfo->Grid()->LatLon(pdebug);
			cout << p.X() << " " << p.Y() << " " << k << '\n';
			cout << "few " << fewbase[k] << '\n';
			cout << "sct " << sctbase[k] << '\n';
			cout << "bkn " << bknbase[k] << '\n';
			cout << "ovc " << ovcbase[k] << '\n';
			cout << "cb " << cbbase[k] << '\n';
			cout << "tcu " << TC->Data().At(k) << '\n';
			cout << "ceil2 " << Ceiling2->Data().At(k) << '\n';
			for (auto& x : c_l[k])
			{
				cout << "base " << x.base << " top " << x.top << " amount " << x.amount << '\n';
			}
		}
	}
	for (auto& x : cloud_profile)
	{
		cout << "height " << x.first << " N " << x.second << '\n';
	}
#endif

	myTargetInfo->ParamIndex(0);
	myTargetInfo->Grid()->Data().Set(move(fewbase));

	myTargetInfo->ParamIndex(1);
	myTargetInfo->Grid()->Data().Set(move(sctbase));

	myTargetInfo->ParamIndex(2);
	myTargetInfo->Grid()->Data().Set(move(bknbase));

	myTargetInfo->ParamIndex(3);
	myTargetInfo->Grid()->Data().Set(move(ovcbase));

	myTargetInfo->ParamIndex(4);
	myTargetInfo->Grid()->Data().Set(move(cbbase));

	myTargetInfo->ParamIndex(5);
	myTargetInfo->Grid()->Data().Set(move(cbN));

	string deviceType = "CPU";
	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
