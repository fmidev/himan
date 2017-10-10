#include "auto_taf.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "radon.h"
#include "util.h"
#include <algorithm>
#include <boost/format.hpp>

using namespace std;
using namespace himan;
using namespace himan::plugin;

struct cloud_layer
{
	cloud_layer() : base(MissingDouble()), amount(MissingDouble()), top(MissingDouble()){};
	~cloud_layer() = default;

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

bool CBLayer(const double& TC, const vector<double>& base, const vector<cloud_layer>& c_l, const double& cbbase,
             size_t i)
{
	if (TC > -50.0 && TC < 50.0)
		return false;
	else if (base.size() == 1)
		return true;
	else if (i == 0 && (abs(c_l.at(i).base - cbbase) <= abs(c_l.at(i + 1).base - cbbase)))
		return true;
	else if (i == 0)
		return false;
	else if (i == (base.size() - 1) && (abs(c_l.at(i).base - cbbase) < abs(c_l.at(i - 1).base - cbbase)))
		return true;
	else if (i == (base.size() - 1))
		return false;
	else if (abs(c_l.at(i).base - cbbase) > abs(c_l.at(i + 1).base - cbbase))
		return false;
	else if (abs(c_l.at(i).base - cbbase) >= abs(c_l.at(i - 1).base - cbbase))
		return false;
	else
		return true;
}

int CHClass(double base)
{
	int chval = 5;
	if (base < 200.0)
	{
		chval = 1;
	}
	else if (base < 500.0)
	{
		chval = 2;
	}
	else if (base < 1000.0)
	{
		chval = 3;
	}
	else if (base < 1500.0)
	{
		chval = 4;
	}
	return chval;
}

int CLClass(double amount)
{
	const double sct = .370;
	const double bkn = .620;
	const double ovc = .900;

	int clval = 8;
	if (amount < sct)
	{
		clval = 1;
	}
	else if (amount < bkn)
	{
		clval = 3;
	}
	else if (amount < ovc)
	{
		clval = 6;
	}
	return clval;
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
double LowestLayer(const vector<cloud_layer>& c_l, double threshold, size_t end)
{
	for (size_t i = 0; i < end; ++i)
	{
		if (c_l[i].amount > threshold) return c_l[i].base;
	}
	return MissingDouble();
}

auto_taf::auto_taf() : itsStrictMode(false) { itsLogger = logger("auto_taf"); }
void auto_taf::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param FEW("FEW-FT");
	param SCT("SCT-FT");
	param BKN("BKN-FT");
	param OVC("OVC-FT");
	param CB("CB-FT");
	param CBN("CB-PRCNT");

	CBN.Unit(kPrcnt);

	if (itsConfiguration->GetValue("strict") == "true")
	{
		itsStrictMode = true;
	}

	SetParams({FEW, SCT, BKN, OVC, CB, CBN});

	Start();
}

void auto_taf::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	// Required source parameters
	const params Nparam{param("N-0TO1"), param("N-PRCNT")};
	const param TCU_CB("CBTCU-FL");
	const param C2("CL-2-FT");
	const param LCL("LCL-HPA");

	// thresholds for cloud values
	const double cloud_treshold = 0.05;
	const double few = .130;  // 1/8
	const double sct = .370;  // 3/8
	const double bkn = .620;  // 5/8
	const double ovc = .900;  // 7/8

	// Get producer meta information
	producer prod = itsConfiguration->SourceProducer(0);
	auto r = GET_PLUGIN(radon);
	stringstream query, query2;

	query << "SELECT value FROM producer_meta "
	      << "WHERE attribute = 'first hybrid level number' "
	      << "AND producer_id = " << prod.Id();

	r->RadonDB().Query(query.str());
	vector<string> row = r->RadonDB().FetchRow();
	size_t firstLevel = boost::lexical_cast<size_t>(row[0]);

	query2 << "SELECT value FROM producer_meta "
	       << "WHERE attribute = 'last hybrid level number' "
	       << "AND producer_id = " << prod.Id();

	r->RadonDB().Query(query2.str());
	row = r->RadonDB().FetchRow();
	size_t lastLevel = boost::lexical_cast<size_t>(row[0]);
	// Finished get producer meta information

	auto myThreadedLogger = logger("auto_taf_pluginThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()));

	// find height of cb base and convert to feet
	level HL500 = level(kHeightLayer, 500, 0);
	info_t LCL500 = Fetch(forecastTime, HL500, LCL, forecastType, false);

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
	info_t TC;
	TC = Fetch(forecastTime, level(kHeight, 0.0), TCU_CB, forecastType, false);

	auto Ceiling2 = Fetch(forecastTime, level(kHeight, 0.0), C2, forecastType, false);
	for (auto&& tup : zip_range(cbbase, VEC(TC)))
	{
		double& _CB = tup.get<0>();
		double& _TC = tup.get<1>();
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

	vector<vector<bool>> cloud = vector<vector<bool>>(grd_size, vector<bool>(lastLevel, false));
	vector<vector<double>> top = vector<vector<double>>(grd_size, vector<double>());
	vector<vector<double>> base = vector<vector<double>>(grd_size, vector<double>());
	vector<vector<double>> sct_base = vector<vector<double>>(grd_size, vector<double>());
	vector<vector<double>> few_base = vector<vector<double>>(grd_size, vector<double>());
	vector<vector<double>> N_max = vector<vector<double>>(grd_size, vector<double>());

	size_t max_num_cl = 4;  // maximum number of cloud layers

	for (size_t j = lastLevel - 1; j > firstLevel + 1; --j)
	{
		auto N = Fetch(forecastTime, level(kHybrid, static_cast<double>(j + 1)), Nparam, forecastType, false);
		auto N_upper = Fetch(forecastTime, level(kHybrid, static_cast<double>(j)), Nparam, forecastType, false);
		auto N_upper_upper =
		    Fetch(forecastTime, level(kHybrid, static_cast<double>(j - 1)), Nparam, forecastType, false);
		auto Height =
		    Fetch(forecastTime, level(kHybrid, static_cast<double>(j + 1)), param("HL-M"), forecastType, false);
		for (size_t k = 0; k < grd_size; ++k)
		{
			const double& _N = N->Data().At(k);
			const double& _N_upper = N_upper->Data().At(k);
			const double& _N_upper_upper = N_upper_upper->Data().At(k);
			const double& _Height = Height->Data().At(k);

			if (_Height > 1000.0)
			{
				if (_N < cloud_treshold)
				{
					cloud[k][j] = false;
					if (base[k].size() > top[k].size())
					{
						double newbase = _Height / 0.3048;
						top[k].push_back(newbase);
					}
				}
				else
				{
					cloud[k][j + 1] = true;
					if (base[k].size() == top[k].size())
					{
						double newbase = _Height / 0.3048;
						base[k].push_back(newbase);
						N_max[k].push_back(_N);
						if (_N > sct)
						{
							sct_base[k].push_back(newbase);
						}
						else if (_N > few)
						{
							few_base[k].push_back(newbase);
						}
					}
					else if (_N > N_max[k].back())
					{
						N_max[k].back() = _N;
						double newbase = _Height / 0.3048;
						if (_N > sct && (sct_base[k].size() < base[k].size()))
						{
							sct_base[k].push_back(newbase);
						}
						else if (_N > few && (few_base[k].size() < base[k].size()))
						{
							few_base[k].push_back(newbase);
						}
					}
				}
			}
			else  // height is below 1000m
			{
				// three consecutive layers with cloud cover below threshold
				if ((_N < cloud_treshold) && (_N_upper < cloud_treshold) && (_N_upper_upper < cloud_treshold))
				{
					cloud[k][j] = false;
					// if we are above a cloud base cover it with a top.
					if (base[k].size() > top[k].size())
					{
						double newtop = _Height / 0.3048;
						top[k].push_back(newtop);
					}
				}
				else if ((_N > cloud_treshold) && (_N_upper > cloud_treshold) && (_N_upper_upper > cloud_treshold))
				{
					cloud[k][j] = true;
					if (base[k].size() == top[k].size())
					{
						double newbase = _Height / 0.3048;
						base[k].push_back(newbase);
						N_max[k].push_back(_N);
						if (_N > sct)
						{
							sct_base[k].push_back(newbase);
						}
						else if (_N > few)
						{
							few_base[k].push_back(newbase);
						}
					}
					else if (_N > N_max[k].back())
					{
						N_max[k].back() = _N;
						double newbase = _Height / 0.3048;
						if (_N > sct && (sct_base[k].size() < base[k].size()))
						{
							sct_base[k].push_back(newbase);
						}
						else if (_N > few && (sct_base[k].size() < base[k].size()))
						{
							few_base[k].push_back(newbase);
						}
					}
				}
				else if ((_N > cloud_treshold) && (base[k].size() > top[k].size()))
				{
					cloud[k][j] = true;
					if (_N > N_max[k].back())
					{
						N_max[k].back() = _N;
					}
				}
			}
			// update the maximum number of cloud layers in the whole grid
			if (base[k].size() > max_num_cl) max_num_cl = base[k].size();
		}
	}
	// end mark cloud layers

	// gather cloud data in cloud layer struct
	vector<vector<cloud_layer>> c_l = vector<vector<cloud_layer>>(grd_size, vector<cloud_layer>(max_num_cl));

	for (auto&& tup : zip_range(c_l, N_max, sct_base, few_base, base, top))
	{
		vector<cloud_layer>& _c_l = tup.get<0>();
		vector<double>& _N_max = tup.get<1>();
		vector<double>& _sct_base = tup.get<2>();
		vector<double>& _few_base = tup.get<3>();
		vector<double>& _base = tup.get<4>();
		vector<double>& _top = tup.get<5>();
		for (size_t j = 0; j < _base.size(); ++j)
		{
			if (_N_max[j] >= bkn && _sct_base.size() > j)
			{
				_c_l[j].base = _sct_base[j];
			}
			else if (_N_max[j] >= sct && _few_base.size() > j)
			{
				_c_l[j].base = _few_base[j];
			}
			else
			{
				_c_l[j].base = _base[j];
			}

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
				if (j == base[k].size() - 1)
				{
					c_l[k][j].base = Ceiling2->Data().At(k);
				}
				else if (c_l[k][j + 1].base > Ceiling2->Data().At(k))
				{
					c_l[k][j].base = Ceiling2->Data().At(k);
				}
				else
				{
					if (c_l[k][j].amount > c_l[k][j + 1].amount)
					{
						c_l[k][j + 1].amount = c_l[k][j].amount;
					}
					c_l[k][j].amount = .50;
				}
			}
		}
	}

	// cut off odd cloud layers
	for (size_t k = 0; k < grd_size; ++k)
	{
		if (base[k].size() < 3) continue;
		for (size_t j = 0; j < base[k].size() - 2; ++j)
		{
			if (CBLayer(TC->Data().At(k), base[k], c_l[k], cbbase[k], j) ==
			    CBLayer(TC->Data().At(k), base[k], c_l[k], cbbase[k], j + 1))
			{
				if (c_l[k][j].base >= 1500.0 && (CLClass(c_l[k][j + 1].amount) > CLClass(c_l[k][j].amount)))
				{
					if (RoundedBase(c_l[k][j + 1].base) - RoundedBase(c_l[k][j].base < 6.0))
					{
						c_l[k][j].amount = 0.0;
					}
				}
				else if (c_l[k][j].base >= 1500.0)
				{
					if (RoundedBase(c_l[k][j + 1].base) - RoundedBase(c_l[k][j].base) < 6.0)
					{
						c_l[k][j + 1].amount = 0.0;
					}
				}
				else if (CHClass(c_l[k][j].base) == CHClass(c_l[k][j + 1].base))
				{
					if (CLClass(c_l[k][j + 1].amount) > CLClass(c_l[k][j].amount))
					{
						c_l[k][j].amount = 0.0;
					}
					else
					{
						c_l[k][j + 1].amount = 0.0;
					}
				}
			}
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
		size_t n = base[k].size() - 1;
		if (c_l[k][3].amount > ovc || c_l[k][2].amount > ovc || c_l[k][1].amount > ovc || c_l[k][0].amount > ovc)
		{
			ovcbase[k] = LowestLayer(c_l[k], ovc, 4);
			continue;
		}
		if (abs(TC->Data().At(k)) > 50.0)
		{
			cbbase[k] = c_l[k][n].base;
			cbN[k] = c_l[k][n].amount * 100.0;  // cloud amount in %
		}
		if (c_l[k][2].amount > bkn || c_l[k][1].amount > bkn || c_l[k][0].amount > bkn)
		{
			bknbase[k] = LowestLayer(c_l[k], bkn, 3);
			continue;
		}
		if (c_l[k][1].amount > sct || c_l[k][0].amount > sct)
		{
			sctbase[k] = LowestLayer(c_l[k], sct, 2);
			continue;
		}
		if (c_l[k][0].amount > few)
		{
			fewbase[k] = c_l[k][0].base;
		}
	}

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
