#include "forecast_time.h"
#include "info.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "auto_taf.h"


using namespace std;
using namespace himan::plugin;

struct cloud_layer
{
	cloud_layer():base(MissingDouble()),amount(MissingDouble()),top(MissingDouble()){};

	double base;
	double amount;
	double top;
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

	POT.Unit(kPrcnt);

	if (itsConfiguration->GetValue("strict") == "true")
	{
		itsStrictMode = true;
	}

	SetParams({FEW,SCT,BKN,OVC,CB});

	Start();
}

void auto_taf::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	// Required source parameters
	const param CL("CL-FT");
	const param TCU_CB("TCU_CB-FL");
	const param C2("CL-2-FT");
	const param LCL("LCL-HPA");

	// thresholds for cloud values
	const double cloud_threshold = 5.0;
	const double few = 13.0;
	const double sct = 37.0;
	const double bkn = 62.0;
	const double ovc = 90.0;

	// Get producer meta information
	auto r = GET_PLUGIN(radon);
	stringstream query;

	query << "SELECT value FROM producer_meta "
	      << "WHERE attribute = 'first hybrid level number' "
	      << "AND producer_id = " << producerId;

	r->RadonDB().Query(query.str());
	size_t firstLevel = r->RadonDB().FetchRow();

        query << "SELECT value FROM producer_meta "
              << "WHERE attribute = 'last hybrid level number' "
              << "AND producer_id = " << producerId;

        r->RadonDB().Query(query.str());
	size_t lastLevel = r->RadonDB().FetchRow();
	// Finished get producer meta information

	auto myThreadedLogger = logger("auto_taf_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	forecast_time startTime = myTargetInfo->Time();

	// find height of cb base and convert to feet
	level HL500 = level(HPLevelType.kHeightLayer,500,0);
	info_t LCL500 = Fetch(forecastTime, HL500, LCL, forecastType, false);
	
        auto h = GET_PLUGIN(hitool);
        h->Configuration(itsConfiguration);
        h->Time(myTargetInfo->Time());
        h->ForecastType(myTargetInfo->ForecastType());
        h->HeightUnit(kHPa);

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), LCL500);

        h->HeightUnit(kM);

	auto cbbase = h->VerticalHeight(param("P-HPA"), levels.first, levels.second, VEC(LCL500));
	auto TC = Fetch(forecastTime, level(kGround), TCU_CB, forecastType, false);

	for (auto&& tup : zip_range(cbbase, VEC(TC)))
	{
		double& CB = tup.get<0>();
		double& TC = tup.get<1>();
		if (abs(TC) > 50.0)
		{
			CB=CB/0.3048;
		}
		else
		{
			CB = MissingDouble();
		}
	}
	// end find height of cb base

	// mark cloud layers
	size_t grd_size, lvl_size;

	grd_size = VEC(TC).size();
	lvl_size = lastLevel - firstLevel + 1;

	vector<vector<bool>> cloud = vector<vector<bool>>(grd_size,vector<bool>(lastLevel,false));
	vector<vector<double>> top = vector<vector<double>>(grd_size,vector<double>());
	vector<vector<double>> base = vector<vector<double>>(grd_size,vector<double>());
	vector<vector<double>> N_max = vector<vector<double>>(grd_size,vector<double>);

	for (size_t j = lastLevel-1; j > firstLevel; --j)
	{
		auto N = Fetch(forecastTime,level(kHybrid,j+1),CL,forecastType,false);
		auto N_upper = Fetch(forecastTime,level(kHybrid,j),CL,forecastType,false);
                auto N_uper_upper = Fetch(forecastTime,level(kHybrid,j-1),CL,forecastType,false);

		auto Height = Fetch(forecastTime,level(kHybrid,j+1),param("HL-M"),forecastType,false);
		for (size_t k = 0; k < grd_size; ++k)
		{
			const double& _N = N->Value(k);
			const double& _N_upper = N_upper->Value(k);
			const double& _N_upper_upper = N_upper_upper->Value(k);
			const double& _Height = Height->Value(k);

			if(_Height > 1000.0)
			{
				if (_N<cloud_treshold) 
				{
					cloud[k][j]=false;
					if (base[k].size() > top[k].size())
					{
						double newtop = middle(a,j-1,r,loc);
						top[k].push_back(newtop);
					}
				}
				else
				{
					cloud[k][j+1]=true;
					if (base[k].size() <= top[k].size())
					{
						double newbase = middle(a,j-1,r,loc);
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
					else if (_N > N_max[k][base[k].size()])
					{
						N_max[k][base[k].size()] = _N;
						if (_N > sct && (sct_base[k].size() < base[k].size()))
						{
							sct_base[k][base[k].size()] = middle(a,j-1,r,loc);
						}
						else if (_N > few && (sct_base[k].size() < base[k].size()))
						{
							few_base[k][base[k].size()] = middle(a,j-1,r,loc);
						}
					}
				}
			}
			else
			{
				// three consecutive layers with cloud cover below threshold
				if ((_N < cloud_treshold) && (_N_upper < cloud_treshold) && (_N_upper_upper < cloud_treshold))
				{
					cloud[k][j]=false;
					// if we are above a cloud base cover it with a top.
                                        if (base[k].size() > top[k].size())
                                        {
                                                double newtop = middle(a,j-1,r,loc);
                                                top[k].push_back(newtop);
                                        }

				}
				else if ((_N > cloud_treshold) && (_N_upper > cloud_treshold) && (_N_upper_upper > cloud_treshold))
				{
                                        cloud[k][j]=true;
                                        if (base[k].size() <= top[k].size())
                                        {
                                                double newbase = middle(a,j-1,r,loc);
                                                base[k].push_back(newbase);
                                                N_max[k][base[k].size()] = _N;
                                                if (_N > sct)
                                                {
                                                        sct_base[k][base[k].size()] = newbase;
                                                }
                                                else if (N > few)
                                                {
                                                        few_base[base[k].size()] = newbase;
                                                }
                                        }
                                        else if (_N > N_max[k][base.size()])
                                        {
                                                N_max[k][base[k].size()] = _N;
                                                if (_N > sct && (sct_base[k].size() < base[k].size()))
                                                {
                                                        sct_base[k][base[k].size()] = middle(a,j-1,r,loc);
                                                }
                                                else if (_N > few && (sct_base[k].size() < base[k].size()))
                                                {
                                                        few_base[k][base[k].size()] = middle(a,j-1,r,loc);
                                                }
                                        }
				}
				else if ((_N > cloud_treshold) && (base[k].size() > top[k].size())
				{
					cloud[k][j]=true;
					if ( _N > N_max[k][base[k].size()])
					{
						N_max[k][base[k].size()]=_N;
					}
				}
				else if ((_N < cloud_treshold) && (base[k].size() <= top[k].size())
				{
                                        cloud[k][j]=false;
				}
			}
		}
	}
	// end mark cloud layers

        vector<vector<cloud_layer>> c_l = vector<vector<cloud_layer>>(grd_size,vector<cloud_layer>);

	for ( auto&& tup : zip_range(c_l,N_max,sct_base,few_base,base)
	{
		for(int i=0; i<_base.size(); ++i)
		{
			if ((_N_max[i] >= bkn) && (_sct_base[i]))
			{
				_cloud_layer[i].base = _sct_base[i];
			}
			else if ((_N_max[i] >= sct) && (_few_base[i]))
			{
				_cloud_layer[i].base = _few_base[i];
			}
			else
			{
				_cloud_layer[i].base = _base[i];
			}
			if (_top[i])
			{
				_cloud_layer[i].top = _top[i];
			}
			_cloud_layer[i].amount = _N_max[i];
		}
	}

	// check lowest base is above ground
	for ( auto && tup zip_range(c_l,base))
	{
		if (_base.size() > 0)
		{
			_c_l[1].base, _c_l[1].top = cloud_min(_c_l[1].base,_c_l[1].top)
		}
	}

	// change the corridors below the ceiling2 from the lowest BKN layer to the sct layers
	for (int k=0; k<grd_size; ++k)
	{
		for (int i=0; i<base[k].size(); ++i)
		{
			if(c_l[k][i].base<Ceiling2[k] && c_l[k][i].amount>=bkn)
			{
				if (i==base.size()-1)
				{
					c_l[k][i].base = Ceiling2[k];
				}
				else if (c_l[k][i+1].base > Ceiling2[k])
				{
					c_l[k][i].base = Ceiling2[k];
				}
				else
				{
					if (c_l[k][i].amount>c_l[k][i+1].amount
					{
						c_l[k][i+1].amount = c_l[k][i].amount;
					}
					c_l[k][i].amount = 50.0;
				}
			}
		}
	}

	// cut off odd cloud layers
	for (int k=0; k<grd_size; ++k)
	{
		for (int i=0; i<base[k].size()-1; ++i)
		{
			if (cblayer[k][i] == cblayer[k][i+1])
			{
				if (c_l[k][i].base >= 1500.0 && (clclass(c_l[k][i+1].amount) > clclass(c_l[k][i].amount)))
				{
					if (roundedbase(c_l[k][i+1].base) - roundedbase(c_l[k][i].base < 6.0))
					{
						c_l[k][i].amount = 0.0;
					}
				}
				else if (c_l[k][i].base >= 1500.0)
				{
					if (roundedbase(c_l[k][i+1].base) - roundedbase(c_l[k][i].base) < 6.0)
					{
						c_l[k][i+1].amount = 0.0;
					}
				}
				else if (samehclass(c_l[k][i].base,c_l[k][i+1].base)
				{
					if (clclass(c_l[k][i+1].amount) > clclass(c_l[k][i]))
					{
						c_l[k][i].amount = 0.0;
					}
					else
					{
						c_l[k][i+1].amount = 0.0;
					}
				}
			}
		}
	}

	vector<double> ovcbase(grd_size,MissingDouble());
	vector<double> bknbase(grd_size,MissingDouble());
        vector<double> sctbase(grd_size,MissingDouble());
        vector<double> fewbase(grd_size,MissingDouble());

	vector<double> cbN(grd_size,MissingDouble());

	for (int k=0; k<grd_size; ++k)
	{
        	for (int i=0; i<base[k].size(); ++i)
        	{
			if (IsMissing(ovcbase[k]))
			{
				if (abs(TC->Value(k))>50.0 && IsMissing(cbN[k]) && (i == base[k].size()-1 || abs(c_l[k][i].base-cbbase[k]) < abs(c_l[k][i+1].base-cbbase[k])))
				{
					cbN[k] = c_l[k][i].amount;
					if ( cbbase[k]-c_l[k][i].base > 500.0 && i==0)
					{
						if ( c_l[k][i].amount >= bkn )
						{
							bknbase[k] = c_l[k][i].base;
						}
						else if ( c_l[k][i].amount >= sct )
						{
							sctbase[k] = c_l[k][i].base;
						}
						else
						{
							fewbase[k] = c_l[k][i].base;
						}
					}
					else if (c_l[k][i].base >= 1500.0 && c_l[k][i].base <= 5000.0)
					{
						cbbase[k] = c_l[k][i].base;
					}
				}
				else if (c_l[k][i].amount >= ovc && IsMissing(cbN[k]) || cbN[k]>=ovc[k])
				{
					ovcbase[k]=c_l[k][i].base;
					if (bknbase[k] >= ovcbase[k])
					{
						bknbase[k]=MissingDouble();
					}
					else if (IsMissing(bknbase[k]) && c_l[k][i].amount >= bkn && (IsMissing(cbN[k]) || cbN[k] >= bkn))
					{
						bknbase[k]=c_l[k][i].base;
					}
					else if (IsMissing(bknbase[k]) && IsMissing(sctbase[k]) && c_l[k][i].amount >= sct && (IsMissing(cbN[k]) || cbN[k] >= sct)
					{
						sctbase[k]=c_l[k][i].base;
					}
					else if (IsMissing(bknbase[k]) && IsMissing(sctbase[k]) && IsMissing(fewbase[k]) && IsMissing(cbN[k]))
					{
						fewbase[k]=c_l[k][i].base;
					}
				}
			}
		}
	}

	string deviceType = "CPU";


	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
