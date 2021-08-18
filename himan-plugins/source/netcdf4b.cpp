/**
 * @file netcdf4.cpp
 *
 */

#include "fminc4.h"
#include "variable.h"
#include "group.h"
#include "dimension.h"
#include "common.h"

#include "grid.h"

#include "producer.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include <fstream>

#include "netcdf4.h"

#include "plugin_factory.h"
#include "radon.h"

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#endif

#ifdef __clang__

#pragma clang diagnostic pop

#endif

using namespace std;
using namespace himan;
using namespace himan::plugin;
using namespace fminc4;

void WriteAreaAndGrid(const shared_ptr<himan::grid>& grid, const producer& prod, nc_group group)
{
        switch (grid->Type())
        {
                case kLatitudeLongitude:
                {
                        auto rg = dynamic_pointer_cast<latitude_longitude_grid>(grid);

                        nc_dim lat;
                        nc_dim lon;

                       	group.AddDim("latitude", rg->Nj());
                       	lat = group.GetDim("latitude");

                        group.AddDim("longitude", rg->Ni());
                        lon = group.GetDim("longitude");

                        himan::point firstGridPoint = rg->FirstPoint();
                        himan::point lastGridPoint = rg->LastPoint();

                        switch (rg->ScanningMode())
                        {
                                case kTopLeft:
                                {
                                        std::vector<double> v(rg->Ni());
					for(size_t i = 0; i < rg->Ni(); ++i)
					{
						v[i] = firstGridPoint.X() + double(i) * rg->Di();
					}

					try
					{
                                        	group.AddVar<double>("longitude",{lon});
					}
					catch (...)
					{
						//nothing
					}

                                        nc_var<double> longvar = group.GetVar<double>("longitude");

					try
					{
                                        	longvar.Write(v);
					}
					catch (...)
					{
						//nothing
					}

                                        std::vector<double> w(rg->Nj());
                                        for(size_t j = 0; j < rg->Nj(); ++j)
                                        {
                                                w[j] = firstGridPoint.Y() - double(j) * rg->Dj();
                                        }

					try
					{
                                        	group.AddVar<double>("latitude",{lat});
					}
					catch (...)
					{
						//nothing
					}
                                        nc_var<double> latvar = group.GetVar<double>("latitude");
					try
					{
                                        	latvar.Write(w);
					}
					catch (...)
					{
						//nothing
					}

                                        break;
                                }
                                default:
                                        himan::Abort();
                        }

                        break;
                }
                default:
                        himan::Abort();
        }
}

void WriteTime(const forecast_time& ftime, const producer& prod, nc_group group, size_t tidx)
{
	nc_dim time;
	nc_var<long> timevar;

	try
	{
		group.AddDim("time", NC_UNLIMITED);
		time = group.GetDim("time");

		group.AddVar<long>("time",{time});
		timevar = group.GetVar<long>("time");

		timevar.AddAtt<string>("units",{"Hours since " + ftime.OriginDateTime().String("%Y-%m-%d %H:%M:%S")});
	}
	catch (...)
	{
		std::cout << "time error\n";
	}

	try
	{
		timevar = group.GetVar<long>("time");
	}
	catch (...)
	{
		cout << "failed get timevar\n";
	}

	try
	{
		timevar.Write(ftime.Step().Hours(),{tidx});
	}
	catch (...)
	{
		cout << "timewrite fail\n";
	}
}

void WriteLevel(const level& lev, const producer& prod, nc_group group)
{
	nc_dim level;

	try
	{
		level = group.GetDim("level");
	}
	catch(...)
	{
		group.AddDim("level", NC_UNLIMITED);
		level = group.GetDim("level");
	}

	switch(lev.Type())
	{
		case kGround :
		{
			break;
		}
		case kPressure :
		{
			break;
		}
		case kHeight :
		{
			try
			{
		        	group.AddVar<double>("level",{level});
			}
			catch (...)
			{
				//nothing
			}
			nc_var<double> levelvar = group.GetVar<double>("level");
			if (lev.Index() != 999999)
			{
				try
				{
					levelvar.Write(lev.Value(),{lev.Index()});
				}
				catch (...)
				{
					std::cout << "level error\n";
				}
			}
			else
			{
				try
				{
					levelvar.Write(lev.Value(),{0});
				}
				catch (...)
				{
					std::cout << "level error\n";
				}
			}

			try
			{
				levelvar.AddAtt<std::string>("units",{"meters"});
				levelvar.AddAtt<std::string>("positive",{"up"});
			}
			catch (...)
			{
				std::cout << "level atts  error\n";
			}

			break;
		}
		case kHybrid :
		{
			break;
		}
		case kMaximumThetaE :
		{
			break;
		}
		default :
			himan::Abort();
	}
}

template <typename T>
void WriteParam(const param& par, const producer& prod, nc_group group, const std::vector<T>& vals, size_t tidx)
{
	nc_var<T> parameter;

	std::vector<nc_dim> dims;
	dims.reserve(4);
	dims.push_back(group.GetDim("time"));
	dims.push_back(group.GetDim("level"));
	dims.push_back(group.GetDim("latitude"));
	dims.push_back(group.GetDim("longitude"));

	try
	{
		group.AddVar<T>(par.Name(), dims);
	}
	catch(...)
	{
		std::cout << "creation failed\n";
		// nothing
	}

	try
        {
                parameter = group.GetVar<T>(par.Name());
        }
        catch(...)
        {
		std::cout << "fetch failed\n";
        }

	try
	{
		parameter.Write(vals, {tidx,0,0,0}, {1,1,1801,3600});
	}
	catch (...)
	{
		cout << "write fail\n";
	}
}
template void WriteParam<double>(const param&, const producer&, nc_group, const std::vector<double>&, size_t);
template void WriteParam<float>(const param&, const producer&, nc_group, const std::vector<float>&, size_t);

netcdf4::netcdf4()
{
	itsLogger = logger("netcdf4");
}

/*netcdf4::~netcdf4()
{
      fminc4::Finalize(); // perhaps this finalize call needs to be moved elsewhere if multiple instances of netcdf plugin coexist.
}*/

bool netcdf4::ToFile(info<double>& anInfo, string& outputFile, bool appendToFile)
{
        return ToFile<double>(anInfo, outputFile, appendToFile);
}

template <typename T>
bool netcdf4::ToFile(info<T>& anInfo, string& outputFile, bool appendToFile)
{
        // Write only that data which is currently set at descriptors
        nc_group theFile;

        if(appendToFile)
        {
               	theFile = Open(outputFile);
        }
        else
        {
               	theFile = Create(outputFile);
        }

	WriteAreaAndGrid(anInfo.Grid(), anInfo.Producer(), theFile);

	WriteTime(anInfo.Time(), anInfo.Producer(), theFile, anInfo.template Index<forecast_time>());

	WriteLevel(anInfo.Level(), anInfo.Producer(), theFile);

	WriteParam<T>(anInfo.Param(), anInfo.Producer(), theFile, anInfo.Data().Values(), anInfo.template Index<forecast_time>());

        string verb = (appendToFile ? "Appended to " : "Wrote ");
        itsLogger.Info(verb + "file '" + outputFile);

        return true;
}
template bool netcdf4::ToFile<double>(info<double>&, string&, bool);
template bool netcdf4::ToFile<float>(info<float>&, string&, bool);
