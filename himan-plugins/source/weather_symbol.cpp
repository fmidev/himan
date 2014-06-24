/**
 * @file weather_symbol.cpp
 *
 *  @date: Jan 23, 2013
 *  @author aaltom
 */

#include "weather_symbol.h"
#include <iostream>
#include <map>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"
#include "NFmiGrid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const int rain[4][3] = {
        {1,2,3},
        {0,21,31},
        {0,22,32},
        {0,23,33}
};
const int thunder_and_rain[4][3] = {
        {1,2,3},
        {0,21,31},
        {0,61,61},
        {0,62,62}
};
const int snow[4][3] = {
        {1,2,3},
        {0,41,51},
        {0,42,52},
        {0,43,53}
}; 


weather_symbol::weather_symbol()
{
	itsClearTextFormula = "weather_symbol = ";
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("weather_symbol"));

    // hilake: etsi_pilvi.F
	cloudMap.insert(std::make_pair(0,1));
	cloudMap.insert(std::make_pair(704,0));
	cloudMap.insert(std::make_pair(1301,1));
	cloudMap.insert(std::make_pair(1302,2));
	cloudMap.insert(std::make_pair(1303,2));
	cloudMap.insert(std::make_pair(1305,2));
	cloudMap.insert(std::make_pair(1309,2));
	cloudMap.insert(std::make_pair(1403,2));
	cloudMap.insert(std::make_pair(1501,1));
	cloudMap.insert(std::make_pair(2307,2));
	cloudMap.insert(std::make_pair(2403,2));
	cloudMap.insert(std::make_pair(2305,2));
	cloudMap.insert(std::make_pair(2501,2));
	cloudMap.insert(std::make_pair(2302,2));
	cloudMap.insert(std::make_pair(2303,2));
    cloudMap.insert(std::make_pair(2304,2));
    cloudMap.insert(std::make_pair(3309,3));
    cloudMap.insert(std::make_pair(3307,3));
    cloudMap.insert(std::make_pair(3604,3)); 
    cloudMap.insert(std::make_pair(3405,3));
    cloudMap.insert(std::make_pair(3306,3));
    cloudMap.insert(std::make_pair(3502,2)); 

}

void weather_symbol::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to weather_symbol
	 * - name HESSAA-N
	 * - univ_id 80
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("HESSAA-N", 338);

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void weather_symbol::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param CParam("CLDSYM-N");
	param RTParam("HSADE1-N");  

	level HLevel(himan::kHeight, 0, "HEIGHT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("weather_symbolThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		//myTargetInfo->Data()->Resize(conf->Ni(), conf->Nj());

		shared_ptr<info> CInfo;
		shared_ptr<info> RTInfo;

		try
		{
			// Source info for clouds
			CInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 CParam);				
			// Source info for hsade
			RTInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 RTParam);

				
		}
		catch (HPExceptionType& e)
		{

			switch (e)
			{
			case kFileDataNotFound:
				itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
				myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

				if (itsConfiguration->StatisticsEnabled())
				{
					itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
					itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
				}

				continue;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
			}
		}
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> CGrid(CInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RTGrid(RTInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *CInfo->Grid() &&
							*myTargetInfo->Grid() == *RTInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		string deviceType = "CPU";

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double cloudSymbol = kFloatMissing;
			double rainType = kFloatMissing;
			
			InterpolateToPoint(targetGrid, CGrid, equalGrids, cloudSymbol);
			InterpolateToPoint(targetGrid, RTGrid, equalGrids, rainType);


			if (cloudSymbol == kFloatMissing || rainType == kFloatMissing )
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);  // No missing values
				continue;
			}

			double weather_symbol;
			double ctype, rtype, rform, wtype;
		
			weather_symbol = 0;
			ctype = cloud_type(cloudSymbol);
			rtype = rain_type(rainType);
			rform = rain_form(rainType);
			wtype = weather_type(rainType);

			if (rform == 1)
			{   
				if(rtype == 1)
				{
					if (ctype > 0 && wtype > 0)
					{
						// jatkuva vesisade
						weather_symbol = rain[static_cast<int>(wtype)-1][static_cast<int>(ctype)-1];					
					}
				}
				else if (rtype == 2)
				{
					if (ctype > 0 && wtype > 0)
					{
						// kuurottainen vesisade
						weather_symbol = rain[static_cast<int>(wtype)-1][static_cast<int>(ctype)-1];
					}
				}
				else if (rtype == 3)
				{
					if (ctype > 0 && wtype > 0)
					{
						// ukkonen ja vesisade
						weather_symbol = thunder_and_rain[static_cast<int>(wtype)-1][static_cast<int>(ctype)-1];
						
					}
				}
			}
			else if (rform == 2) 
			{
				if(rtype == 1)
				{
					if (ctype > 0 && wtype > 0)
					{
						// jatkuva lumisade
						weather_symbol = snow[static_cast<int>(wtype)-1][static_cast<int>(ctype)-1];
					}
				}
				else if (rtype == 2)
				{
					if (ctype > 0 && wtype > 0)
					{
						// kuurottainen lumisade
						weather_symbol = snow[static_cast<int>(wtype)-1][static_cast<int>(ctype)-1];
					}
				}
				else if (rtype == 3)
				{
					if (ctype > 0 && wtype > 0)
					{
						// ukkonen ja lumisade
						weather_symbol = snow[static_cast<int>(wtype)-1][static_cast<int>(ctype)-1];
					}
				}
			}

			if (!myTargetInfo->Value(weather_symbol))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}  

		/*
		 * Newbase normalizes scanning mode to bottom left -- if that's not what
		 * the target scanning mode is, we have to swap the data back.
		 */

		SwapTo(myTargetInfo, kBottomLeft);

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);
		}
		
		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}

double weather_symbol::rain_form(double rr) {
  
    double rain_form;
	if (rr <=67 || rr == 80 || rr == 81 || rr == 82 || rr == 95 || rr == 97)
	{
        rain_form = 1;
	}
	else 
	{
		rain_form = 2;
	}
	return rain_form;
}

double weather_symbol::weather_type(double rr) {

    double weather_type;
    // or maybe std::map?
    if (rr >= 50 && rr <=54)
	{
		weather_type = 2;
	}
	else if(rr >= 56 && rr <=61) 
	{
		weather_type = 2;
	}
	else if(rr >= 76 && rr <= 80)
	{
		weather_type = 2;
	}
	else if(rr == 66 || rr == 70 || rr == 71 || rr == 85 || rr == 2070)
	{
		weather_type = 2;
	}
	else if(rr == 55 || rr == 62 || rr == 63 || rr == 67 || rr == 68 || rr == 72 || rr == 73 || rr == 81 || rr == 95) 
	{
		weather_type = 3;
	}
	else if(rr == 64 || rr == 65 || rr == 69 || rr == 74 || rr == 75 || rr == 82 || rr == 86 || rr == 97)
	{
		weather_type = 4;
	}
	else if (rr == 0)
	{
		weather_type = 1;
	}
	else
	{
       throw runtime_error("Invalid value for rr: Fix me!");
	}


    return weather_type;
}

double weather_symbol::rain_type(double rr) {

	double rain_type;
	if (rr <=79 || rr == 2070)
	{
		rain_type = 1;
	}
	else if(rr >= 80 && rr <=86) 
	{
		rain_type = 2;
	}
	else if(rr == 95 || rr == 97)
	{
		rain_type = 3;
	}
	else
	{
		throw runtime_error("Invalid value for rr: Fix me!");
	}
	
    return rain_type;
}

double weather_symbol::cloud_type(double cloud) {
    // etsi_pilvityyppi
	typedef std::map<double, double>::const_iterator MapIter;
	for (MapIter iter = cloudMap.begin(); iter != cloudMap.end(); iter++)
    {
    	if (iter->first == cloud)
    	{
    		return iter->second; 
    	}
  
    }
    return 0;
}
