/**
 * @file util.cpp
 *
 * @brief Different utility functions and classes in a namespace
 *
 * @date Dec 1, 2012
 * @author partio
 */

#include "util.h"
#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <iomanip>

using namespace himan;
using namespace std;

const double kInterpolatedValueEpsilon = 0.00001;

bool himan::util::InterpolateToPoint(shared_ptr<const NFmiGrid> targetGrid, shared_ptr<NFmiGrid> sourceGrid, bool gridsAreEqual, double& value)
{
	if (gridsAreEqual)
	{
		value = sourceGrid->FloatValue();
		return true;
	}

	const NFmiPoint targetLatLonPoint = targetGrid->LatLon();
	const NFmiPoint targetGridPoint = targetGrid->GridPoint();
	const NFmiPoint TGridPoint = sourceGrid->LatLonToGrid(targetLatLonPoint);

	bool noInterpolation = (fabs(targetGridPoint.X() - round(TGridPoint.X())) < kInterpolatedValueEpsilon &&
		 fabs(targetGridPoint.Y() - round(TGridPoint.Y())) < kInterpolatedValueEpsilon);

	if (noInterpolation)
	{
		value = sourceGrid->FloatValue();
		return true;
	}

	return sourceGrid->InterpolateToLatLonPoint(targetLatLonPoint, value);

}
string util::MakeNeonsFileName(shared_ptr<const info> info)
{

    ostringstream neonsFileName;

    string base = "/tmp"; //"/cluster/hiladata/BDAP/REFStorage/";

    neonsFileName 	<< base
                    << "/"
                    << info->Producer().Centre()
                    << "_"
                    << info->Producer().Process()
                    << "/"
                    << info->Time().OriginDateTime()->String("%Y%m%d%H%M")
                    << "/"
                    << info->Param().Name()
                    << "_"
                    << HPLevelTypeToString.at(info->Level().Type())
                    << "_"
                    << info->Level().Value()
                    << "_"
                    << HPProjectionTypeToString.at(info->Projection())
                    << "_"
                    << info->Ni()
                    << "_"
                    << info->Nj()
                    << "_0_"
                    << setw(3)
                    << setfill('0')
                    << info->Time().Step()
                    ;

    return neonsFileName.str();

}

himan::HPFileType util::FileType(const string& theFile)
{

    using namespace std;

    if (!boost::filesystem::exists(theFile))
    {
        cerr << "Input file '" + theFile + "' does not exist" << endl;
        exit(1);
    }

    ifstream f(theFile.c_str(), ios::in | ios::binary);

    char* content;

    short keywordLength = 4;

    content = static_cast<char*> (malloc((keywordLength + 1) * sizeof(char)));

    f.read(content, keywordLength);

    HPFileType ret = kUnknownFile;

    if (strncmp(content, "GRIB", 4) == 0)
    {
        ret = kGRIB;
    }
    else if (strncmp(content, "CDF", 3) == 0)
    {
        ret = kNetCDF;
    }
    else
    {
        // Not GRIB or NetCDF, keep on searching

        keywordLength = 5;

        free(content);

        content = static_cast<char*> (malloc((keywordLength + 1) * sizeof(char)));

        f.read(content, keywordLength);

        if (strncmp(content, "QINFO", 5) == 0)
        {
            ret = kQueryData;
        }

    }

    free (content);

    return ret;
}


// copied from http://stackoverflow.com/questions/236129/splitting-a-string-in-c and modified a bit

vector<string> util::Split(const string& s, const std::string& delims, bool fill)
{

    string item;

    vector<string> orig_elems;

    boost::split(orig_elems, s, boost::is_any_of(delims));

    if (!fill || orig_elems.size() == 0)
    {
        return orig_elems;
    }

    vector<string> filled_elems;
    vector<string> splitted_elems;

    vector<string>::iterator it;

    for (it = orig_elems.begin(); it != orig_elems.end(); )
    {

        boost::split(splitted_elems, *it, boost::is_any_of("-"));

        if (splitted_elems.size() == 2)
        {
            it = orig_elems.erase(it);

            for (int i = boost::lexical_cast<int> (splitted_elems[0]); i <= boost::lexical_cast<int> (splitted_elems[1]); i++)
            {
                filled_elems.push_back(boost::lexical_cast<string> (i));
            }
        }
        else
        {
            ++it;
        }
    }

    vector<string> all_elems;

    all_elems.reserve(orig_elems.size() + filled_elems.size());

    all_elems.insert(all_elems.end(), orig_elems.begin(), orig_elems.end());
    all_elems.insert(all_elems.end(), filled_elems.begin(), filled_elems.end());

    return all_elems;
}

bool util::thread_manager::AdjustLeadingDimension(shared_ptr<info> myTargetInfo)
{

    lock_guard<mutex> lock(itsAdjustDimensionMutex);

    // Leading dimension can be: time or level

    if (itsLeadingDimension == kTimeDimension)
    {
        if (!itsFeederInfo->NextTime())
        {
            return false;
        }

        myTargetInfo->Time(itsFeederInfo->Time());
    }
    else if (itsLeadingDimension == kLevelDimension)
    {
        if (!itsFeederInfo->NextLevel())
        {
            return false;
        }

        myTargetInfo->Level(itsFeederInfo->Level());
    }
    else
    {
        throw runtime_error(ClassName() + ": Invalid dimension type: " + boost::lexical_cast<string> (itsLeadingDimension));
    }

    return true;
}

bool util::thread_manager::AdjustNonLeadingDimension(shared_ptr<info> myTargetInfo)
{
    if (itsLeadingDimension == kTimeDimension)
    {
        return myTargetInfo->NextLevel();
    }
    else if (itsLeadingDimension == kLevelDimension)
    {
        return myTargetInfo->NextTime();
    }
    else
    {
        throw runtime_error(ClassName() + ": unsupported leading dimension: " + boost::lexical_cast<string> (itsLeadingDimension));
    }
}

void util::thread_manager::ResetNonLeadingDimension(shared_ptr<info> myTargetInfo)
{
    if (itsLeadingDimension == kTimeDimension)
    {
        myTargetInfo->ResetLevel();
    }
    else if (itsLeadingDimension == kLevelDimension)
    {
        myTargetInfo->ResetTime();
    }
    else
    {
        throw runtime_error(ClassName() + ": unsupported leading dimension: " + boost::lexical_cast<string> (itsLeadingDimension));
    }
}
