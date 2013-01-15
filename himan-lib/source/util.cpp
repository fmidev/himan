/**
 * @file util.cpp
 *
 * @brief Different utility functions and classes in a namespace
 *
 * @date Dec 1, 2012
 * @author partio
 */

#include "util.h"
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <iomanip>

using namespace himan;
using namespace std;

string util::MakeNeonsFileName(shared_ptr<const info> info)
{

    ostringstream neonsFileName;

    string base = ".";

    char* path;

    path = std::getenv("NEONS_REF_BASE");

    if (path != NULL)
    {
    	base = string(path);
    }

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
