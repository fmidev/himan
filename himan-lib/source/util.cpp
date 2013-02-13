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
#include <NFmiStereographicArea.h>

using namespace himan;
using namespace std;

const double kDegToRad = 0.017453292; // PI / 180

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
                    << HPProjectionTypeToString.at(info->Grid()->Projection())
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

pair<point,point> util::CoordinatesFromFirstGridPoint(const point& firstPoint, size_t ni, size_t nj, double di, double dj, HPScanningMode scanningMode)
{
	double dni = static_cast<double> (ni) - 1;
	double dnj = static_cast<double> (nj) - 1;

	point bottomLeft, topRight, topLeft, bottomRight;

	switch (scanningMode)
	{
	case kBottomLeft:
		bottomLeft = firstPoint;
		topRight = point(bottomLeft.X() + dni*di, bottomLeft.Y() + dnj*dj);
		break;

	case kTopLeft: // +x-y
		bottomRight = point(firstPoint.X() + dni*di, firstPoint.Y() - dnj*dj);
		bottomLeft = point(bottomRight.X() - dni*di, firstPoint.Y() - dnj*dj);
		topRight = point(bottomLeft.X() + dni*di, bottomLeft.Y() + dnj*dj);
		break;

	case kTopRight: // -x-y
		topRight = firstPoint;
		bottomLeft = point(topRight.X() - dni*di, topRight.Y() - dnj*dj);
		break;

	case kBottomRight: // -x+y
		topLeft = point(firstPoint.X() - dni*di, firstPoint.Y() + dnj*dj);
		bottomLeft = point(firstPoint.X() - dni*di, topLeft.Y() - dnj*dj);
		topRight = point(bottomLeft.X() + dni*di, bottomLeft.Y() + dnj*dj);
		break;

	default:
		throw runtime_error("util::CoordinatesFromFirstGridPoint(): Calculating first grid point when scanning mode is unknown");
		break;

	}

	return pair<point,point> (bottomLeft, topRight);
}

pair<point,point> util::CoordinatesFromFirstGridPoint(const point& firstPoint, double orientation, size_t ni, size_t nj, double xSizeInMeters, double ySizeInMeters)
{

	double xWidthInMeters = (static_cast<double> (ni)-1.) * xSizeInMeters;
	double yWidthInMeters = (static_cast<double> (nj)-1.) * ySizeInMeters;

	NFmiStereographicArea a(firstPoint.ToNFmiPoint(),
								xWidthInMeters,
								yWidthInMeters,
								orientation,
								NFmiPoint(0,0),
								NFmiPoint(1,1),
								90.);

	point bottomLeft(a.BottomLeftLatLon());
	point topRight(a.TopRightLatLon());

	return pair<point,point> (bottomLeft, topRight);

}

point util::UVToEarthRelative(const point& regPoint, const point& rotPoint, const point& southPole, const point& UV)
{

	point newSouthPole;

	if (southPole.Y() > 0)
	{
		newSouthPole.Y(-southPole.Y());
		newSouthPole.X(0);
	}
	else
	{
		newSouthPole = southPole;
	}

	double sinPoleY = sin(kDegToRad * (newSouthPole.Y()+90)); // zsyc
	double cosPoleY = cos(kDegToRad * (newSouthPole.Y()+90)); // zcyc

	//double zsxreg = sin(kDegToRad * regPoint.X());
	//double zcxreg = cos(kDegToRad * regPoint.X());
	//double zsyreg = sin(kDegToRad * regPoint.Y());
	double cosRegY = cos(kDegToRad * regPoint.Y()); // zcyreg

	double zxmxc = kDegToRad * (regPoint.X() - newSouthPole.X());
	double sinxmxc = sin(zxmxc); // zsxmxc
	double cosxmxc = cos(zxmxc); // zcxmxc

	double sinRotX = sin(kDegToRad * rotPoint.X()); // zsxrot
	double cosRotX = cos(kDegToRad * rotPoint.X()); // zcxrot
	double sinRotY = sin(kDegToRad * rotPoint.Y()); // zsyrot
	double cosRotY = cos(kDegToRad * rotPoint.Y()); // zcyrot

	double PA = cosxmxc * cosRotX + cosPoleY * sinxmxc * sinRotX;
	double PB = cosPoleY * sinxmxc * cosRotX * sinRotY + sinPoleY * sinxmxc * cosRotY - cosxmxc * sinRotX * sinRotY;
	double PC = (-sinPoleY) * sinRotX / cosRegY;
	double PD = (cosPoleY * cosRotY - sinPoleY * cosRotX * sinRotY) / cosRegY;

	double U = PA * UV.X() + PB * UV.Y();
	double V = PC * UV.X() + PD * UV.Y();

	return point(U,V);
}

point util::UVToGridRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV)
{

	point newSouthPole;

	if (southPole.Y() > 0)
	{
		newSouthPole.Y(-southPole.Y());
		newSouthPole.X(0);
	}
	else
	{
		newSouthPole = southPole;
	}

	double sinPoleY = sin(kDegToRad * (newSouthPole.Y()+90)); // zsyc
	double cosPoleY = cos(kDegToRad * (newSouthPole.Y()+90)); // zcyc

	//double sinRegX = sin(kDegToRad * regPoint.X()); // zsxreg
	//double cosRegX = cos(kDegToRad * regPoint.X()); // zcxreg
	double sinRegY = sin(kDegToRad * regPoint.Y()); // zsyreg
	double cosRegY = cos(kDegToRad * regPoint.Y()); // zcyreg

	double zxmxc = kDegToRad * (regPoint.X() - newSouthPole.X());
	double sinxmxc = sin(zxmxc); // zsxmxc
	double cosxmxc = cos(zxmxc); // zcxmxc

	double sinRotX = sin(kDegToRad * rotPoint.X()); // zsxrot
	double cosRotX = cos(kDegToRad * rotPoint.X()); // zcxrot
	//double sinRotY = sin(kDegToRad * rotPoint.Y()); // zsyrot
	double cosRotY = cos(kDegToRad * rotPoint.Y()); // zcyrot

	double PA = cosPoleY * sinxmxc * sinRotX + cosxmxc * cosRotX;
	double PB = cosPoleY * cosxmxc * sinRegY * sinRotX - sinPoleY * cosRegY * sinRotX - sinxmxc * sinRegY * cosRotX;
	double PC = sinPoleY * sinxmxc / cosRotY;
	double PD = (sinPoleY * cosxmxc * sinRegY + cosPoleY * cosRegY) / cosRotY;

	double U = PA * UV.X() + PB * UV.Y();
	double V = PC * UV.X() + PD * UV.Y();

	return point(U,V);
}

point util::UVToGeographical(double longitude, const point& stereoUV)
{

	double U,V;

	if (stereoUV.X() == 0 && stereoUV.Y() == 0)
	{
		return point(0,0);
	}

	double sinLon = sin(longitude * kDegToRad);
	double cosLon = cos(longitude * kDegToRad);

	U = stereoUV.X() * cosLon + stereoUV.Y() * sinLon;
	V = -stereoUV.X() * sinLon + stereoUV.Y() * cosLon;

	return point(U,V);
}
