/**
 * @file util.cpp
 *
 * @brief Different utility functions and classes in a namespace
 */

#include "util.h"
#include "cuda_helper.h"
#include "forecast_time.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "param.h"
#include "plugin_factory.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/math/constants/constants.hpp>
#include <iomanip>
#include <sstream>
#include <wordexp.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "cache.h"
#include "radon.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

template <typename T>
string util::MakeFileName(HPFileWriteOption fileWriteOption, const info<T>& info, const configuration& conf)
{
	ostringstream fileName;
	ostringstream base;

	const auto ftype = info.template Value<forecast_type>();
	const auto ftime = info.template Value<forecast_time>();
	const auto lvl = info.template Value<level>();
	const auto par = info.template Value<param>();

	base.str(".");

	// For neons get base directory

	if (fileWriteOption == kDatabase)
	{
		char* path;

		path = std::getenv("MASALA_PROCESSED_DATA_BASE");

		if (path != NULL)
		{
			base.str("");
			base << path;
		}
		else
		{
			cout << "Warning::util MASALA_PROCESSED_DATA_BASE not set" << endl;
		}

		base << "/" << info.Producer().Id() << "/" << ftime.OriginDateTime().String("%Y%m%d%H%M") << "/"
		     << conf.TargetGeomName() << "/" << ftime.Step();
	}

	// Create a unique file name when creating multiple files from one info

	if (fileWriteOption == kDatabase || fileWriteOption == kMultipleFiles)
	{
		fileName << base.str() << "/" << par.Name() << "_" << HPLevelTypeToString.at(lvl.Type()) << "_" << lvl.Value();

		if (!IsKHPMissingValue(lvl.Value2()))
		{
			fileName << "-" << lvl.Value2();
		}

		fileName << "_" << HPGridTypeToString.at(info.Grid()->Type());

		if (info.Grid()->Class() == kRegularGrid)
		{
			fileName << "_" << dynamic_pointer_cast<regular_grid>(info.Grid())->Ni() << "_"
			         << dynamic_pointer_cast<regular_grid>(info.Grid())->Nj();
		}

		fileName << "_0_" << setw(3) << setfill('0') << ftime.Step();
		if (static_cast<int>(ftype.Type()) > 2)
		{
			fileName << "_" << static_cast<int>(ftype.Type()) << "_" << ftype.Value();
		}
	}
	else
	{
		// TODO!

		fileName << base.str() << "/"
		         << "TODO.file";
	}

	return fileName.str();
}

template string util::MakeFileName<double>(HPFileWriteOption, const info<double>&, const configuration&);
template string util::MakeFileName<float>(HPFileWriteOption, const info<float>&, const configuration&);

himan::HPFileType util::FileType(const string& theFile)
{
	using namespace std;

	// First check by extension since its cheap

	boost::filesystem::path p(theFile);

	if (p.extension().string() == ".gz" || p.extension().string() == ".bz2")
	{
		p = p.stem();
	}

	string ext = p.extension().string();

	if (ext == ".csv")
	{
		return kCSV;
	}
	else if (ext == ".fqd" || ext == ".sqd")
	{
		return kQueryData;
	}
	else if (ext == ".grib")
	{
		return kGRIB;
	}
	else if (ext == ".idx")
	{
		return kGRIBIndex;
	}
	else if (ext == ".grib2")
	{
		return kGRIB2;
	}
	else if (ext == ".nc")
	{
		return kNetCDF;
	}

	// Try the check the file header; CSV is not possible anymore

	ifstream f(theFile.c_str(), ios::in | ios::binary);

	long keywordLength = 4;

	char content[keywordLength];

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

		char qcontent[keywordLength];

		f.read(qcontent, keywordLength);

		if (strncmp(qcontent, "QINFO", 5) == 0)
		{
			ret = kQueryData;
		}
	}

	return ret;
}

// copied from http://stackoverflow.com/questions/236129/splitting-a-string-in-c and modified a bit

vector<string> util::Split(const string& s, const string& delims, bool fill)
{
	vector<string> orig_elems;

	boost::split(orig_elems, s, boost::is_any_of(delims));

	if (!fill || orig_elems.size() == 0)
	{
		return orig_elems;
	}

	vector<string> filled_elems;
	vector<string> splitted_elems;

	vector<string>::iterator it;

	for (it = orig_elems.begin(); it != orig_elems.end();)
	{
		boost::split(splitted_elems, *it, boost::is_any_of("-"));

		if (splitted_elems.size() == 2)
		{
			it = orig_elems.erase(it);

			int first = stoi(splitted_elems[0]);
			int last = stoi(splitted_elems.back());

			if (first <= last)
			{
				// levels are 1-65
				for (int i = first; i <= last; ++i)
					filled_elems.push_back(to_string(i));
			}
			else
			{
				// levels are 65-1
				for (int i = first; i >= last; --i)
					filled_elems.push_back(to_string(i));
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

string util::Join(const vector<string>& elements, const string& delim)
{
	ostringstream s;

	for (size_t i = 0; i < elements.size(); i++)
	{
		if (i != 0)
		{
			s << delim;
		}
		s << elements[i];
	}

	return s.str();
}

tuple<double, double, double, double> util::EarthRelativeUVCoefficients(const himan::point& regPoint,
                                                                        const himan::point& rotPoint,
                                                                        const himan::point& southPole)
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

	double southPoleY = constants::kDeg * (newSouthPole.Y() + 90);
	double sinPoleY, cosPoleY;

	sincos(southPoleY, &sinPoleY, &cosPoleY);

	double cosRegY = cos(constants::kDeg * regPoint.Y());  // zcyreg

	double zxmxc = constants::kDeg * (regPoint.X() - newSouthPole.X());

	double sinxmxc, cosxmxc;

	sincos(zxmxc, &sinxmxc, &cosxmxc);

	double rotXRad = constants::kDeg * rotPoint.X();
	double rotYRad = constants::kDeg * rotPoint.Y();

	double sinRotX, cosRotX;
	sincos(rotXRad, &sinRotX, &cosRotX);

	double sinRotY, cosRotY;
	sincos(rotYRad, &sinRotY, &cosRotY);

	double PA = cosxmxc * cosRotX + cosPoleY * sinxmxc * sinRotX;
	double PB = cosPoleY * sinxmxc * cosRotX * sinRotY + sinPoleY * sinxmxc * cosRotY - cosxmxc * sinRotX * sinRotY;
	double PC = (-sinPoleY) * sinRotX / cosRegY;
	double PD = (cosPoleY * cosRotY - sinPoleY * cosRotX * sinRotY) / cosRegY;

	return make_tuple(PA, PB, PC, PD);
}

tuple<double, double, double, double> util::GridRelativeUVCoefficients(const himan::point& regPoint,
                                                                       const himan::point& rotPoint,
                                                                       const himan::point& southPole)
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

	double sinPoleY = sin(constants::kDeg * (newSouthPole.Y() + 90));  // zsyc
	double cosPoleY = cos(constants::kDeg * (newSouthPole.Y() + 90));  // zcyc

	// double sinRegX = sin(constants::kDeg * regPoint.X()); // zsxreg
	// double cosRegX = cos(constants::kDeg * regPoint.X()); // zcxreg
	double sinRegY = sin(constants::kDeg * regPoint.Y());  // zsyreg
	double cosRegY = cos(constants::kDeg * regPoint.Y());  // zcyreg

	double zxmxc = constants::kDeg * (regPoint.X() - newSouthPole.X());
	double sinxmxc = sin(zxmxc);  // zsxmxc
	double cosxmxc = cos(zxmxc);  // zcxmxc

	double sinRotX = sin(constants::kDeg * rotPoint.X());  // zsxrot
	double cosRotX = cos(constants::kDeg * rotPoint.X());  // zcxrot
	// double sinRotY = sin(constants::kDeg * rotPoint.Y()); // zsyrot
	double cosRotY = cos(constants::kDeg * rotPoint.Y());  // zcyrot

	double PA = cosPoleY * sinxmxc * sinRotX + cosxmxc * cosRotX;
	double PB = cosPoleY * cosxmxc * sinRegY * sinRotX - sinPoleY * cosRegY * sinRotX - sinxmxc * sinRegY * cosRotX;
	double PC = sinPoleY * sinxmxc / cosRotY;
	double PD = (sinPoleY * cosxmxc * sinRegY + cosPoleY * cosRegY) / cosRotY;

	return make_tuple(PA, PB, PC, PD);
}

point util::UVToGeographical(double longitude, const point& stereoUV)
{
	double U, V;

	if (stereoUV.X() == 0 && stereoUV.Y() == 0)
	{
		return point(0, 0);
	}

	double sinLon, cosLon;

	sincos(longitude * constants::kDeg, &sinLon, &cosLon);

	U = stereoUV.X() * cosLon + stereoUV.Y() * sinLon;
	V = -stereoUV.X() * sinLon + stereoUV.Y() * cosLon;

	return point(U, V);
}

double util::ToPower(double value, double power)
{
	double divisor = 1.0;

	while (value < 0)
	{
		divisor /= power;
		value++;
	}

	while (value > 0)
	{
		divisor *= power;
		value--;
	}

	return divisor;
}

template <typename T>
pair<matrix<T>, matrix<T>> util::CentralDifference(matrix<T>& A, vector<T>& dx, vector<T>& dy)
{
	matrix<T> dA_dx(A.SizeX(), A.SizeY(), 1, A.MissingValue());
	matrix<T> dA_dy(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());

	ASSERT(dy.size() == A.SizeX() && dx.size() == A.SizeY());

	// calculate for inner field
	for (int j = 1; j < ASizeY - 1; ++j)  // rows
	{
		for (int i = 1; i < ASizeX - 1; ++i)  // columns
		{
			dA_dx.Set(i, j, 0,
			          (A.At(i + 1, j, 0) - A.At(i - 1, j, 0)) / (2 * dx[j]));  // central difference in x-direction
			dA_dy.Set(i, j, 0,
			          (A.At(i, j + 1, 0) - A.At(i, j - 1, 0)) / (2 * dy[i]));  // central difference in y-direction
		}
	}

	// treat boundaries separately
	for (int i = 1; i < ASizeX - 1; ++i)  // rows
	{
		// calculate for upper boundary
		dA_dx.Set(i, 0, 0, (A.At(i + 1, 0, 0) - A.At(i - 1, 0, 0)) / (2 * dx[0]));  // central difference in x-direction
		dA_dy.Set(i, 0, 0, (A.At(i, 1, 0) - A.At(i, 0, 0)) / dy[i]);  // foreward difference in y-direction

		// calculate for lower boundary
		dA_dx.Set(i, ASizeY - 1, 0,
		          (A.At(i + 1, ASizeY - 1, 0) - A.At(i - 1, ASizeY - 1, 0)) /
		              (2 * dx[ASizeY - 1]));  // central difference in x-direction
		dA_dy.Set(i, ASizeY - 1, 0,
		          (A.At(i, ASizeY - 1, 0) - A.At(i, ASizeY - 2, 0)) / dy[i]);  // backward difference in y-direction
	}

	for (int j = 1; j < ASizeY - 1; ++j)  // columns
	{
		// calculate for left boundary
		dA_dx.Set(0, j, 0, (A.At(1, j, 0) - A.At(0, j, 0)) / dx[j]);  // foreward difference in x-direction
		dA_dy.Set(0, j, 0, (A.At(0, j + 1, 0) - A.At(0, j - 1, 0)) / (2 * dy[0]));  // central difference in y-direction

		// calculate for right boundary
		dA_dx.Set(ASizeX - 1, j, 0,
		          (A.At(ASizeX - 1, j, 0) - A.At(ASizeX - 2, j, 0)) / dx[j]);  // backward difference in x-direction
		dA_dy.Set(ASizeX - 1, j, 0,
		          (A.At(ASizeX - 1, j + 1, 0) - A.At(ASizeX - 1, j - 1, 0)) /
		              (2 * dy[ASizeX - 1]));  // central difference in y-direction
	}

	// corner values last
	// top left
	dA_dx.Set(0, 0, 0, (A.At(1, 0, 0) - A.At(0, 0, 0)) / dx[0]);  // foreward difference in x-direction
	dA_dy.Set(0, 0, 0, (A.At(0, 1, 0) - A.At(0, 0, 0)) / dy[0]);  // foreward difference in y-direction

	// top right
	dA_dx.Set(ASizeX - 1, 0, 0,
	          (A.At(ASizeX - 1, 0, 0) - A.At(ASizeX - 2, 0, 0)) / dx[0]);  // foreward difference in x-direction
	dA_dy.Set(
	    ASizeX - 1, 0, 0,
	    (A.At(ASizeX - 1, 1, 0) - A.At(ASizeX - 1, 0, 0)) / dy[ASizeX - 1]);  // backward difference in y-direction

	// bottom left
	dA_dx.Set(
	    0, ASizeY - 1, 0,
	    (A.At(1, ASizeY - 1, 0) - A.At(0, ASizeY - 1, 0)) / dx[ASizeY - 1]);  // foreward difference in x-direction
	dA_dy.Set(0, ASizeY - 1, 0,
	          (A.At(0, ASizeY - 1, 0) - A.At(0, ASizeY - 2, 0)) / dy[0]);  // backward difference in y-direction

	// bottom right
	dA_dx.Set(ASizeX - 1, ASizeY - 1, 0,
	          (A.At(ASizeX - 1, ASizeY - 1, 0) - A.At(ASizeX - 2, ASizeY - 1, 0)) /
	              dx[ASizeY - 1]);  // backward difference in x-direction
	dA_dy.Set(ASizeX - 1, ASizeY - 1, 0,
	          (A.At(ASizeX - 1, ASizeY - 1, 0) - A.At(ASizeX - 1, ASizeY - 2, 0)) /
	              dy[ASizeX - 1]);  // backward difference in y-direction

	pair<matrix<T>, matrix<T>> ret(dA_dx, dA_dy);
	return ret;
}

template pair<matrix<double>, matrix<double>> util::CentralDifference<double>(matrix<double>& A, vector<double>& dx,
                                                                              vector<double>& dy);
template pair<matrix<float>, matrix<float>> util::CentralDifference<float>(matrix<float>& A, vector<float>& dx,
                                                                           vector<float>& dy);

double util::LatitudeLength(double phi)
{
	double cos_phi = cos(phi * constants::kDeg);
	return 2 * boost::math::constants::pi<double>() * constants::kR * abs(cos_phi);
}

double util::round(double val, unsigned short numdigits)
{
	int div = static_cast<int>(pow(10, static_cast<double>(numdigits)));
	return std::round(val * div) / div;
}

string util::MakeSQLInterval(const himan::forecast_time& theTime)
{
	int step = theTime.Step();
	string ret;

	char i[11];

	if (theTime.StepResolution() == himan::kHourResolution)
	{
		snprintf(i, 11, "%02d:00:00", step);

		ret = i;
	}
	else
	{
		int hours, minutes;

		hours = static_cast<int>(floor(static_cast<double>(step) / 60));
		minutes = step - hours * 60;

		snprintf(i, 11, "%02d:%02d:00", hours, minutes);

		ret = i;
	}

	return ret;
}

string util::Expand(const string& in)
{
	if (in.find("$") == string::npos)
		return in;

	wordexp_t p;

	wordexp(in.c_str(), &p, 0);
	char** out = p.we_wordv;

	string ret(out[0]);

	wordfree(&p);

	return ret;
}

void util::DumpVector(const vector<double>& vec, const string& name)
{
	return DumpVector<double>(vec, name);
}

template <typename T>
void util::DumpVector(const vector<T>& vec, const string& name)
{
	T min = numeric_limits<T>::max(), max = numeric_limits<T>::lowest(), sum = 0;
	size_t count = 0, missing = 0, nan = 0;

	for (const T& val : vec)
	{
		if (IsMissing(val))
		{
			missing++;
			continue;
		}

		if (std::isnan(val))
		{
			nan++;
			continue;
		}

		min = (val < min) ? val : min;
		max = (val > max) ? val : max;
		count++;
		sum += val;
	}

	T mean = numeric_limits<T>::quiet_NaN();

	if (count > 0)
	{
		mean = sum / static_cast<T>(count);
	}

	if (!name.empty())
	{
		cout << name << "\t";
	}

	cout << "min " << min << " max " << max << " mean " << mean << " count " << count << " nan " << nan << " missing "
	     << missing << endl;

	if (min != max && count > 0)
	{
		long binn = (count < 10) ? count : 10;

		T binw = (max - min) / static_cast<T>(binn);

		T binmin = min;
		T binmax = binmin + binw;

		cout << "distribution (bins=" << binn << "):" << endl;

		for (int i = 1; i <= binn; i++)
		{
			if (i == binn)
				binmax += 0.001f;

			count = 0;

			for (const T& val : vec)
			{
				if (IsMissing(val))
					continue;

				if (val >= binmin && val < binmax)
				{
					count++;
				}
			}

			if (i == binn)
				binmax -= 0.001f;

			cout << binmin << ":" << binmax << " " << count << std::endl;

			binmin += binw;
			binmax += binw;
		}
	}
}

template void util::DumpVector<double>(const vector<double>&, const string&);
template void util::DumpVector<float>(const vector<float>&, const string&);

string util::GetEnv(const string& key)
{
	const char* var = getenv(key.c_str());
	if (!var)
	{
		throw runtime_error("Environment variable '" + string(key) + "' not found");
	}
	return string(var);
}

template <typename T>
shared_ptr<info<T>> util::CSVToInfo(const vector<string>& csv)
{
	vector<forecast_time> times;
	vector<param> params;
	vector<level> levels;
	vector<station> stats;
	vector<forecast_type> ftypes;

	producer prod;

	for (auto line : csv)
	{
		auto elems = util::Split(line, ",", false);

		if (elems.size() != 14)
		{
			std::cerr << "Ignoring line '" << line << "'" << std::endl;
			continue;
		}

		// 0 producer_id
		// 1 origin time
		// 2 station_id
		// 3 station_name
		// 4 longitude
		// 5 latitude
		// 6 param_name
		// 7 level_name
		// 8 level_value
		// 9 level_value2
		// 10 forecast period
		// 11 forecast_type_id
		// 12 forecast_type_value
		// 13 value

		if (elems[0][0] == '#')
			continue;

		// producer, only single producer per file is supported for now
		prod.Id(stoi(elems[0]));

		// forecast_time
		raw_time originTime(elems[1]), validTime(elems[1]);

		// split HHH:MM:SS and extract hours and minutes
		auto timeparts = Split(elems[10], ":", false);

		validTime.Adjust(kHourResolution, stoi(timeparts[0]));
		validTime.Adjust(kMinuteResolution, stoi(timeparts[1]));

		forecast_time f(originTime, validTime);

		// level
		level l;

		try
		{
			l = level(static_cast<HPLevelType>(HPStringToLevelType.at(boost::algorithm::to_lower_copy(elems[7]))),
			          stod(elems[8]));
		}
		catch (std::out_of_range& e)
		{
			std::cerr << "Level type " << elems[7] << " is not recognized" << std::endl;
			himan::Abort();
		}

		if (!elems[9].empty())
		{
			l.Value2(stod(elems[9]));
		}

		// param
		const param p(boost::algorithm::to_upper_copy(elems[6]));

		// forecast_type
		const forecast_type ftype(static_cast<HPForecastType>(stoi(elems[11])), stod(elems[12]));

		// station
		const int stationId = (elems[2].empty()) ? kHPMissingInt : stoi(elems[2]);
		const double longitude = (elems[4].empty()) ? kHPMissingValue : stod(elems[4]);
		const double latitude = (elems[5].empty()) ? kHPMissingValue : stod(elems[5]);

		const station s(stationId, elems[3], longitude, latitude);

		/* Prevent duplicates */

		if (find(times.begin(), times.end(), f) == times.end())
		{
			times.push_back(f);
		}

		if (find(levels.begin(), levels.end(), l) == levels.end())
		{
			levels.push_back(l);
		}

		if (find(params.begin(), params.end(), p) == params.end())
		{
			params.push_back(p);
		}

		if (find(ftypes.begin(), ftypes.end(), ftype) == ftypes.end())
		{
			ftypes.push_back(ftype);
		}

		if (find(stats.begin(), stats.end(), s) == stats.end())
		{
			stats.push_back(s);
		}
	}

	if (times.size() == 0 || params.size() == 0 || levels.size() == 0 || ftypes.size() == 0)
	{
		return nullptr;
	}

	auto ret = make_shared<info<T>>();

	ret->Producer(prod);
	ret->template Set<forecast_time>(times);
	ret->template Set<param>(params);
	ret->template Set<level>(levels);
	ret->template Set<forecast_type>(ftypes);

	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(new point_list(stats));
	ret->Create(b, true);

	for (auto line : csv)
	{
		auto elems = util::Split(line, ",", false);

		if (elems.size() != 14)
		{
			std::cerr << "Ignoring line '" << line << "'" << std::endl;
			continue;
		}

		// 0 producer_id
		// 1 origin time
		// 2 station_id
		// 3 station_name
		// 4 longitude
		// 5 latitude
		// 6 param_name
		// 7 level_name
		// 8 level_value
		// 9 level_value2
		// 10 forecast period
		// 11 forecast_type_id
		// 12 forecast_type_value
		// 13 value

		if (elems[0][0] == '#')
			continue;

		// forecast_time
		raw_time originTime(elems[1]), validTime(elems[1]);

		auto timeparts = Split(elems[10], ":", false);

		validTime.Adjust(kHourResolution, stoi(timeparts[0]));
		validTime.Adjust(kMinuteResolution, stoi(timeparts[1]));

		const forecast_time f(originTime, validTime);

		// level
		level l(static_cast<HPLevelType>(HPStringToLevelType.at(boost::algorithm::to_lower_copy(elems[7]))),
		        stod(elems[8]));

		if (!elems[9].empty())
		{
			l.Value2(stod(elems[9]));
		}

		// param
		const param p(elems[6]);

		// forecast_type
		const forecast_type ftype(static_cast<HPForecastType>(stoi(elems[11])), stod(elems[12]));

		// station
		const int stationId = (elems[2].empty()) ? kHPMissingInt : stoi(elems[2]);
		const double longitude = (elems[4].empty()) ? kHPMissingValue : stod(elems[4]);
		const double latitude = (elems[5].empty()) ? kHPMissingValue : stod(elems[5]);

		const station s(stationId, elems[3], longitude, latitude);

		if (!ret->template Find<param>(p))
			continue;
		if (!ret->template Find<forecast_time>(f))
			continue;
		if (!ret->template Find<level>(l))
			continue;
		if (!ret->template Find<forecast_type>(ftype))
			continue;

		for (size_t i = 0; i < stats.size(); i++)
		{
			if (s == stats[i])
			{
				// Add the data point
				ret->Data().Set(i, static_cast<T>(stod(elems[13])));
			}
		}
	}

	return ret;
}

template shared_ptr<info<double>> util::CSVToInfo<double>(const vector<string>&);
template shared_ptr<info<float>> util::CSVToInfo<float>(const vector<string>&);

double util::MissingPercent(const himan::info<double>& info)
{
	auto cp = info;

	cp.First<forecast_type>();
	cp.First<forecast_time>();
	cp.First<level>();
	cp.Reset<param>();

	size_t missing = 0, total = 0;

	while (cp.Next())
	{
		if (cp.IsValidGrid() == false)
		{
			continue;
		}

		const auto g = cp.Grid();

		missing += cp.Data().MissingCount();
		total += cp.Data().Size();
	}

	return (total == 0) ? himan::MissingDouble()
	                    : static_cast<int>(100 * static_cast<float>(missing) / static_cast<float>(total));
}

bool util::ParseBoolean(const string& val)
{
	string lval = boost::algorithm::to_lower_copy(val);

	if (lval == "yes" || lval == "true" || lval == "1")
	{
		return true;
	}
	else if (lval == "no" || lval == "false" || lval == "0")
	{
		return false;
	}
	else
	{
		cerr << "Invalid boolean value: " << val << endl;
		himan::Abort();
	}
}

#ifdef HAVE_CUDA
template <typename T>
void util::Unpack(vector<shared_ptr<info<T>>> infos, bool addToCache)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	auto c = GET_PLUGIN(cache);

	for (auto& info : infos)
	{
		auto& data = info->Data();
		auto pdata = info->PackedData();

		if (pdata == nullptr || pdata->HasData() == false)
		{
			// Safeguard: This particular info does not have packed data
			continue;
		}

		ASSERT(pdata->packingType == kSimplePacking);

		T* arr = 0;
		const size_t N = pdata->unpackedLength;

		ASSERT(N > 0);
		ASSERT(data.Size() == N);

		arr = const_cast<T*>(data.ValuesAsPOD());

		CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(arr), sizeof(T) * N, 0));

		ASSERT(arr);

		packing::Unpack<T>(dynamic_pointer_cast<simple_packed>(pdata).get(), arr, &stream);

		CUDA_CHECK(cudaHostUnregister(arr));
		CUDA_CHECK(cudaStreamSynchronize(stream));

		pdata->Clear();

		if (addToCache)
		{
			c->Insert(info);
		}
	}
	CUDA_CHECK(cudaStreamDestroy(stream));
}

template void util::Unpack<double>(vector<shared_ptr<info<double>>>, bool);
template void util::Unpack<float>(vector<shared_ptr<info<float>>>, bool);

#endif

unique_ptr<grid> util::GridFromDatabase(const string& geom_name)
{
	using himan::kBottomLeft;
	using himan::kTopLeft;

	unique_ptr<grid> g;

	auto r = GET_PLUGIN(radon);

	auto geominfo = r->RadonDB().GetGeometryDefinition(geom_name);

	if (geominfo.empty())
	{
		throw invalid_argument(geom_name + " not found from database.");
	}

	const auto scmode = HPScanningModeFromString.at(geominfo["scanning_mode"]);

	if (geominfo["grid_type_id"] == "1")
	{
		g = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid);
		latitude_longitude_grid* const llg = dynamic_cast<latitude_longitude_grid*>(g.get());

		const double di = stod(geominfo["pas_longitude"]);
		const double dj = stod(geominfo["pas_latitude"]);

		llg->Ni(static_cast<size_t>(stol(geominfo["col_cnt"])));
		llg->Nj(static_cast<size_t>(stol(geominfo["row_cnt"])));
		llg->Di(di);
		llg->Dj(dj);
		llg->ScanningMode(scmode);

		const double X0 = stod(geominfo["long_orig"]);
		const double Y0 = stod(geominfo["lat_orig"]);

		const double X1 = fmod(X0 + static_cast<double>(llg->Ni() - 1) * di, 360);
		double Y1 = kHPMissingValue;

		switch (llg->ScanningMode())
		{
			case kTopLeft:
				Y1 = Y0 - static_cast<double>(llg->Nj() - 1) * dj;
				break;
			case kBottomLeft:
				Y1 = Y0 + static_cast<double>(llg->Nj() - 1) * dj;
				break;
			default:
				break;
		}

		llg->FirstPoint(point(X0, Y0));
		llg->LastPoint(point(X1, Y1));
	}
	else if (geominfo["grid_type_id"] == "4")
	{
		g = unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid);
		rotated_latitude_longitude_grid* const rllg = dynamic_cast<rotated_latitude_longitude_grid*>(g.get());

		const double di = stod(geominfo["pas_longitude"]);
		const double dj = stod(geominfo["pas_latitude"]);

		rllg->Ni(static_cast<size_t>(stol(geominfo["col_cnt"])));
		rllg->Nj(static_cast<size_t>(stol(geominfo["row_cnt"])));
		rllg->Di(di);
		rllg->Dj(dj);
		rllg->ScanningMode(scmode);

		rllg->SouthPole(point(stod(geominfo["geom_parm_2"]), stod(geominfo["geom_parm_1"])));

		const double X0 = stod(geominfo["long_orig"]);
		const double Y0 = stod(geominfo["lat_orig"]);

		const double X1 = fmod(X0 + static_cast<double>(rllg->Ni() - 1) * di, 360);

		double Y1 = kHPMissingValue;

		switch (rllg->ScanningMode())
		{
			case kTopLeft:
				Y1 = Y0 - static_cast<double>(rllg->Nj() - 1) * dj;
				break;
			case kBottomLeft:
				Y1 = Y0 + static_cast<double>(rllg->Nj() - 1) * dj;
				break;
			default:
				break;
		}

		rllg->FirstPoint(point(X0, Y0));
		rllg->LastPoint(point(X1, Y1));
	}
	else if (geominfo["grid_type_id"] == "2")
	{
		g = unique_ptr<stereographic_grid>(new stereographic_grid);
		stereographic_grid* const sg = dynamic_cast<stereographic_grid*>(g.get());

		const double di = stod(geominfo["pas_longitude"]);
		const double dj = stod(geominfo["pas_latitude"]);

		sg->Orientation(stod(geominfo["geom_parm_1"]));
		sg->Di(di);
		sg->Dj(dj);

		sg->Ni(static_cast<size_t>(stol(geominfo["col_cnt"])));
		sg->Nj(static_cast<size_t>(stol(geominfo["row_cnt"])));
		sg->ScanningMode(scmode);

		const double X0 = stod(geominfo["long_orig"]);
		const double Y0 = stod(geominfo["lat_orig"]);

		sg->FirstPoint(point(X0, Y0));
	}

	else if (geominfo["grid_type_id"] == "6")
	{
		g = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
		reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(g.get());

		gg->N(stoi(geominfo["n"]));

		auto strlongitudes = himan::util::Split(geominfo["longitudes_along_parallels"], ",", false);
		vector<int> longitudes;

		for (auto& l : strlongitudes)
		{
			longitudes.push_back(stoi(l));
		}

		gg->NumberOfPointsAlongParallels(longitudes);
	}

	else if (geominfo["grid_type_id"] == "5")
	{
		g = unique_ptr<lambert_conformal_grid>(new lambert_conformal_grid);
		lambert_conformal_grid* const lcg = dynamic_cast<lambert_conformal_grid*>(g.get());

		lcg->Ni(stoi(geominfo["ni"]));
		lcg->Nj(stoi(geominfo["nj"]));

		lcg->Di(stod(geominfo["di"]));
		lcg->Dj(stod(geominfo["dj"]));

		lcg->Orientation(stod(geominfo["orientation"]));

		lcg->StandardParallel1(stod(geominfo["latin1"]));

		if (geominfo["latin2"].empty())
		{
			lcg->StandardParallel2(lcg->StandardParallel1());
		}
		else
		{
			lcg->StandardParallel2(stod(geominfo["latin2"]));
		}

		if (!geominfo["south_pole_lon"].empty())
		{
			const point sp(stod(geominfo["south_pole_lon"]), stod(geominfo["south_pole_lat"]));

			lcg->SouthPole(sp);
		}

		const point first(stod(geominfo["first_point_lon"]), stod(geominfo["first_point_lat"]));

		lcg->ScanningMode(scmode);

		if (geominfo["scanning_mode"] == "+x-y")
		{
			lcg->TopLeft(first);
		}
		else if (geominfo["scanning_mode"] == "+x+y")
		{
			lcg->BottomLeft(first);
		}
	}

	else
	{
		throw invalid_argument("Invalid grid type id for geometry " + geom_name);
	}
	// Until shape of earth is added to radon, hard code default value for all geoms
	// in radon to sphere with radius 6371220, which is the one used in newbase
	// (in most cases that's not *not* the one used by the weather model).
	// Exception to this lambert conformal conic where we use radius 6367470.

	if (g)
	{
		if (g->Type() == kLambertConformalConic)
		{
			g->EarthShape(earth_shape<double>(6367470.));
		}
		else
		{
			g->EarthShape(earth_shape<double>(6371220.));
		}
	}

	return g;
}

template <typename T>
void util::Flip(matrix<T>& mat)
{
	// Flip with regards to x axis

	const size_t halfSize = static_cast<size_t>(floor(mat.SizeY() / 2));

	for (size_t y = 0; y < halfSize; y++)
	{
		for (size_t x = 0; x < mat.SizeX(); x++)
		{
			T upper = mat.At(x, y);
			T lower = mat.At(x, mat.SizeY() - 1 - y);

			mat.Set(x, y, 0, lower);
			mat.Set(x, mat.SizeY() - 1 - y, 0, upper);
		}
	}
}

template void util::Flip<double>(matrix<double>&);
template void util::Flip<float>(matrix<float>&);
