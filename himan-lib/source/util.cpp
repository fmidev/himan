/**
 * @file util.cpp
 *
 * @brief Different utility functions and classes in a namespace
 */

#include "util.h"
#include "cuda_helper.h"
#include "forecast_time.h"
#include "lagged_ensemble.h"
#include "lambert_conformal_grid.h"
#include "lambert_equal_area_grid.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "numerical_functions.h"
#include "param.h"
#include "plugin_factory.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "s3.h"
#include "stereographic_grid.h"
#include "time_ensemble.h"
#include "transverse_mercator_grid.h"
#include <boost/algorithm/string.hpp>
#include <boost/math/constants/constants.hpp>
#include <filesystem>
#include <fmt/printf.h>
#include <iomanip>
#include <ogr_spatialref.h>
#include <regex>
#include <sstream>
#include <wordexp.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "cache.h"
#include "radon.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

himan::HPFileType util::FileType(const string& theFile)
{
	using namespace std;

	// First check by extension since its cheap

	std::filesystem::path p(theFile);

	if (p.extension().string() == ".gz" || p.extension().string() == ".bz2")
	{
		p = p.stem();
	}

	string ext = p.extension().string();
	boost::to_lower(ext);

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
	else if (ext == ".tif" || ext == ".tiff")
	{
		return kGeoTIFF;
	}
	// Try the check the file header; CSV is not possible anymore

	const long keywordLength = 9;
	char content[keywordLength];

	if (theFile.substr(0, 5) == "s3://")
	{
		himan::file_information finfo;
		finfo.message_no = std::nullopt;
		finfo.offset = 0;
		finfo.length = 5;
		finfo.storage_type = himan::kS3ObjectStorageSystem;
		finfo.file_location = theFile;
		finfo.file_server = util::GetEnv("S3_HOSTNAME");
		auto buffer = s3::ReadFile(finfo);
		ASSERT(keywordLength == buffer.length);
		memcpy(&content, buffer.data, keywordLength);
	}
	else
	{
		ifstream f(theFile.c_str(), ios::in | ios::binary);
		f.read(content, keywordLength);
	}

	HPFileType ret = kUnknownFile;

	static const char* grib = "GRIB";
	static const unsigned char ncv3[4] = {0x43, 0x44, 0x46, 0x01};
	static const unsigned char ncv4[4] = {0xD3, 0x48, 0x44, 0x46};  // 211 H D F
	static const char* tiff = "II*\0";
	static const char* qinfo = "@$\260\243QINFO";

	if (strncmp(content, grib, 4) == 0)
	{
		ret = kGRIB;
	}
	else if (memcmp(content, ncv3, 4) == 0)
	{
		ret = kNetCDF;
	}
	else if (memcmp(content, ncv4, 4) == 0)
	{
		ret = kNetCDFv4;
	}
	else if (memcmp(content, tiff, 4) == 0)
	{
		ret = kGeoTIFF;
	}
	else if (strncmp(content, qinfo, 9) == 0)
	{
		ret = kQueryData;
	}

	return ret;
}

namespace
{
template <typename T>
T TypeCast(const std::string& str)
{
	std::istringstream ss(str);
	T num;
	ss >> num;
	return num;
}

template <>
std::string TypeCast<std::string>(const std::string& val)
{
	return val;
}
}  // namespace

template <typename T>
vector<T> util::Split(const string& s, const string& delims)
{
	vector<string> orig;

	boost::split(orig, s, boost::is_any_of(delims));

	vector<T> ret;
	ret.reserve(orig.size());
	std::transform(orig.begin(), orig.end(), std::back_inserter(ret),
	               [](std::string& ss) { return ::TypeCast<T>(ss); });

	return ret;
}

template vector<string> util::Split<string>(const string&, const string&);
template vector<float> util::Split<float>(const string&, const string&);
template vector<double> util::Split<double>(const string&, const string&);
template vector<int> util::Split<int>(const string&, const string&);
template vector<size_t> util::Split<size_t>(const string&, const string&);

vector<int> util::ExpandString(const std::string& identifier)
{
	// identifier is a string with number separated by commas and dashes
	// 1,5,10-12
	// --> return a vector of:
	// 1,5,10,11,12
	// 1,5,10-16-2
	// --> return a vector of:
	// 1,5,10,12,14,16

	vector<int> ret;
	const auto split1 = Split(identifier, ",");
	for (const auto& tok : split1)
	{
		auto split2 = Split<int>(tok, "-");

		int step = 1;

		if (split2.size() == 3)
		{
			step = split2[2];
			split2.pop_back();
		}

		if (split2.size() == 2)
		{
			int a = split2[0], b = split2[1];
			if (a > b)
			{
				step *= -1;
			}

			const auto vals = numerical_functions::Arange(
			    split2[0], split2[1] + step, step);  // arange return half-open interval excluding the endvalue
			std::copy(vals.begin(), vals.end(), std::back_inserter(ret));
		}
		else if (split2.size() == 1)
		{
			ret.push_back(split2[0]);
		}
	}
	return ret;
}

vector<time_duration> util::ExpandTimeDuration(const std::string& identifier)
{
	vector<time_duration> ret;
	const auto split1 = Split(identifier, ",");
	for (const auto& tok : split1)
	{
		auto split2 = Split<std::string>(tok, "-");

		time_duration step = ONE_HOUR;

		if (split2.size() == 3)
		{
			step = time_duration(split2[2]);
			split2.pop_back();
		}

		if (split2.size() == 2)
		{
			time_duration a = split2[0], b = split2[1];
			if (a > b)
			{
				step *= -1;
			}

			while (a != b)
			{
				ret.push_back(time_duration(a));
				a += step;
			}
			ret.push_back(time_duration(a));  // include end range value
		}
		else if (split2.size() == 1)
		{
			ret.push_back(split2[0]);
		}
	}
	return ret;
}

vector<string> util::Split(const string& s, const string& delims)
{
	return Split<std::string>(s, delims);
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
pair<matrix<T>, matrix<T>> util::CentralDifference(matrix<T>& A, vector<T>& dx, vector<T>& dy, bool jPositive)
{
	matrix<T> dA_dx(A.SizeX(), A.SizeY(), 1, A.MissingValue());
	matrix<T> dA_dy(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());

	ASSERT(dy.size() == A.SizeX() && dx.size() == A.SizeY());

	// compute x-derivative
	for (int j = 0; j < ASizeY; ++j)
	{
		T dxj = jPositive ? dx[j] : dx[ASizeY - 1 - j];
		for (int i = 1; i < ASizeX - 1; ++i)  // columns
		{
			dA_dx.Set(i, j, 0,
			          (A.At(i + 1, j, 0) - A.At(i - 1, j, 0)) / (2 * dxj));  // central difference in x-direction
		}

		// left boundary
		dA_dx.Set(0, j, 0, (A.At(1, j, 0) - A.At(0, j, 0)) / dxj);  // forward difference in x-direction

		// right boundary
		dA_dx.Set(ASizeX - 1, j, 0,
		          (A.At(ASizeX - 1, j, 0) - A.At(ASizeX - 2, j, 0)) / dxj);  // backward difference in x-direction
	}

	if (jPositive)
	{
		for (int i = 0; i < ASizeX; ++i)
		{
			for (int j = 1; j < ASizeY - 1; ++j)
			{
				dA_dy.Set(i, j, 0,
				          (A.At(i, j + 1, 0) - A.At(i, j - 1, 0)) / (2 * dy[i]));  // central difference in y-direction
			}
			dA_dy.Set(i, 0, 0, (A.At(i, 1, 0) - A.At(i, 0, 0)) / dy[i]);  // forward difference in y-direction
			dA_dy.Set(i, ASizeY - 1, 0,
			          (A.At(i, ASizeY - 1, 0) - A.At(i, ASizeY - 2, 0)) / dy[i]);  // backward difference in y-direction
		}
	}
	else
	{
		for (int i = 0; i < ASizeX; ++i)
		{
			for (int j = 1; j < ASizeY - 1; ++j)
			{
				dA_dy.Set(i, j, 0,
				          -(A.At(i, j + 1, 0) - A.At(i, j - 1, 0)) / (2 * dy[i]));  // central difference in y-direction
			}
			dA_dy.Set(i, 0, 0, -(A.At(i, 1, 0) - A.At(i, 0, 0)) / dy[i]);  // forward difference in y-direction
			dA_dy.Set(
			    i, ASizeY - 1, 0,
			    -(A.At(i, ASizeY - 1, 0) - A.At(i, ASizeY - 2, 0)) / dy[i]);  // backward difference in y-direction
		}
	}

	pair<matrix<T>, matrix<T>> ret(dA_dx, dA_dy);
	return ret;
}

template pair<matrix<double>, matrix<double>> util::CentralDifference<double>(matrix<double>& A, vector<double>& dx,
                                                                              vector<double>& dy, bool jPositive);
template pair<matrix<float>, matrix<float>> util::CentralDifference<float>(matrix<float>& A, vector<float>& dx,
                                                                           vector<float>& dy, bool jPositive);

template <typename T>
T util::LatitudeLength(T phi)
{
	T cos_phi = cos(phi * static_cast<T>(constants::kDeg));
	return 2 * boost::math::constants::pi<T>() * static_cast<float>(constants::kR) * abs(cos_phi);
}
template double util::LatitudeLength(double phi);
template float util::LatitudeLength(float phi);

double util::round(double val, unsigned short numdigits)
{
	int div = static_cast<int>(pow(10, static_cast<double>(numdigits)));
	return std::round(val * div) / div;
}

string util::MakeSQLInterval(const himan::forecast_time& theTime)
{
	return static_cast<string>(theTime.Step());
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

void util::DumpVector(const vector<float>& vec, const string& name)
{
	return DumpVector<float>(vec, name);
}

template <typename T>
void util::DumpVector(const vector<T>& vec, const string& name)
{
	T min = numeric_limits<T>::max(), max = numeric_limits<T>::lowest();
	double sum = 0;
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
		sum += static_cast<double>(val);
	}

	double mean = numeric_limits<double>::quiet_NaN();

	if (count > 0)
	{
		mean = static_cast<double>(sum / static_cast<double>(count));
	}

	if (!name.empty())
	{
		fmt::print("{}\t", name);
	}

	fmt::print("min: {} max: {} mean: {:.3f} count: {} nan: {} missing: {}\n", min, max, mean, count, nan, missing);

	if (min != max && count > 0)
	{
		long binn = (count < 10) ? count : 10;

		T binw = static_cast<T>((max - min) / static_cast<T>(binn));

		T binmin = min;
		T binmax = static_cast<T>(binmin + binw);

		fmt::print("distribution (bins={}):\n", binn);

		for (int i = 1; i <= binn; i++)
		{
			// if (i == binn)
			//	binmax += 0.001f;
			//
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

			// if (i == binn)
			//	binmax -= 0.001f;

			fmt::print("{}:{} {}\n", binmin, binmax, count);

			binmin = static_cast<T>(binmin + binw);
			binmax = static_cast<T>(binmax + binw);
		}
	}
}

template void util::DumpVector<double>(const vector<double>&, const string&);
template void util::DumpVector<float>(const vector<float>&, const string&);
template void util::DumpVector<short>(const vector<short>&, const string&);
template void util::DumpVector<unsigned char>(const vector<unsigned char>&, const string&);

string util::GetEnv(const string& key)
{
	const char* var = getenv(key.c_str());
	if (!var)
	{
		throw std::invalid_argument("Environment variable '" + string(key) + "' not found");
	}
	return string(var);
}

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

		const auto pck = std::dynamic_pointer_cast<simple_packed>(pdata);

		if (std::is_same<T, double>::value || std::is_same<T, float>::value)
		{
			NFmiGribPacking::simple_packing::Unpack<T>(arr, pck->data, pck->bitmap, pck->unpackedLength,
			                                           pck->packedLength, pck->coefficients, stream);
		}
		else
		{
			fmt::print("cuda unpacking for other data types than double and float not implemented yet\n");
			himan::Abort();
		}

		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaHostUnregister(arr));

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
template void util::Unpack<short>(vector<shared_ptr<info<short>>>, bool);
template void util::Unpack<unsigned char>(vector<shared_ptr<info<unsigned char>>>, bool);

#endif

unique_ptr<grid> util::GridFromDatabase(const string& geom_name)
{
	using himan::kBottomLeft;
	using himan::kTopLeft;

	logger logr("util");

	auto r = GET_PLUGIN(radon);

	auto geominfo = r->RadonDB().GetGeometryDefinition(geom_name);

	if (geominfo.empty())
	{
		logr.Fatal(fmt::format("Geometry '{}' not found from database", geom_name));
		himan::Abort();
	}

	const auto scmode = HPScanningModeFromString.at(geominfo["scanning_mode"]);

	int gridid;

	try
	{
		gridid = stoi(geominfo["grid_type_id"]);
	}
	catch (const invalid_argument& e)
	{
		logr.Fatal(fmt::format("{} is not an integer", geominfo["grid_type_id"]));
		himan::Abort();
	}

	// Until shape of earth is added to radon, hard code default value for all geoms
	// in radon to sphere with radius 6371220, which is the one used in newbase
	// (in most cases that's not *not* the one used by the weather model).
	// Exception to this lambert conformal conic where we use radius 6367470.

	auto earth = (gridid == 5) ? earth_shape<double>(6367470.) : earth_shape<double>(6371220.);

	try
	{
		earth =
		    earth_shape<double>(stod(geominfo["earth_semi_major"]), stod(geominfo["earth_semi_minor"]),
		                        geominfo.count("earth_ellipsoid_name") == 1 && !geominfo["earth_ellipsoid_name"].empty()
		                            ? geominfo["earth_ellipsoid_name"]
		                            : "");
	}
	catch (const invalid_argument& e)
	{
	}

	// TODO: in the future when proj4 string information has more coverage in radon,
	// we can just create the area directly with that without the switch below

	switch (gridid)
	{
		case 1:
			// clang-format off
		return unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(
		    scmode,
		    point(stod(geominfo["long_orig"]), stod(geominfo["lat_orig"])),
		    stol(geominfo["col_cnt"]),
		    stol(geominfo["row_cnt"]),
		    stod(geominfo["pas_longitude"]),
		    stod(geominfo["pas_latitude"]),
		    earth,
		    geominfo["name"]
		));
			// clang-format on

		case 4:
			// clang-format off
		return unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid(
		    scmode,
		    point(stod(geominfo["long_orig"]), stod(geominfo["lat_orig"])),
		    stol(geominfo["col_cnt"]),
		    stol(geominfo["row_cnt"]),
		    stod(geominfo["pas_longitude"]),
		    stod(geominfo["pas_latitude"]),
		    earth,
		    point(stod(geominfo["geom_parm_2"]), stod(geominfo["geom_parm_1"])),
		    true,
		    geominfo["name"]
		));
			// clang-format on

		case 2:
			// clang-format off
		return unique_ptr<stereographic_grid>(new stereographic_grid(
		    scmode,
		    point(stod(geominfo["long_orig"]), stod(geominfo["lat_orig"])),
		    stol(geominfo["col_cnt"]),
		    stol(geominfo["row_cnt"]),
		    stod(geominfo["pas_longitude"]),
		    stod(geominfo["pas_latitude"]),
		    stod(geominfo["geom_parm_1"]),
		    stod(geominfo["latin"]),
		    stod(geominfo["lat_ts"]),
		    earth,
		    false,
		    geominfo["name"]
		));
			// clang-format on

		case 6:
		{
			auto g = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
			g->EarthShape(earth);

			reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(g.get());

			gg->N(stoi(geominfo["n"]));

			auto strlongitudes = himan::util::Split(geominfo["longitudes_along_parallels"], ",");
			vector<int> longitudes;
			longitudes.reserve(strlongitudes.size());

			for (auto& l : strlongitudes)
			{
				longitudes.push_back(stoi(l));
			}

			gg->NumberOfPointsAlongParallels(longitudes);
			return g;
		}

		case 5:
			// clang-format off
		return unique_ptr<lambert_conformal_grid>(new lambert_conformal_grid(
		    scmode,
		    point(stod(geominfo["first_point_lon"]), stod(geominfo["first_point_lat"])),
		    stoi(geominfo["ni"]),
		    stoi(geominfo["nj"]),
		    stod(geominfo["di"]),
		    stod(geominfo["dj"]),
		    stod(geominfo["orientation"]),
		    stod(geominfo["latin1"]),
		    (geominfo["latin2"].empty() ? stod(geominfo["latin1"]) : stod(geominfo["latin2"])),
		    earth,
		    false,
		    geominfo["name"]
		));
			// clang-format on

		case 7:
			// clang-format off
		return unique_ptr<lambert_equal_area_grid>(new lambert_equal_area_grid(
		    scmode,
		    point(stod(geominfo["first_point_lon"]), stod(geominfo["first_point_lat"])),
		    stoi(geominfo["ni"]),
		    stoi(geominfo["nj"]),
		    stod(geominfo["di"]),
		    stod(geominfo["dj"]),
		    stod(geominfo["orientation"]),
		    stod(geominfo["latin"]),
		    earth,
		    false,
		    geominfo["name"]
		));
			// clang-format on

		case 8:
			// clang-format off
		return unique_ptr<transverse_mercator_grid>(new transverse_mercator_grid(
		    scmode,
		    point(stod(geominfo["first_point_lon"]), stod(geominfo["first_point_lat"])),
		    stoi(geominfo["ni"]),
		    stoi(geominfo["nj"]),
		    stod(geominfo["di"]),
		    stod(geominfo["dj"]),
		    stod(geominfo["orientation"]),
		    stod(geominfo["latin"]),
		    stod(geominfo["scale"]),
		    0, // TODO: false easting, this value might not be correct
		    0, // TODO: false northing, this value might not be correct
		    earth,
		    false,
		    geominfo["name"]
		));
			// clang-format on

		default:
			logr.Fatal(fmt::format("Invalid grid type id '{}' for geometry '{}'", gridid, geom_name));
			himan::Abort();
	}
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
template void util::Flip<short>(matrix<short>&);
template void util::Flip<unsigned char>(matrix<unsigned char>&);

string util::UniqueName(const plugin::search_options& options)
{
	ASSERT(options.configuration->DatabaseType() == kNoDatabase || options.prod.Id() != kHPMissingInt);

	try
	{
		return fmt::format("{}_{}_{}_{}_{}_{}_{}_{}", options.prod.Id(), options.time.OriginDateTime().ToSQLTime(),
		                   options.time.ValidDateTime().ToSQLTime(), options.param.Name(),
		                   static_cast<string>(options.level), static_cast<int>(options.ftype.Type()),
		                   options.ftype.Value(), options.configuration->BaseGrid()->Size());
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

template <typename T>
string util::UniqueName(const info<T>& info)
{
	try
	{
		return fmt::format("{}_{}_{}_{}_{}_{}_{}_{}", info.Producer().Id(), info.Time().OriginDateTime().ToSQLTime(),
		                   info.Time().ValidDateTime().ToSQLTime(), info.Param().Name(),
		                   static_cast<string>(info.Level()), static_cast<int>(info.ForecastType().Type()),
		                   info.ForecastType().Value(), info.Grid()->Size());
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

template string util::UniqueName(const info<double>&);
template string util::UniqueName(const info<float>&);
template string util::UniqueName(const info<short>&);
template string util::UniqueName(const info<unsigned char>&);

aggregation util::GetAggregationFromParamName(const std::string& name, const forecast_time& ftime)
{
	// match any precipitation accumulation, also probabilities
	// eg. PROB-RR12-1 RRR-KGM2 RR-3-MM SN-6-MM
	const std::regex r1(R"((?:PROB-)?(?:CONV-)?(?:SN|RR|RRS|RRC|RRL|RAIN|RAINL|RAINC)-?([0-9]+)-(?:KGM2|MM|M|\d+)$)");
	// match any max or min parameter
	// note: only time dimension is considered, for example not
	// maximum thickness of some level
	// eg. TMAX-K, TMIN12H-C
	const std::regex r2("(?:T|POT)(MAX|MIN)([0-9]+H)?-[A-Z0-9]+");
	// match wind gust, also components
	// eg. FFG-MS, FFG3H-MS, WGU-MS
	const std::regex r3("(?:FFG|WGU|WGV)([0-9]+H)?-MS");
	// match one hour precipitation accumulations
	const std::regex r4("SNR[CL]-KGM2|SNR-KGM2|RRR-KGM2|RRR[SLC]-KGM2");
	// match precipitation accumulation from forecast start
	const std::regex r5("RR-KGM2|SNACC-KGM2|RR[CL]-KGM2|RAIN[CL]-KGM2");
	// match daily periods
	const std::regex r6("PREC([0-9D]+)-(?:KGM2|MM)");

	std::smatch m;

	if (std::regex_match(name, m, r1))
	{
		const std::string& period = m[1];

		try
		{
			return himan::aggregation(kAccumulation, ONE_HOUR * stoi(period));
		}
		catch (const std::exception& e)
		{
			// de nada
		}
	}
	else if (std::regex_match(name, m, r2))
	{
		const std::string& at = m[1];

		try
		{
			if (m[2].matched)
			{
				const std::string& period = m[2];
				return himan::aggregation(at == "MIN" ? kMinimum : kMaximum, ONE_HOUR * stoi(period));
			}
			else
			{
				return himan::aggregation(at == "MIN" ? kMinimum : kMaximum, ONE_HOUR);
			}
		}
		catch (const std::exception& e)
		{
		}
	}
	else if (std::regex_match(name, m, r3))
	{
		try
		{
			const int period = (m[1].matched) ? stoi(m[1]) : 1;
			return himan::aggregation(kMaximum, ONE_HOUR * period);
		}
		catch (const std::exception& e)
		{
		}
	}
	else if (std::regex_search(name, m, r4))
	{
		return himan::aggregation(kAccumulation, ONE_HOUR);
	}
	else if (std::regex_search(name, m, r5))
	{
		return himan::aggregation(kAccumulation, ftime.Step());
	}
	else if (std::regex_match(name, m, r6))
	{
		std::string period = m[1];

		period.erase(std::remove(period.begin(), period.end(), 'D'), period.end());

		try
		{
			return himan::aggregation(kAccumulation, ONE_HOUR * 24 * stoi(period));
		}
		catch (const std::exception& e)
		{
		}
	}

	return himan::aggregation();
}

processing_type util::GetProcessingTypeFromParamName(const string& name)
{
	if (name.find("-STDDEV-") != string::npos)
	{
		return himan::processing_type(kStandardDeviation);
	}
	else if (name.find("EFI-") != string::npos)
	{
		return himan::processing_type(kEFI);
	}
	// No support for fractile/probability yet

	return himan::processing_type();
}

param util::InitializeParameter(const producer& prod, const param& par, const level& lvl)
{
	logger logr("util");
	auto r = GET_PLUGIN(radon);

	param p = par;

	if (lvl.Type() != kUnknownLevel)
	{
		auto levelInfo =
		    r->RadonDB().GetLevelFromDatabaseName(boost::to_upper_copy(HPLevelTypeToString.at(lvl.Type())));

		if (levelInfo.empty())
		{
			logr.Warning(fmt::format("Level type '{}' not found from radon", HPLevelTypeToString.at(lvl.Type())));
			return par;
		}

		auto paraminfo =
		    r->RadonDB().GetParameterFromDatabaseName(prod.Id(), par.Name(), stoi(levelInfo["id"]), lvl.Value());

		if (paraminfo.empty())
		{
			logr.Warning(fmt::format("Parameter '{}' not found from radon", par.Name()));
			return par;
		}

		p = param(paraminfo);
	}

	if (par.InterpolationMethod() != kUnknownInterpolationMethod)
	{
		// User has specified interpolation method, use that
		p.InterpolationMethod(par.InterpolationMethod());
	}
	else
	{
		// Ignore interpolation method from database - those
		// are not up to date. By setting method to unknown here,
		// the correct method will be picked up at interpolation.cpp
		p.InterpolationMethod(kUnknownInterpolationMethod);
	}

	// database does not provide aggregation or processing type information,
	// but we can guess, unless the calling code has already passed an aggregation

	if (par.ProcessingType().Type() == kUnknownProcessingType)
	{
		p.ProcessingType(GetProcessingTypeFromParamName(p.Name()));
	}
	else
	{
		p.ProcessingType(par.ProcessingType());
	}

	// If processing type is ensemble mean, do not set aggregation to mean as aggregation
	// mainly describes *time* based aggregation. But we don't know that from database name only.
	if (par.Aggregation().Type() == kUnknownAggregationType)
	{
		p.Aggregation(GetAggregationFromParamName(p.Name(), forecast_time()));
	}
	else
	{
		p.Aggregation(par.Aggregation());
	}

	return p;
}

vector<forecast_type> util::ForecastTypesFromString(const string& types)
{
	vector<forecast_type> forecastTypes;
	vector<string> typeList = util::Split(types, ",");

	for (string& type : typeList)
	{
		boost::algorithm::to_lower(type);

		if (type.find("pf") != string::npos)
		{
			string list = type.substr(2, string::npos);

			vector<string> range = util::Split(list, "-");

			if (range.size() == 1)
			{
				forecastTypes.push_back(forecast_type(kEpsPerturbation, stod(range[0])));
			}
			else
			{
				ASSERT(range.size() == 2);

				int start = stoi(range[0]);
				int stop = stoi(range[1]);

				while (start <= stop)
				{
					forecastTypes.push_back(forecast_type(kEpsPerturbation, start));
					start++;
				}
			}
		}
		else if (type.find("cf") != string::npos)
		{
			const string strnum = type.substr(2, string::npos);
			int num = 0;

			if (!strnum.empty())
			{
				try
				{
					num = stoi(strnum);
				}
				catch (const invalid_argument& e)
				{
					throw std::invalid_argument(fmt::format("Invalid forecast type specifier: {}", type));
				}
			}

			forecastTypes.push_back(forecast_type(kEpsControl, num));
		}
		else if (type == "det" || type == "deterministic")
		{
			forecastTypes.push_back(forecast_type(kDeterministic));
		}
		else if (type == "an" || type == "analysis")
		{
			forecastTypes.push_back(forecast_type(kAnalysis));
		}
		else if (type == "sp" || type == "statistical")
		{
			forecastTypes.push_back(forecast_type(kStatisticalProcessing));
		}
	}

	return forecastTypes;
}

std::unique_ptr<ensemble> util::CreateEnsembleFromConfiguration(const std::shared_ptr<const plugin_configuration>& conf)
{
	std::unique_ptr<ensemble> ens;

	logger log("util");

	std::string paramName = conf->GetValue("param");

	if (paramName.empty())
	{
		paramName = "XX-X";
	}

	const aggregation agg(conf->GetValue("aggregation"));

	const param par(paramName, agg, processing_type());

	// ENSEMBLE TYPE

	HPEnsembleType ensType = kPerturbedEnsemble;

	if (conf->GetValue("ensemble_type").empty() == false)
	{
		ensType = HPStringToEnsembleType.at(conf->GetValue("ensemble_type"));
	}
	else if (conf->GetValue("lag").empty() == false || conf->GetValue("lagged_members").empty() == false ||
	         conf->GetValue("named_ensemble").empty() == false)
	{
		ensType = kLaggedEnsemble;
	}
	else if (conf->GetValue("secondary_time_len").empty() == false)
	{
		ensType = kTimeEnsemble;
	}

	// ENSEMBLE SIZE

	size_t ensSize = 0;

	if (conf->Exists("ensemble_size"))
	{
		ensSize = std::stoi(conf->GetValue("ensemble_size"));
	}
	else if (ensType == kPerturbedEnsemble || ensType == kLaggedEnsemble)
	{
		// Regular ensemble size is static, get it from database if user
		// hasn't specified any size

		auto r = GET_PLUGIN(radon);

		std::string ensembleSizeStr = r->RadonDB().GetProducerMetaData(conf->SourceProducer(0).Id(), "ensemble size");

		if (ensembleSizeStr.empty())
		{
			log.Error("Unable to find ensemble size from database");
			return nullptr;
		}

		ensSize = std::stoi(ensembleSizeStr);
	}

	int maximumMissing = 0;

	if (conf->Exists("max_missing_forecasts"))
	{
		maximumMissing = std::stoi(conf->GetValue("max_missing_forecasts"));
	}

	switch (ensType)
	{
		case kPerturbedEnsemble:
		{
			if (conf->GetValue("members").empty() == false)
			{
				const std::vector<forecast_type> members = util::ForecastTypesFromString(conf->GetValue("members"));

				ens = make_unique<ensemble>(par, members, maximumMissing);
			}
			else
			{
				ens = make_unique<ensemble>(par, ensSize, maximumMissing);
			}
			break;
		}
		case kTimeEnsemble:
		{
			int secondaryLen = 0, secondaryStep = 1;
			HPTimeResolution secondarySpan = kHourResolution;

			if (conf->Exists(("secondary_time_len")))
			{
				secondaryLen = std::stoi(conf->GetValue("secondary_time_len"));
			}
			if (conf->Exists(("secondary_time_step")))
			{
				secondaryStep = std::stoi(conf->GetValue("secondary_time_step"));
			}
			if (conf->Exists(("secondary_time_span")))
			{
				secondarySpan = HPStringToTimeResolution.at(conf->GetValue("secondary_time_span"));
			}

			ens = make_unique<time_ensemble>(par, ensSize, kYearResolution, secondaryLen, secondaryStep, secondarySpan,
			                                 maximumMissing);
		}
		break;
		case kLaggedEnsemble:
		{
			const auto name = conf->GetValue("named_ensemble");

			if (name.empty() == false)
			{
				ens = make_unique<lagged_ensemble>(par, name, maximumMissing);
			}
			else if (conf->GetValue("lagged_members").empty() == false)
			{
				const std::vector<forecast_type> members =
				    util::ForecastTypesFromString(conf->GetValue("lagged_members"));
				const std::vector<std::string> lags = util::Split(conf->GetValue("lags"), ",");

				if (members.size() != lags.size())
				{
					log.Fatal(fmt::format("Size of members ({}) does not match size of lags ({}): {} and {}",
					                      members.size(), lags.size(), conf->GetValue("lagged_members"),
					                      conf->GetValue("lags")));
					himan::Abort();
				}

				std::vector<std::pair<forecast_type, himan::time_duration>> forecasts;
				forecasts.reserve(members.size());

				for (size_t i = 0; i < members.size(); i++)
				{
					forecasts.emplace_back(members[i], himan::time_duration(lags[i]));
				}

				ens = std::make_unique<lagged_ensemble>(par, forecasts, maximumMissing);
			}
			else
			{
				auto lagstr = conf->GetValue("lag");
				if (lagstr.empty())
				{
					log.Fatal("specify lag value for lagged_ensemble");
					himan::Abort();
				}

				int lag = std::stoi(conf->GetValue("lag"));

				if (lag == 0)
				{
					log.Fatal("lag value needs to be negative integer");
					himan::Abort();
				}
				else if (lag > 0)
				{
					log.Warning("negating lag value " + std::to_string(-lag));
					lag = -lag;
				}

				auto stepsstr = conf->GetValue("lagged_steps");

				if (stepsstr.empty())
				{
					log.Fatal("specify lagged_steps value for lagged_ensemble");
					himan::Abort();
				}

				int steps = std::stoi(conf->GetValue("lagged_steps"));

				if (steps <= 0)
				{
					log.Fatal("invalid lagged_steps value. Allowed range >= 0");
					himan::Abort();
				}

				steps++;

				ens = make_unique<lagged_ensemble>(par, ensSize, time_duration(kHourResolution, lag), steps,
				                                   maximumMissing);
			}
		}
		break;
		default:
			log.Fatal(fmt::format("Unknown ensemble type: {}", ensType));
			himan::Abort();
	}

	ASSERT(ens);

	log.Trace(fmt::format("Created ensemble of type: {}", HPEnsembleTypeToString.at(ens->EnsembleType())));

	return ens;
}

std::pair<long, long> util::GetScaledValue(double v)
{
	std::pair<long, long> s(static_cast<long>(v), 0l);

	if (v == 0.)
	{
		return s;
	}

	// Scale a float value so it can be encoded as a long
	// For example, value=273.15 --> scaled_value=27315, scale_factor=-2
	//              value=100    --> scaled_value=100, scale_factor=0

	const double r = std::fmod(fabs(v), 1.0);  // trailing decimals

	// convert to string so that counting is possible
	// note to future refactorers: __builtin_ctz() should not be used here

	auto str = fmt::format("{}", v);

	if (r > 0 || v < 1.0)
	{
		// value has decimals
		// - remove all trailing zeros
		// - count number of digits
		// --> that will be the scale factor
		str.erase(str.find_last_not_of('0') + 1, std::string::npos);
		const auto dot = str.find('.');
		size_t num_digits = (dot == string::npos) ? 0 : str.length() - dot - 1;
		num_digits = std::min(num_digits, static_cast<size_t>(8));  // prevent neverending floats to inflate digitcount
		const double scaled = ::round(
		    v * pow(10., static_cast<double>(num_digits)));  // round before conversion to long,
		                                                     // because f.ex. 273.15 * pow(10, 2) = 27314.999999999996

		s = make_pair(static_cast<long>(scaled), -num_digits);
	}
	else
	{
		// find the position of first zero that has no other digit behind it
		auto last = str.find_last_not_of('0');

		if (last == std::string::npos)
		{
			// No trailing zeros
			return s;
		}

		unsigned lv = static_cast<unsigned>(v);

		// count number of trailing zeros
		size_t nz = str.size() - (last + 1);

		lv /= static_cast<unsigned>(pow(10., static_cast<double>(nz)));

		s = make_pair(lv, -nz);
	}

	return s;
}

level util::CreateHybridLevel(const producer& prod, const std::string& first_last)
{
	if (first_last != "first" && first_last != "last")
	{
		logger logr("util");
		logr.Error(fmt::format("Argument 'first_last' must be either 'first' or 'last', got: {}", first_last));
		return level();
	}

	auto r = GET_PLUGIN(radon);

	const std::string levType = r->RadonDB().GetProducerMetaData(prod.Id(), "hybrid level type");
	const std::string valKey = fmt::format("{} hybrid level number", first_last);

	int val = stoi(r->RadonDB().GetProducerMetaData(prod.Id(), valKey));
	HPLevelType hybType = levType.empty() ? kHybrid : HPStringToLevelType.at(levType);

	switch (hybType)
	{
		default:
		case kHybrid:
			return level(hybType, val);
		case kGeneralizedVerticalLayer:
			return level(hybType, val, val + 1);
	}
}
