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
#include "stereographic_grid.h"
#include "time_ensemble.h"
#include "transverse_mercator_grid.h"
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/regex.hpp>
#include <filesystem>
#include <iomanip>
#include <ogr_spatialref.h>
#include <sstream>
#include <wordexp.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "cache.h"
#include "radon.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

namespace
{
template <typename T>
string MakeFileNameFromTemplate(const info<T>& info, const plugin_configuration& conf, const string filenameTemplate)
{
	const auto ftype = info.template Value<forecast_type>();
	const auto ftime = info.template Value<forecast_time>();
	const auto lvl = info.template Value<level>();
	const auto par = info.template Value<param>();
	const auto prod = info.Producer();

	// Allowed template keys:
	// {analysis_time:FORMAT_SPECIFIER}        - analysis time, default format %Y%m%d%H%M%S
	// {forecast_time:FORMAT_SPECIFIER}        - forecast time, default format %Y%m%d%H%M%S
	// {step:FORMAT_SPECIFIER}                 - leadtime, default format %H:%M
	// {geom_name}                             - geometry name
	// {grid_name}                             - grid (projection) name
	// {grid_ni}                               - grid size in x direction
	// {grid_nj}                               - grid size in y direction
	// {param_name}                            - parameter name
	// {aggregation_name}                      - aggregation name
	// {aggregation_duration:FORMAT_SPECIFIER} - aggregation duration
	// {processing_type_name}                  - processing type name
	// {processing_type_value:FORMAT_SPECIFIER}- processing type value
	// {processing_type_value:FORMAT_SPECIFIER}- second possible processing type value
	// {level_name }                           - level name
	// {level_value:FORMAT_SPECIFIER}          - level value
	// {level_value2:FORMAT_SPECIFIER}         - second possible level value
	// {forecast_type_name}                    - forecast type name, like 'sp' or 'det' (short forms used)
	// {forecast_type_id:FORMAT_SPECIFIER}     - forecast type id, 1 .. 5
	// {forecast_type_value:FORMAT_SPECIFIER}  - possible forecast type value
	// {producer_id}                           - radon producer id
	// {file_type}                             - file type extension, like grib, grib2, fqd, ...
	// {wall_time:FORMAT_SPECIFIER}	           - current wall clock time

	enum class Component
	{
		kMasalaBase,
		kAnalysisTime,
		kForecastTime,
		kStep,
		kGeometryName,
		kGridName,
		kGridNi,
		kGridNj,
		kParamName,
		kAggregationName,
		kAggregationDuration,
		kProcessingTypeName,
		kProcessingTypeValue,
		kProcessingTypeValue2,
		kLevelName,
		kLevelValue,
		kLevelValue2,
		kForecastTypeId,
		kForecastTypeName,
		kForecastTypeValue,
		kProducerId,
		kFileType,
		kWallTime
	};

	auto ForecastTypeToShortString = [](HPForecastType type) -> string
	{
		switch (type)
		{
			case kDeterministic:
				return "det";
			case kAnalysis:
				return "an";
			case kEpsControl:
				return "cf";
			case kEpsPerturbation:
				return "pf";
			case kStatisticalProcessing:
				return "sp";
			default:
				return "unknown";
		}
	};

	auto FileTypeToShortString = [](HPFileType type) -> string
	{
		switch (type)
		{
			case kGRIB:
			case kGRIB1:
				return "grib";
			case kGRIB2:
				return "grib2";
			case kQueryData:
				return "fqd";
			case kCSV:
				return "csv";
			case kNetCDF:
			case kNetCDFv4:
				return "nc";
			case kGeoTIFF:
				return "tif";
			default:
				return "unknown";
		}
	};

	auto DefaultFormat = [](Component k) -> string
	{
		switch (k)
		{
			case Component::kAnalysisTime:
			case Component::kForecastTime:
				return "%Y%m%d%H%M";
			case Component::kStep:
			case Component::kAggregationDuration:
				return "%Hh%Mm";
			case Component::kProcessingTypeValue:
			case Component::kProcessingTypeValue2:
			case Component::kLevelValue:
			case Component::kLevelValue2:
			case Component::kForecastTypeId:
			case Component::kForecastTypeValue:
			case Component::kProducerId:
			case Component::kGridNi:
			case Component::kGridNj:
				return "%d";
			case Component::kWallTime:
				return "%Y%m%d%H%M%S";
			default:
				return "";
		}
	};

	auto GetMasalaBase = [](HPProgramName name) -> string
	{
		switch (name)
		{
			case kHiman:
				return util::GetEnv("MASALA_PROCESSED_DATA_BASE");
			case kGridToRadon:
				try
				{
					return util::GetEnv("MASALA_REF_BASE");
				}
				catch (const invalid_argument& e)
				{
					try
					{
						return util::GetEnv("NEONS_REF_BASE");
					}
					catch (const invalid_argument& ee)
					{
						throw invalid_argument(
						    "Neither 'MASALA_REF_BASE' nor 'NEONS_REF_BASE' environment variable defined");
					}
				}
			default:
				return "";
		}
	};

	auto ReplaceTemplateValue = [&](const boost::regex& re, string& filename, Component k)
	{
		boost::smatch what;

		while (boost::regex_search(filename, what, re))
		{
			string fmt = DefaultFormat(k);

			if (what.size() == 3 && string(what[2]).empty() == false)
			{
				fmt = string(what[2]);
				fmt.erase(fmt.begin());  // remove starting ':'
			}

			string replacement;

			switch (k)
			{
				case Component::kMasalaBase:
					replacement = GetMasalaBase(conf.ProgramName());
					break;
				case Component::kAnalysisTime:
					replacement = ftime.OriginDateTime().String(fmt);
					break;
				case Component::kForecastTime:
					replacement = ftime.ValidDateTime().String(fmt);
					break;
				case Component::kStep:
					replacement = ftime.Step().String(fmt);
					break;
				case Component::kGeometryName:
					replacement = conf.TargetGeomName();
					break;
				case Component::kGridName:
					replacement = HPGridTypeToString.at(info.Grid()->Type());
					break;
				case Component::kGridNi:
				{
					if (info.Grid()->Class() == kRegularGrid)
					{
						replacement =
						    (boost::format(fmt) % (dynamic_pointer_cast<regular_grid>(info.Grid())->Ni())).str();
					}
					break;
				}
				case Component::kGridNj:
				{
					if (info.Grid()->Class() == kRegularGrid)
					{
						replacement =
						    (boost::format(fmt) % (dynamic_pointer_cast<regular_grid>(info.Grid())->Nj())).str();
					}
					break;
				}
				case Component::kParamName:
					replacement = par.Name();
					break;
				case Component::kAggregationName:
					replacement = HPAggregationTypeToString.at(par.Aggregation().Type());
					break;
				case Component::kAggregationDuration:
					replacement = par.Aggregation().TimeDuration().String(fmt);
					break;
				case Component::kProcessingTypeName:
					replacement = HPProcessingTypeToString.at(par.ProcessingType().Type());
					break;
				case Component::kProcessingTypeValue:
					replacement = (boost::format(fmt) % par.ProcessingType().Value()).str();
					break;
				case Component::kProcessingTypeValue2:
					replacement = (boost::format(fmt) % par.ProcessingType().Value2()).str();
					break;
				case Component::kLevelName:
					replacement = HPLevelTypeToString.at(lvl.Type());
					break;
				case Component::kLevelValue:
					replacement = (boost::format(fmt) % lvl.Value()).str();
					break;
				case Component::kLevelValue2:
					replacement = (boost::format(fmt) % lvl.Value2()).str();
					break;
				case Component::kForecastTypeId:
					replacement = (boost::format(fmt) % ftype.Type()).str();
					break;
				case Component::kForecastTypeName:
					replacement = ForecastTypeToShortString(ftype.Type());
					break;
				case Component::kForecastTypeValue:
					replacement = (boost::format(fmt) % ftype.Value()).str();
					break;
				case Component::kProducerId:
					replacement = (boost::format(fmt) % prod.Id()).str();
					break;
				case Component::kFileType:
					replacement = FileTypeToShortString(conf.OutputFileType());
					break;
				case Component::kWallTime:
					replacement = raw_time::Now().String(fmt);
					break;
				default:
					break;
			}

			filename = boost::regex_replace(filename, re, replacement, boost::regex_constants::format_first_only);
		}
	};

	string filename = filenameTemplate;

	const static vector<pair<Component, string>> regexs{
	    make_pair(Component::kMasalaBase, R"(\{(masala_base)\})"),
	    make_pair(Component::kAnalysisTime, R"(\{(analysis_time)(:[%a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kForecastTime, R"(\{(forecast_time)(:[%a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kStep, R"(\{(step)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kGeometryName, R"(\{(geom_name)\})"),
	    make_pair(Component::kGridName, R"(\{(grid_name)\})"),
	    make_pair(Component::kGridNi, R"(\{(grid_ni)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kGridNj, R"(\{(grid_nj)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kParamName, R"(\{(param_name)\})"),
	    make_pair(Component::kAggregationName, R"(\{(aggregation_name)\})"),
	    make_pair(Component::kAggregationDuration, R"(\{(aggregation_duration)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kProcessingTypeName, R"(\{(processing_type_name)\})"),
	    make_pair(Component::kProcessingTypeValue, R"(\{(processing_type_value)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kProcessingTypeValue2, R"(\{(processing_type_value2)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kLevelName, R"(\{(level_name)\})"),
	    make_pair(Component::kLevelValue, R"(\{(level_value)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kLevelValue2, R"(\{(level_value2)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kForecastTypeId, R"(\{(forecast_type_id)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kForecastTypeName, R"(\{(forecast_type_name)\})"),
	    make_pair(Component::kForecastTypeValue, R"(\{(forecast_type_value)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kProducerId, R"(\{(producer_id)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kFileType, R"(\{(file_type)\})"),
	    make_pair(Component::kWallTime, R"(\{(wall_time)(:[%a-zA-Z_/-]*)*\})")};

	for (const auto& p : regexs)
	{
		ReplaceTemplateValue(boost::regex(p.second), filename, p.first);
	}

	return filename;
}

template <typename T>
string DetermineDefaultFileName(const info<T>& info, const plugin_configuration& conf)
{
	// ostringstream fileName;
	stringstream ss;

	const auto ftype = info.template Value<forecast_type>();
	const auto ftime = info.template Value<forecast_time>();
	const auto lvl = info.template Value<level>();
	const auto par = info.template Value<param>();

	// For database writes get base directory

	logger logr("util::MakeFileName");

	if (conf.WriteToDatabase())
	{
		// directory structure when writing to database:
		//
		// all:
		//   /path/to/files/<producer_id>/<analysis_time>/<geometry_name>
		// few:
		//   /path/to/files/<producer_id>/<analysis_time>/<geometry_name>
		// single:
		//   /path/to/files/<producer_id>/<analysis_time>/<geometry_name>/<step>

		ss << "{masala_base}/{producer_id}/{analysis_time:%Y%m%d%H%M}/{geom_name}/";

		if (conf.WriteMode() == kSingleGridToAFile)
		{
			if (ftime.Step().Minutes() % 60 != 0)
			{
				ss << "{step:%m}/";
			}
			else
			{
				ss << "{step:%h}/";
			}
		}
	}

	if (conf.WriteMode() == kSingleGridToAFile)
	{
		ss << "{param_name}_{level_name}_{level_value}";

		if (!IsKHPMissingValue(lvl.Value2()))
		{
			ss << "-{level_value2}";
		}

		ss << "_{grid_name}";

		if (info.Grid()->Class() == kRegularGrid)
		{
			ss << "_{grid_ni}_{grid_nj}";
		}

		const auto step = ftime.Step();

		ss << "_0_";

		if ((step.Minutes() - step.Hours() * 60) != 0)
		{
			ss << "{step:%03hh%02Mmin}";
		}
		else
		{
			ss << "{step:%03h}";
		}

		if (static_cast<int>(ftype.Type()) > 2)  // backwards compatibility
		{
			ss << "_{forecast_type_id}";

			if (ftype.Type() == kEpsControl || ftype.Type() == kEpsPerturbation)
			{
				ss << "_{forecast_type_value}";
			}
		}

		ss << ".{file_type}";
	}
	else if (conf.WriteMode() == kFewGridsToAFile)
	{
		ss << "fc{analysis_time:%Y%m%d%H%M}+{step:%03hh%02Mm}_" << conf.Name() << "#" << conf.RelativeOrdinalNumber()
		   << ".{file_type}";
	}
	else if (conf.WriteMode() == kAllGridsToAFile)
	{
		if (conf.LegacyWriteMode())
		{
			// legacy mode for 'all grids to a file' does not support database
			// --> no need for 'base'

			if (conf.ConfigurationFileName() == "-")
			{
				std::cerr << "Legacy write mode not supported when reading configuration from stdin" << std::endl;
			}

			ss.str("");
			ss << conf.ConfigurationFileName() << ".{file_type}";
		}
		else
		{
			ss << "fc{analysis_time:%Y%m%d%H%M}+{step:%03hh%02Mm}.{file_type}";
		}
	}

	return ss.str();
}
}  // namespace

template <typename T>
string util::MakeFileName(const info<T>& info, const plugin_configuration& conf)
{
	if (conf.WriteMode() == kNoFileWrite)
	{
		return "";
	}

	string filenameTemplate = conf.FilenameTemplate();

	if (filenameTemplate.empty())
	{
		filenameTemplate = DetermineDefaultFileName(info, conf);
	}

	return MakeFileNameFromTemplate(info, conf, filenameTemplate);
}

template string util::MakeFileName<double>(const info<double>&, const plugin_configuration&);
template string util::MakeFileName<float>(const info<float>&, const plugin_configuration&);
template string util::MakeFileName<short>(const info<short>&, const plugin_configuration&);
template string util::MakeFileName<unsigned char>(const info<unsigned char>&, const plugin_configuration&);

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

	ifstream f(theFile.c_str(), ios::in | ios::binary);

	long keywordLength = 4;

	char content[keywordLength];

	f.read(content, keywordLength);

	HPFileType ret = kUnknownFile;

	static const char* grib = "GRIB";
	static const unsigned char ncv3[4] = {0x43, 0x44, 0x46, 0x01};
	static const unsigned char ncv4[4] = {0xD3, 0x48, 0x44, 0x46};  // 211 H D F
	static const char* tiff = "II*\0";

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

	vector<int> ret;
	const auto split1 = Split(identifier, ",");
	for (const auto& tok : split1)
	{
		const auto split2 = Split<int>(tok, "-");

		if (split2.size() == 2)
		{
			int a = split2[0], b = split2[1], step = 1;
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
		auto elems = util::Split(line, ",");

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
		auto timeparts = Split(elems[10], ":");

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
		auto elems = util::Split(line, ",");

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

		auto timeparts = Split(elems[10], ":");

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
template shared_ptr<info<short>> util::CSVToInfo<short>(const vector<string>&);
template shared_ptr<info<unsigned char>> util::CSVToInfo<unsigned char>(const vector<string>&);

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
			return std::move(g);
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
		return fmt::format("{}_{}_{}_{}_{}_{}_{}", options.prod.Id(), options.time.OriginDateTime().ToSQLTime(),
		                   options.time.ValidDateTime().ToSQLTime(), options.param.Name(),
		                   static_cast<string>(options.level), static_cast<int>(options.ftype.Type()),
		                   options.ftype.Value());
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
		return fmt::format("{}_{}_{}_{}_{}_{}_{}", info.Producer().Id(), info.Time().OriginDateTime().ToSQLTime(),
		                   info.Time().ValidDateTime().ToSQLTime(), info.Param().Name(),
		                   static_cast<string>(info.Level()), static_cast<int>(info.ForecastType().Type()),
		                   info.ForecastType().Value());
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
	if (name == "RRR-KGM2")
	{
		return himan::aggregation(kAccumulation, ONE_HOUR);
	}
	else if (name.find("RR-") != string::npos)
	{
		const auto tokens = util::Split(name, "-");

		if (tokens.size() == 2 && tokens[1] == "KGM2")
		{
			// RR-KGM2
			return himan::aggregation(kAccumulation, ftime.Step());
		}

		if (tokens[0] != "RR")
		{
			// PROB-RR-1 does not refer to 1-hour precipitation probability
			return himan::aggregation();
		}
		try
		{
			return himan::aggregation(kAccumulation, ONE_HOUR * stoi(tokens[1]));
		}
		catch (const std::exception& e)
		{
			// de nada
		}
	}
	else if (name.find("-MAX-") != string::npos)
	{
		return himan::aggregation(kMaximum);
	}
	else if (name.find("-MIN-") != string::npos)
	{
		return himan::aggregation(kMinimum);
	}
	else if (name.find("-MEAN-") != string::npos)
	{
		return himan::aggregation(kAverage);
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

param util::GetParameterInfoFromDatabaseName(const producer& prod, const param& par, const level& lvl)
{
	logger logr("util");
	auto r = GET_PLUGIN(radon);

	auto levelInfo = r->RadonDB().GetLevelFromDatabaseName(boost::to_upper_copy(HPLevelTypeToString.at(lvl.Type())));

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

	param p(paraminfo);

	// database does not provide aggregation or processing type information,
	// but we can guess, unless the calling code has already passed an aggregation
	//
	// todo: figure out a better way to deal with parametres that *always* have
	// aggregation and/or processing type

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
	if (par.Aggregation().Type() == kUnknownAggregationType && p.ProcessingType().Type() != kEnsembleMean)
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

				ens = make_unique<ensemble>(param(paramName), members, maximumMissing);
			}
			else
			{
				ens = make_unique<ensemble>(param(paramName), ensSize, maximumMissing);
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

			ens = make_unique<time_ensemble>(param(paramName), ensSize, kYearResolution, secondaryLen, secondaryStep,
			                                 secondarySpan, maximumMissing);
		}
		break;
		case kLaggedEnsemble:
		{
			const auto name = conf->GetValue("named_ensemble");

			if (name.empty() == false)
			{
				ens = make_unique<lagged_ensemble>(param(paramName), name, maximumMissing);
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

				ens = std::make_unique<lagged_ensemble>(param(paramName), forecasts, maximumMissing);
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

				ens = make_unique<lagged_ensemble>(param(paramName), ensSize, time_duration(kHourResolution, lag),
				                                   steps, maximumMissing);
			}
		}
		break;
		default:
			log.Fatal(fmt::format("Unknown ensemble type: {}", ensType));
			himan::Abort();
	}

	ASSERT(ens);

	log.Trace(fmt::format("Created ensemble of type: {}", HPEnsembleTypeToString.at(ens->EnsembleType())));

	return std::move(ens);
}

std::pair<long, long> util::GetScaledValue(double v)
{
	// Scale a float value so it can be encoded as a long
	// For example, value=273.15 --> scaled_value=27315, scale_factor=-2
	//              value=100    --> scaled_value=100, scale_factor=0

	const double r = floor(std::fmod(v, 1.0));
	std::pair<long, long> s(static_cast<long>(v), 0l);

	if (r != v || v < 1.0)
	{
		// value has decimals
		// - convert to string
		// - remove all trailing zeros
		// - count number of digits
		// --> that will be the scale factor
		auto str = fmt::format("{}", v);
		str.erase(str.find_last_not_of('0') + 1, std::string::npos);
		const auto dot = str.find('.');
		size_t num_digits = (dot == string::npos) ? 0 : str.length() - dot - 1;
		num_digits = std::min(num_digits, static_cast<size_t>(8));  // prevent neverending floats to inflate digitcount
		const double scaled = ::round(
		    v * pow(10., static_cast<double>(num_digits)));  // round before conversion to long,
		                                                     // because f.ex. 273.15 * pow(10, 2) = 27314.999999999996

		s = make_pair(static_cast<long>(scaled), -num_digits);
	}

	return s;
}
