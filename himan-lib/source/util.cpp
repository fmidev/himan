/**
 * @file util.cpp
 *
 * @brief Different utility functions and classes in a namespace
 */

#include "util.h"
#include "cuda_helper.h"
#include "forecast_time.h"
#include "lambert_conformal_grid.h"
#include "lambert_equal_area_grid.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "param.h"
#include "plugin_factory.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/regex.hpp>
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
		kFileType
	};

	auto ForecastTypeToShortString = [](HPForecastType type) -> string {
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

	auto FileTypeToShortString = [](HPFileType type) -> string {
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
			default:
				return "unknown";
		}
	};

	auto DefaultFormat = [](Component k) -> string {
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
			default:
				return "";
		}
	};
	auto ReplaceTemplateValue = [&](const boost::regex& re, string& filename, Component k) {
		boost::smatch what;

		if (boost::regex_search(filename, what, re) == false)
		{
			return;
		}

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
				replacement = util::GetEnv("MASALA_PROCESSED_DATA_BASE");
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
					replacement = (boost::format(fmt) % (dynamic_pointer_cast<regular_grid>(info.Grid())->Ni())).str();
				}
				break;
			}
			case Component::kGridNj:
			{
				if (info.Grid()->Class() == kRegularGrid)
				{
					replacement = (boost::format(fmt) % (dynamic_pointer_cast<regular_grid>(info.Grid())->Nj())).str();
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
			default:
				break;
		}

		filename = boost::regex_replace(filename, re, replacement);
	};

	string filename = filenameTemplate;

	const static vector<pair<Component, string>> regexs{
	    make_pair(Component::kMasalaBase, R"(\{(masala_base)\})"),
	    make_pair(Component::kAnalysisTime, R"(\{(analysis_time)(:[%a-zA-Z_]*)*\})"),
	    make_pair(Component::kForecastTime, R"(\{(forecast_time)(:[%a-zA-Z_]*)*\})"),
	    make_pair(Component::kStep, R"(\{(step)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kGeometryName, R"(\{(geom_name)\})"),
	    make_pair(Component::kGridName, R"(\{(grid_name)\})"),
	    make_pair(Component::kGridNi, R"(\{(grid_ni)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kGridNj, R"(\{(grid_nj)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kParamName, R"(\{(param_name)\})"),
	    make_pair(Component::kAggregationName, R"(\{(aggregation_name)\})"),
	    make_pair(Component::kAggregationDuration, R"(\{(aggregation_duration)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kProcessingTypeName, R"(\{(processing_type_name)\})"),
	    make_pair(Component::kProcessingTypeValue, R"(\{(processing_type_value)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kProcessingTypeValue2, R"(\{(processing_type_value2)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kLevelName, R"(\{(level_name)\})"),
	    make_pair(Component::kLevelValue, R"(\{(level_value)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kLevelValue2, R"(\{(level_value2)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kForecastTypeId, R"(\{(forecast_type_id)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kForecastTypeName, R"(\{(forecast_type_name)\})"),
	    make_pair(Component::kForecastTypeValue, R"(\{(forecast_type_value)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kProducerId, R"(\{(producer_id)(:[\.%0-9a-zA-Z]*)*\})"),
	    make_pair(Component::kFileType, R"(\{(file_type)\})")};

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
		ss << "/fc{analysis_time:%Y%m%d%H%M}+{step:%03hh%02Mm}_" << conf.Name() << "#" << conf.RelativeOrdinalNumber()
		   << ".{file_type}";
	}
	else if (conf.WriteMode() == kAllGridsToAFile)
	{
		if (conf.LegacyWriteMode())
		{
			// legacy mode for 'all grids to a file' does not support database
			// --> no need for 'base'

			ss.str("");
			ss << conf.ConfigurationFile() << ".{file_type}";
		}
		else
		{
			ss << "/fc{analysis_time:%Y%m%d%H%M}+{step:%03hh%02Mm}.{file_type}";
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

		const auto pck = std::dynamic_pointer_cast<simple_packed>(pdata);
		NFmiGribPacking::simple_packing::Unpack<T>(arr, pck->data, pck->bitmap, pck->unpackedLength, pck->packedLength,
		                                           pck->coefficients, stream);

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

	auto r = GET_PLUGIN(radon);

	auto geominfo = r->RadonDB().GetGeometryDefinition(geom_name);

	if (geominfo.empty())
	{
		throw invalid_argument(geom_name + " not found from database.");
	}

	const auto scmode = HPScanningModeFromString.at(geominfo["scanning_mode"]);

	// Until shape of earth is added to radon, hard code default value for all geoms
	// in radon to sphere with radius 6371220, which is the one used in newbase
	// (in most cases that's not *not* the one used by the weather model).
	// Exception to this lambert conformal conic where we use radius 6367470.

	if (geominfo["grid_type_id"] == "1")
	{
		// clang-format off
		return unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(
		    scmode,
		    point(stod(geominfo["long_orig"]), stod(geominfo["lat_orig"])),
		    stol(geominfo["col_cnt"]),
		    stol(geominfo["row_cnt"]),
		    stod(geominfo["pas_longitude"]),
		    stod(geominfo["pas_latitude"]),
		    earth_shape<double>(6371220.)
		));
		// clang-format on
	}
	else if (geominfo["grid_type_id"] == "4")
	{
		// clang-format off
		return unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid(
		    scmode,
		    point(stod(geominfo["long_orig"]), stod(geominfo["lat_orig"])),
		    stol(geominfo["col_cnt"]),
		    stol(geominfo["row_cnt"]),
		    stod(geominfo["pas_longitude"]),
		    stod(geominfo["pas_latitude"]),
		    earth_shape<double>(6371220.),
		    point(stod(geominfo["geom_parm_2"]), stod(geominfo["geom_parm_1"])), true)
		);
		// clang-format on
	}
	else if (geominfo["grid_type_id"] == "2")
	{
		// clang-format off
		return unique_ptr<stereographic_grid>(new stereographic_grid(
		    scmode,
		    point(stod(geominfo["long_orig"]), stod(geominfo["lat_orig"])),
		    stol(geominfo["col_cnt"]),
		    stol(geominfo["row_cnt"]),
		    stod(geominfo["pas_longitude"]),
		    stod(geominfo["pas_latitude"]),
		    stod(geominfo["geom_parm_1"]),
		    earth_shape<double>(6371220.),
		    false
		));
		// clang-format on
	}

	else if (geominfo["grid_type_id"] == "6")
	{
		auto g = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
		g->EarthShape(earth_shape<double>(6371220.));

		reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(g.get());

		gg->N(stoi(geominfo["n"]));

		auto strlongitudes = himan::util::Split(geominfo["longitudes_along_parallels"], ",", false);
		vector<int> longitudes;

		for (auto& l : strlongitudes)
		{
			longitudes.push_back(stoi(l));
		}

		gg->NumberOfPointsAlongParallels(longitudes);
		return std::move(g);
	}

	else if (geominfo["grid_type_id"] == "5")
	{
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
		    earth_shape<double>(6367470.),
		    false
		));
		// clang-format on
	}

	else if (geominfo["grid_type_id"] == "7")
	{
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
		    earth_shape<double>(6371220.),
		    false
		));
		// clang-format on
	}

	else
	{
		throw invalid_argument("Invalid grid type id for geometry " + geom_name);
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

string util::UniqueName(const plugin::search_options& options)
{
	ASSERT(options.configuration->DatabaseType() == kNoDatabase || options.prod.Id() != kHPMissingInt);
	stringstream ss;

	// clang-format off

        ss << options.prod.Id() << "_"
           << options.time.OriginDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
           << options.time.ValidDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
           << options.param.Name() << "_"
           << static_cast<string>(options.level) << "_"
           << options.ftype.Type() << "_"
           << options.ftype.Value();

	// clang-format on

	return ss.str();
}

template <typename T>
string util::UniqueName(const info<T>& info)
{
	stringstream ss;

	// clang-format off

        ss << info.Producer().Id() << "_"
           << info.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
           << info.Time().ValidDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
           << info.Param().Name() << "_"
           << static_cast<string>(info.Level()) << "_"
           << info.ForecastType().Type() << "_"
           << info.ForecastType().Value();

	// clang-format on

	return ss.str();
}

template string util::UniqueName(const info<double>&);
template string util::UniqueName(const info<float>&);
