#include "filename.h"
#include "forecast_time.h"
#include "level.h"
#include "param.h"
#include "util.h"
#include <fmt/printf.h>
#include <iomanip>
#include <regex>
#include <sstream>

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
	// {processing_type_value2:FORMAT_SPECIFIER}- second possible processing type value
	// {level_name }                           - level name
	// {level_value:FORMAT_SPECIFIER}          - level value
	// {level_value2:FORMAT_SPECIFIER}         - second possible level value
	// {forecast_type_name}                    - forecast type name, like 'sp' or 'det' (short forms used)
	// {forecast_type_id:FORMAT_SPECIFIER}     - forecast type id, 1 .. 5
	// {forecast_type_value:FORMAT_SPECIFIER}  - possible forecast type value
	// {producer_id}                           - radon producer id
	// {file_type}                             - file type extension, like grib, grib2, fqd, ...
	// {wall_time:FORMAT_SPECIFIER}	           - current wall clock time
	// {masala_base}                           - environment variable MASALA_PROCESSED_DATA_BASE or
	//                                           MASALA_REF_BASE, depending on program name
	// {env:ENV_VARIABLE}                      - environment variable ENV_VARIABLE

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
		kWallTime,
		kEnvironmentVariable
	};

	auto ComponentToString = [](Component c) -> string
	{
		switch (c)
		{
			case Component::kMasalaBase:
				return "masala_base";
			case Component::kAnalysisTime:
				return "analysis_time";
			case Component::kForecastTime:
				return "forecast_time";
			case Component::kStep:
				return "step";
			case Component::kGeometryName:
				return "geom_name";
			case Component::kGridName:
				return "grid_name";
			case Component::kGridNi:
				return "ni";
			case Component::kGridNj:
				return "nj";
			case Component::kParamName:
				return "param_name";
			case Component::kAggregationName:
				return "aggregation_name";
			case Component::kAggregationDuration:
				return "aggregation_duration";
			case Component::kProcessingTypeName:
				return "processing_type_name";
			case Component::kProcessingTypeValue:
				return "processing_type_value";
			case Component::kProcessingTypeValue2:
				return "processing_type_value2";
			case Component::kLevelName:
				return "level_name";
			case Component::kLevelValue:
				return "level_value";
			case Component::kLevelValue2:
				return "level_value2";
			case Component::kForecastTypeId:
				return "forecast_type_id";
			case Component::kForecastTypeValue:
				return "forecast_type_value";
			case Component::kForecastTypeName:
				return "forecast_type_name";
			case Component::kFileType:
				return "file_type";
			case Component::kProducerId:
				return "producer_id";
			case Component::kWallTime:
				return "wall_time";
			case Component::kEnvironmentVariable:
				return "env";
			default:
				return "unknown";
		}
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
			case Component::kForecastTypeValue:
				return "%.0f";
			case Component::kForecastTypeId:
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
					return util::GetEnv("MASALA_RAW_DATA_BASE");
				}
				catch (const invalid_argument& e)
				{
					try
					{
						return util::GetEnv("RADON_REF_BASE");
					}
					catch (const invalid_argument& ee)
					{
						throw invalid_argument(
						    "Neither 'MASALA_RAW_DATA_BASE' nor 'RADON_REF_BASE' environment variable defined");
					}
				}
			default:
				return "";
		}
	};

	auto ReplaceTemplateValue = [&](const std::regex& re, string& filename, Component k)
	{
		std::smatch what;

		while (std::regex_search(filename, what, re))
		{
			string mask = DefaultFormat(k);

			if (what.size() == 3 && string(what[2]).empty() == false)
			{
				mask = string(what[2]);
				mask.erase(mask.begin());  // remove starting ':'
			}

			string replacement;
			try
			{
				switch (k)
				{
					case Component::kMasalaBase:
						replacement = GetMasalaBase(conf.ProgramName());
						break;
					case Component::kAnalysisTime:
						replacement = ftime.OriginDateTime().String(mask);
						break;
					case Component::kForecastTime:
						replacement = ftime.ValidDateTime().String(mask);
						break;
					case Component::kStep:
						replacement = ftime.Step().String(mask);
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
							replacement = fmt::sprintf(mask, dynamic_pointer_cast<regular_grid>(info.Grid())->Ni());
						}
						break;
					}
					case Component::kGridNj:
					{
						if (info.Grid()->Class() == kRegularGrid)
						{
							replacement = fmt::sprintf(mask, dynamic_pointer_cast<regular_grid>(info.Grid())->Nj());
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
						replacement = par.Aggregation().TimeDuration().String(mask);
						break;
					case Component::kProcessingTypeName:
						replacement = HPProcessingTypeToString.at(par.ProcessingType().Type());
						break;
					case Component::kProcessingTypeValue:
						replacement = fmt::sprintf(mask, par.ProcessingType().Value());
						break;
					case Component::kProcessingTypeValue2:
						replacement = fmt::sprintf(mask, par.ProcessingType().Value2());
						break;
					case Component::kLevelName:
						replacement = HPLevelTypeToString.at(lvl.Type());
						break;
					case Component::kLevelValue:
						replacement = fmt::sprintf(mask, lvl.Value());
						break;
					case Component::kLevelValue2:
						replacement = fmt::sprintf(mask, lvl.Value2());
						break;
					case Component::kForecastTypeId:
						replacement = fmt::sprintf(mask, fmt::underlying(ftype.Type()));
						break;
					case Component::kForecastTypeName:
						replacement = ForecastTypeToShortString(ftype.Type());
						break;
					case Component::kForecastTypeValue:
						replacement = fmt::sprintf(mask, ftype.Value());
						break;
					case Component::kProducerId:
						replacement = fmt::sprintf(mask, prod.Id());
						break;
					case Component::kFileType:
						replacement = FileTypeToShortString(conf.OutputFileType());
						break;
					case Component::kWallTime:
						replacement = raw_time::Now().String(mask);
						break;
					case Component::kEnvironmentVariable:
					{
						const auto tokens = util::Split(string(what[1]), ":");

						if (tokens.size() != 2)
						{
							throw invalid_argument(fmt::format("Invalid environment variable mask: {}", mask));
						}

						auto val = util::GetEnv(tokens[1]);
						if (val.empty())
						{
							throw invalid_argument(fmt::format("Environment variable '{}' not defined", mask));
						}
						replacement = val;
						break;
					}
					default:
						break;
				}

				filename = std::regex_replace(filename, re, replacement, std::regex_constants::format_first_only);
			}
			catch (const std::exception& e)
			{
				auto errstr = fmt::format(
				    "{} mask: {} component: {}\nfmt-library enforces strict type checks, make sure formatting "
				    "specifier is for correct data type\nFor example for level_value change %03d to %03.0f",
				    e.what(), mask, ComponentToString(k));
				throw invalid_argument(errstr);
			}
		}
	};

	string filename = filenameTemplate;

	const static vector<pair<Component, string>> regexs{
	    make_pair(Component::kMasalaBase, R"(\{(masala_base)\})"),
	    make_pair(Component::kAnalysisTime, R"(\{(analysis_time)(:[:%a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kForecastTime, R"(\{(forecast_time)(:[:%a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kStep, R"(\{(step)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kGeometryName, R"(\{(geom_name)\})"),
	    make_pair(Component::kGridName, R"(\{(grid_name)\})"),
	    make_pair(Component::kGridNi, R"(\{(grid_ni)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kGridNj, R"(\{(grid_nj)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kParamName, R"(\{(param_name)\})"),
	    make_pair(Component::kAggregationName, R"(\{(aggregation_name)\})"),
	    make_pair(Component::kAggregationDuration, R"(\{(aggregation_duration)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kProcessingTypeName, R"(\{(processing_type_name)\})"),
	    make_pair(Component::kProcessingTypeValue, R"(\{(processing_type_value)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kProcessingTypeValue2, R"(\{(processing_type_value2)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kLevelName, R"(\{(level_name)\})"),
	    make_pair(Component::kLevelValue, R"(\{(level_value)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kLevelValue2, R"(\{(level_value2)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kForecastTypeId, R"(\{(forecast_type_id)(:[\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kForecastTypeName, R"(\{(forecast_type_name)\})"),
	    make_pair(Component::kForecastTypeValue, R"(\{(forecast_type_value)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kProducerId, R"(\{(producer_id)(:[:\.%0-9a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kFileType, R"(\{(file_type)\})"),
	    make_pair(Component::kWallTime, R"(\{(wall_time)(:[:%a-zA-Z_/-]*)*\})"),
	    make_pair(Component::kEnvironmentVariable, R"(\{(env:[a-zA-Z0-9_]+)\})")};

	for (const auto& p : regexs)
	{
		ReplaceTemplateValue(std::regex(p.second), filename, p.first);
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
string util::filename::MakeFileName(const info<T>& info, const plugin_configuration& conf)
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

template string util::filename::MakeFileName<double>(const info<double>&, const plugin_configuration&);
template string util::filename::MakeFileName<float>(const info<float>&, const plugin_configuration&);
template string util::filename::MakeFileName<short>(const info<short>&, const plugin_configuration&);
template string util::filename::MakeFileName<unsigned char>(const info<unsigned char>&, const plugin_configuration&);
