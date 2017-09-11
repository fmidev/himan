#include "luatool.h"
#include "ensemble.h"
#include "forecast_time.h"
#include "hitool.h"
#include "lagged_ensemble.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include "metutil.h"
#include "neons.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "radon.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#ifndef __clang_analyzer__

extern "C" {
#include <lualib.h>
}

#include <luabind/adopt_policy.hpp>
#include <luabind/iterator_policy.hpp>
#include <luabind/luabind.hpp>
#include <luabind/operator.hpp>

// void ScherlokHoms(info_t& myTargetInfo){std::cout<<myTargetInfo->Data().MissingValue() <<'\n';}

using namespace himan;
using namespace himan::plugin;
using namespace luabind;

#define LUA_MEMFN(r, t, m, ...) static_cast<r (t::*)(__VA_ARGS__)>(&t::m)
#define LUA_CMEMFN(r, t, m, ...) static_cast<r (t::*)(__VA_ARGS__) const>(&t::m)

void BindEnum(lua_State* L);
int BindErrorHandler(lua_State* L);
void BindPlugins(lua_State* L);
void BindLib(lua_State* L);

object VectorToTable(const std::vector<double>& vec);
std::vector<double> TableToVector(const object& table);

boost::thread_specific_ptr<lua_State> myL;

luatool::luatool() : itsWriteOptions()
{
	itsLogger = logger("luatool");
	myL.reset();
}

luatool::~luatool() {}
void luatool::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("DUMMY")});

	Start();
}

void luatool::Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("luatoolThread #" + std::to_string(threadIndex));

	InitLua(myTargetInfo);

	assert(myL.get());
	myThreadedLogger.Info("Calculating time " + static_cast<std::string>(myTargetInfo->Time().ValidDateTime()) +
						  " level " + static_cast<std::string>(myTargetInfo->Level()));

	globals(myL.get())["logger"] = myThreadedLogger;

	for (const std::string& luaFile : itsConfiguration->GetValueList("luafile"))
	{
		if (luaFile.empty())
		{
			continue;
		}

		myThreadedLogger.Info("Starting script " + luaFile);

		ReadFile(luaFile);
	}

	lua_close(myL.get());
	myL.release();
}

void luatool::InitLua(info_t myTargetInfo)
{
	/*
	 * An ideal solution would be to initialize the basic stuff once in a single threaded environment,
	 * and then just set the thread-specific stuff in the thread-specific section of code.
	 *
	 * Unfortunately lua does not support copying of the global lua_State* variable so this is not possible.
	 * So now we have to re-bind all functions for every thread execution :-(
	 */

	lua_State* L = luaL_newstate();

	assert(L);

	luaL_openlibs(L);

	open(L);

	set_pcall_callback(&BindErrorHandler);

	BindEnum(L);
	BindLib(L);
	BindPlugins(L);

	// Set some variable that are needed in luatool calculations
	// but are too hard or complicated to create in the lua side

	globals(L)["luatool"] = boost::ref(*this);
	globals(L)["result"] = myTargetInfo;
	globals(L)["configuration"] = itsConfiguration;
	globals(L)["write_options"] = boost::ref(itsWriteOptions);

	// Useful variables
	globals(L)["current_time"] = forecast_time(myTargetInfo->Time());
	globals(L)["current_level"] = level(myTargetInfo->Level());
	globals(L)["current_forecast_type"] = forecast_type(myTargetInfo->ForecastType());
	globals(L)["missing"] = MissingDouble();

	globals(L)["kKelvin"] = constants::kKelvin;

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(forecast_time(myTargetInfo->Time()));
	h->ForecastType(forecast_type(myTargetInfo->ForecastType()));

	auto n = GET_PLUGIN(neons);
	auto r = GET_PLUGIN(radon);

	// Useful plugins
	globals(L)["hitool"] = h;
	globals(L)["neons"] = n;
	globals(L)["radon"] = r;

	itsLogger.Trace("luabind finished");
	myL.reset(L);
}

bool luatool::ReadFile(const std::string& luaFile)
{
	if (!boost::filesystem::exists(luaFile))
	{
		std::cerr << "Error: script " << luaFile << " does not exist\n";
		return false;
	}

	try
	{
		assert(myL.get());
		if (luaL_dofile(myL.get(), luaFile.c_str()))
		{
			itsLogger.Error(lua_tostring(myL.get(), -1));
			return false;
		}
	}
	catch (const error& e)
	{
		return false;
	}
	catch (const std::exception& e)
	{
		itsLogger.Error(e.what());
		return false;
	}

	return true;
}

int BindErrorHandler(lua_State* L)
{
	// log the error message
	luabind::object msg(luabind::from_stack(L, -1));
	std::ostringstream str;
	str << "lua> run-time error: " << msg;
	std::cout << str.str() << std::endl;

	// log the callstack
	std::string traceback = luabind::call_function<std::string>(luabind::globals(L)["debug"]["traceback"]);
	traceback = std::string("lua> ") + traceback;
	std::cout << traceback.c_str() << std::endl;

	// return unmodified error object
	return 1;
}

// clang-format off

void BindEnum(lua_State* L)
{
	module(L)
	    [class_<HPLevelType>("HPLevelType")
	         .enum_("constants")[
				 value("kUnknownLevel", kUnknownLevel),
				 value("kGround", kGround),
				 value("kTopOfAtmosphere", kTopOfAtmosphere),
				 value("kPressure", kPressure),
				 value("kMeanSea", kMeanSea),
				 value("kAltitude", kAltitude),
				 value("kHeight", kHeight),
				 value("kHybrid", kHybrid),
				 value("kGroundDepth", kGroundDepth),
				 value("kDepth", kDepth),
				 value("kEntireAtmosphere", kEntireAtmosphere),
				 value("kEntireOcean", kEntireOcean),
				 value("kMaximumThetaE", kMaximumThetaE),
				 value("kHeightLayer", kHeightLayer)],
	     class_<HPTimeResolution>("HPTimeResolution")
	         .enum_(
	             "constants")[
					 value("kUnknownTimeResolution", kUnknownTimeResolution),
					 value("kHourResolution", kHourResolution),
					 value("kMinuteResolution", kMinuteResolution)],
	     class_<HPFileType>("HPFileType")
	         .enum_("constants")[
				 value("kUnknownFile", kUnknownFile),
				 value("kGRIB1", kGRIB1),
				 value("kGRIB2", kGRIB2),
				 value("kGRIB", kGRIB),
				 value("kQueryData", kQueryData),
				 value("kNetCDF", kNetCDF)],
	     class_<HPScanningMode>("HPScanningMode")
	         .enum_("constants")[
				 value("kUnknownScanningMode", kUnknownScanningMode),
				 value("kTopLeft", himan::kTopLeft),
				 value("kTopRight", himan::kTopRight),
				 value("kBottomLeft", himan::kBottomLeft),
				 value("kBottomRight", himan::kBottomRight)],
	     class_<HPAggregationType>("HPAggregationType")
	         .enum_("constants")[
				 value("kUnknownAggregationType", kUnknownAggregationType),
				 value("kAverage", kAverage),
				 value("kAccumulation", kAccumulation),
				 value("kMaximum", kMaximum),
				 value("kMinimum", kMinimum),
				 value("kDifference", kDifference)],
	     class_<HPModifierType>("HPModifierType")
	         .enum_("constants")
	             [
		value("kUnknownModifierType", kUnknownModifierType),
				 value("kAverageModifier", kAverageModifier),
				 value("kAccumulationModifier", kAccumulationModifier),
				 value("kMaximumModifier", kMaximumModifier),
				 value("kMinimumModifier", kMinimumModifier),
				 value("kDifferenceModifier", kDifferenceModifier),
				 value("kMaximumMinimumModifier", kMaximumMinimumModifier),
				 value("kCountModifier", kCountModifier),
				 value("kFindHeightModifier", kFindHeightModifier),
				 value("kFindValueModifier", kFindValueModifier),
				 value("kIntegralModifier", kIntegralModifier),
				 value("kPlusMinusAreaModifier", kPlusMinusAreaModifier),
				 value("kFindHeightGreaterThanModifier", kFindHeightGreaterThanModifier),
				 value("kFindHeightLessThanModifier", kFindHeightLessThanModifier)],
	     class_<HPGridClass>("HPGridClass")
	         .enum_("constants")[
				 value("kUnknownGridClass", kUnknownGridClass),
				 value("kRegularGrid", kRegularGrid),
				 value("kIrregularGrid", kIrregularGrid)],
	     class_<HPGridType>("HPGridType")
	         .enum_(
	             "constants")[value("kUnknownGridType", kUnknownGridType),
	                          value("kLatitudeLongitude", kLatitudeLongitude),
							  value("kStereographic", kStereographic),
	                          value("kAzimuthalEquidistant", kAzimuthalEquidistant),
	                          value("kRotatedLatitudeLongitude", kRotatedLatitudeLongitude),
	                          value("kReducedGaussian", kReducedGaussian),
							  value("kPointList", kPointList)],
	     class_<HPParameterUnit>("HPParameterUnit").enum_("constants")[
			 value("kM", kM),
			 value("kHPa", kHPa)],
	     class_<HPForecastType>("HPForecastType")
	         .enum_("constants")[
				 value("kUnknownType", kUnknownType),
				 value("kDeterministic", kDeterministic),
				 value("kAnalysis", kAnalysis),
				 value("kEpsControl", kEpsControl),
				 value("kEpsPerturbation", kEpsPerturbation)]];
}

// clang-format on

namespace info_wrapper
{
// These are convenience functions for accessing info class contents

void SetValue(std::shared_ptr<info>& anInfo, int index, double value) { anInfo->Grid()->Value(--index, value); }
double GetValue(std::shared_ptr<info>& anInfo, int index) { return anInfo->Grid()->Value(--index); }
size_t GetLocationIndex(std::shared_ptr<info> anInfo) { return anInfo->LocationIndex() + 1; }
size_t GetTimeIndex(std::shared_ptr<info> anInfo) { return anInfo->TimeIndex() + 1; }
size_t GetParamIndex(std::shared_ptr<info> anInfo) { return anInfo->ParamIndex() + 1; }
size_t GetLevelIndex(std::shared_ptr<info> anInfo) { return anInfo->LevelIndex() + 1; }
void SetLocationIndex(std::shared_ptr<info> anInfo, size_t theIndex) { anInfo->LocationIndex(--theIndex); }
void SetTimeIndex(std::shared_ptr<info> anInfo, size_t theIndex) { anInfo->TimeIndex(--theIndex); }
void SetParamIndex(std::shared_ptr<info> anInfo, size_t theIndex) { anInfo->ParamIndex(--theIndex); }
void SetLevelIndex(std::shared_ptr<info> anInfo, size_t theIndex) { anInfo->LevelIndex(--theIndex); }
void SetValues(info_t& anInfo, const object& table)
{
	std::vector<double> vals = TableToVector(table);

	if (vals.empty())
	{
		return;
	}

	if (vals.size() != anInfo->Data().Size())
	{
		std::cerr << "Error::luatool input table size is not the same as grid size: " << vals.size() << " vs "
		          << anInfo->Data().Size() << std::endl;
	}
	else
	{
		anInfo->Data().Set(vals);
	}
}

object GetValues(info_t& anInfo) { return VectorToTable(VEC(anInfo)); }
point GetLatLon(info_t& anInfo, size_t theIndex) { return anInfo->Grid()->LatLon(--theIndex); }
double GetMissingValue(info_t& anInfo) { return anInfo->Data().MissingValue(); }
void SetMissingValue(info_t& anInfo, double missingValue) { anInfo->Data().MissingValue(missingValue); }
matrix<double> GetData(info_t& anInfo) { return anInfo->Data(); }
}  // namespace info_wrapper

namespace hitool_wrapper
{
// The following functions are all wrappers for hitool:
// we cannot specify hitool functions directly in the lua binding
// because that leads to undefined symbols in plugins and that
// forces us to link luatool with hitool which is not nice!

object VerticalMaximumGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                           const object& lastLevelValue)
{
	try
	{
		return VectorToTable(
		    h->VerticalMaximum(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalMaximum(std::shared_ptr<hitool> h, const param& theParam, double firstLevelValue, double lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalMaximum(theParam, firstLevelValue, lastLevelValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalMinimumGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                           const object& lastLevelValue)
{
	try
	{
		return VectorToTable(
		    h->VerticalMinimum(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalMinimum(std::shared_ptr<hitool> h, const param& theParam, double firstLevelValue, double lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalMinimum(theParam, firstLevelValue, lastLevelValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalSumGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                       const object& lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalSum(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalSum(std::shared_ptr<hitool> h, const param& theParam, double firstLevelValue, double lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalSum(theParam, firstLevelValue, lastLevelValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalAverageGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                           const object& lastLevelValue)
{
	try
	{
		return VectorToTable(
		    h->VerticalAverage(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalAverage(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue, double lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalAverage(theParams, firstLevelValue, lastLevelValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalCountGrid(std::shared_ptr<hitool> h, const param& theParams, const object& firstLevelValue,
                         const object& lastLevelValue, const object& findValue)
{
	try
	{
		return VectorToTable(h->VerticalCount(theParams, TableToVector(firstLevelValue), TableToVector(lastLevelValue),
		                                      TableToVector(findValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalCount(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue, double lastLevelValue,
                     double findValue)
{
	try
	{
		return VectorToTable(h->VerticalCount(theParams, firstLevelValue, lastLevelValue, findValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalHeightGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                          const object& lastLevelValue, const object& findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeight(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue),
		                                       TableToVector(findValue), findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalHeight(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue, double lastLevelValue,
                      double findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeight(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalHeightGreaterThanGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                                     const object& lastLevelValue, const object& findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeightGreaterThan(theParam, TableToVector(firstLevelValue),
		                                                  TableToVector(lastLevelValue), TableToVector(findValue),
		                                                  findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalHeightGreaterThan(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue,
                                 double lastLevelValue, double findValue, size_t findNth)
{
	try
	{
		return VectorToTable(
		    h->VerticalHeightGreaterThan(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalHeightLessThanGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue,
                                  const object& lastLevelValue, const object& findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeightLessThan(theParam, TableToVector(firstLevelValue),
		                                               TableToVector(lastLevelValue), TableToVector(findValue),
		                                               findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalHeightLessThan(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue,
                              double lastLevelValue, double findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeightLessThan(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalValueGrid(std::shared_ptr<hitool> h, const param& theParam, const object& findValue)
{
	try
	{
		return VectorToTable(h->VerticalValue(theParam, TableToVector(findValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalValue(std::shared_ptr<hitool> h, const param& theParam, double findValue)
{
	try
	{
		return VectorToTable(h->VerticalValue(theParam, findValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalPlusMinusAreaGrid(std::shared_ptr<hitool> h, const param& theParams, const object& firstLevelValue,
                                 const object& lastLevelValue)
{
	try
	{
		return VectorToTable(
		    h->PlusMinusArea(theParams, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

object VerticalPlusMinusArea(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue,
                             double lastLevelValue)
{
	try
	{
		return VectorToTable(h->PlusMinusArea(theParams, firstLevelValue, lastLevelValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	return object();
}

void Time(std::shared_ptr<hitool> h, const forecast_time& theTime) { h->Time(theTime); }
void SetHeightUnit(std::shared_ptr<hitool> h, HPParameterUnit theHeightUnit) { h->HeightUnit(theHeightUnit); }
HPParameterUnit GetHeightUnit(std::shared_ptr<hitool> h) { return h->HeightUnit(); }
}  // namespace hitool_wrapper

namespace modifier_wrapper
{
void SetLowerHeightGrid(modifier& mod, const object& lowerHeight) { mod.LowerHeight(TableToVector(lowerHeight)); }
object GetLowerHeightGrid(modifier& mod) { return VectorToTable(mod.LowerHeight()); }
void SetUpperHeightGrid(modifier& mod, const object& upperHeight) { mod.UpperHeight(TableToVector(upperHeight)); }
object GetUpperHeightGrid(modifier& mod) { return VectorToTable(mod.UpperHeight()); }
void SetFindValueGrid(modifier& mod, const object& findValue) { mod.FindValue(TableToVector(findValue)); }
object GetFindValueGrid(modifier& mod) { return VectorToTable(mod.FindValue()); }
object Result(modifier& mod) { return VectorToTable(mod.Result()); }
namespace findvalue
{
void Process(modifier_findvalue& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace findvalue

namespace findheight
{
void Process(modifier_findheight& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace findheight

namespace findheight_gt
{
void Process(modifier_findheight_gt& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace findheight_gt

namespace findheight_lt
{
void Process(modifier_findheight_lt& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace findheight_lt

namespace max
{
void Process(modifier_max& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace max

namespace min
{
void Process(modifier_min& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace min

namespace maxmin
{
void Process(modifier_maxmin& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace maxmin

namespace count
{
void Process(modifier_count& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace count

namespace mean
{
object Result(modifier_mean& mod) { return VectorToTable(mod.Result()); }
void Process(modifier_mean& mod, const object& data, const object& height)
{
	mod.Process(TableToVector(data), TableToVector(height));
}
}  // namespace mean

}  // namespace modifier_wrapper

namespace neons_wrapper
{
std::string GetProducerMetaData(std::shared_ptr<neons> n, const producer& prod, const std::string& attName)
{
	return n->ProducerMetaData(prod.Id(), attName);
}

}  // namespace neons_wrapper

namespace radon_wrapper
{
std::string GetProducerMetaData(std::shared_ptr<radon> r, const producer& prod, const std::string& attName)
{
	return r->RadonDB().GetProducerMetaData(prod.Id(), attName);
}

}  // namespace radon_wrapper

namespace ensemble_wrapper
{
object Values(const ensemble& ens) { return VectorToTable(ens.Values()); }
object SortedValues(const ensemble& ens) { return VectorToTable(ens.SortedValues()); }
}  // ensemble_wrapper

namespace matrix_wrapper
{
void SetValues(matrix<double>& mat, const object& values) { mat.Set(TableToVector(values)); }
object GetValues(matrix<double>& mat) { return VectorToTable(std::vector<double>(mat.Values())); }
}  // matrix_wrapper

// clang-format off

void BindLib(lua_State* L)
{
	module(L)[class_<himan::info, std::shared_ptr<himan::info>>("info")
	              .def(constructor<>())
	              .def("ClassName", &info::ClassName)
	              .def("First", &info::First)
	              .def("ResetParam", &info::ResetParam)
	              .def("FirstParam", &info::FirstParam)
	              .def("NextParam", &info::NextParam)
	              .def("ResetLevel", &info::ResetLevel)
	              .def("FirstLevel", &info::FirstLevel)
	              .def("NextLevel", &info::NextLevel)
	              .def("ResetTime", &info::ResetTime)
	              .def("FirstTime", &info::FirstTime)
	              .def("NextTime", &info::NextTime)
	              .def("SizeLocations", LUA_CMEMFN(size_t, info, SizeLocations, void))
	              .def("SizeTimes", LUA_CMEMFN(size_t, info, SizeTimes, void))
	              .def("SizeParams", LUA_CMEMFN(size_t, info, SizeParams, void))
	              .def("SizeLevels", LUA_CMEMFN(size_t, info, SizeLevels, void))
	              .def("GetLevel", LUA_CMEMFN(level, info, Level, void))
	              .def("GetTime", LUA_CMEMFN(forecast_time, info, Time, void))
	              .def("GetParam", LUA_CMEMFN(param, info, Param, void))
	              .def("GetGrid", LUA_CMEMFN(grid*, info, Grid, void))
	              .def("SetTime", LUA_MEMFN(void, info, SetTime, const forecast_time&))
	              .def("SetLevel", LUA_MEMFN(void, info, SetLevel, const level&))
	              .def("SetParam", LUA_MEMFN(void, info, SetParam, const param&))
	              // These are local functions to luatool
	              .def("SetIndexValue", &info_wrapper::SetValue)
	              .def("GetIndexValue", &info_wrapper::GetValue)
	              .def("GetTimeIndex", &info_wrapper::GetTimeIndex)
	              .def("GetParamIndex", &info_wrapper::GetParamIndex)
	              .def("GetLevelIndex", &info_wrapper::GetLevelIndex)
	              .def("SetTimeIndex", &info_wrapper::SetTimeIndex)
	              .def("SetParamIndex", &info_wrapper::SetParamIndex)
	              .def("SetLevelIndex", &info_wrapper::SetLevelIndex)
	              .def("SetValues", &info_wrapper::SetValues)
	              .def("GetValues", &info_wrapper::GetValues)
	              .def("GetLatLon", &info_wrapper::GetLatLon)
	              .def("GetMissingValue", &info_wrapper::GetMissingValue)
	              .def("SetMissingValue", &info_wrapper::SetMissingValue)
	              .def("GetData", &info_wrapper::GetData),
	          class_<grid, std::shared_ptr<grid>>("grid")
	              .def("ClassName", &grid::ClassName)
	              .def("GetScanningMode", LUA_CMEMFN(HPScanningMode, grid, ScanningMode, void))
	              .def("GetGridType", LUA_CMEMFN(HPGridType, grid, Type, void))
	              .def("GetGridClass", LUA_CMEMFN(HPGridClass, grid, Class, void))
	              .def("GetAB", LUA_CMEMFN(std::vector<double>, grid, AB, void))
	              .def("GetSize", &grid::Size),
	          class_<latitude_longitude_grid, grid, std::shared_ptr<latitude_longitude_grid>>("latitude_longitude_grid")
	              .def(constructor<>())
	              .def("ClassName", &latitude_longitude_grid::ClassName)
	              .def("GetNi", LUA_CMEMFN(size_t, latitude_longitude_grid, Ni, void))
	              .def("GetNj", LUA_CMEMFN(size_t, latitude_longitude_grid, Nj, void))
	              .def("GetDi", LUA_CMEMFN(double, latitude_longitude_grid, Di, void))
	              .def("GetDj", LUA_CMEMFN(double, latitude_longitude_grid, Dj, void))
	              .def("GetBottomLeft", LUA_CMEMFN(point, latitude_longitude_grid, BottomLeft, void))
	              .def("SetBottomLeft", LUA_MEMFN(void, latitude_longitude_grid, BottomLeft, const point&))
	              .def("GetTopRight", LUA_CMEMFN(point, latitude_longitude_grid, TopRight, void))
	              .def("SetTopRight", LUA_MEMFN(void, latitude_longitude_grid, BottomLeft, const point&))
	              .def("GetFirstPoint", LUA_CMEMFN(point, latitude_longitude_grid, FirstPoint, void))
	              .def("GetLastPoint", LUA_CMEMFN(point, latitude_longitude_grid, LastPoint, void)),
	          class_<rotated_latitude_longitude_grid, latitude_longitude_grid,
	                 std::shared_ptr<rotated_latitude_longitude_grid>>("rotated_latitude_longitude_grid")
	              .def(constructor<>())
	              .def("ClassName", &rotated_latitude_longitude_grid::ClassName)
	              .def("GetSouthPole", LUA_CMEMFN(point, rotated_latitude_longitude_grid, SouthPole, void))
	              .def("SetSouthPole", LUA_MEMFN(void, rotated_latitude_longitude_grid, SouthPole, const point&))
	              .def("GetUVRelativeToGrid", LUA_CMEMFN(bool, rotated_latitude_longitude_grid, UVRelativeToGrid, void))
	              .def("SetUVRelativeToGrid", LUA_MEMFN(void, rotated_latitude_longitude_grid, UVRelativeToGrid, bool)),
	          class_<stereographic_grid, grid, std::shared_ptr<stereographic_grid>>("stereographic_grid")
	              .def(constructor<>())
	              .def("ClassName", &stereographic_grid::ClassName)
	              .def("GetNi", LUA_CMEMFN(size_t, stereographic_grid, Ni, void))
	              .def("GetNj", LUA_CMEMFN(size_t, stereographic_grid, Nj, void))
	              .def("GetDi", LUA_CMEMFN(double, stereographic_grid, Di, void))
	              .def("GetDj", LUA_CMEMFN(double, stereographic_grid, Dj, void))
	              .def("GetBottomLeft", LUA_CMEMFN(point, stereographic_grid, BottomLeft, void))
	              .def("SetBottomLeft", LUA_MEMFN(void, stereographic_grid, BottomLeft, const point&))
	              .def("GetTopRight", LUA_CMEMFN(point, stereographic_grid, TopRight, void))
	              .def("SetTopRight", LUA_MEMFN(void, stereographic_grid, BottomLeft, const point&))
	              .def("GetFirstPoint", LUA_CMEMFN(point, stereographic_grid, FirstPoint, void))
	              .def("GetLastPoint", LUA_CMEMFN(point, stereographic_grid, LastPoint, void))
	              .def("GetOrientation", LUA_CMEMFN(double, stereographic_grid, Orientation, void))
	              .def("SetOrientation", LUA_MEMFN(void, stereographic_grid, Orientation, double)),
	          class_<reduced_gaussian_grid, grid, std::shared_ptr<reduced_gaussian_grid>>("reduced_gaussian_grid")
	              .def(constructor<>())
	              .def("ClassName", &reduced_gaussian_grid::ClassName)
	              .def("GetNj", LUA_CMEMFN(size_t, reduced_gaussian_grid, Nj, void))
	              .def("GetDj", LUA_CMEMFN(double, reduced_gaussian_grid, Dj, void))
	              .def("GetN", LUA_CMEMFN(int, reduced_gaussian_grid, N, void))
	              .def("SetN", LUA_MEMFN(void, reduced_gaussian_grid, N, int))
	              .def("GetBottomLeft", LUA_CMEMFN(point, reduced_gaussian_grid, BottomLeft, void))
	              .def("SetBottomLeft", LUA_MEMFN(void, reduced_gaussian_grid, BottomLeft, const point&))
	              .def("GetTopRight", LUA_CMEMFN(point, reduced_gaussian_grid, TopRight, void))
	              .def("SetTopRight", LUA_MEMFN(void, reduced_gaussian_grid, BottomLeft, const point&))
	              .def("GetFirstPoint", LUA_CMEMFN(point, reduced_gaussian_grid, FirstPoint, void))
	              .def("GetLastPoint", LUA_CMEMFN(point, reduced_gaussian_grid, LastPoint, void))
	          ,
	          class_<matrix<double>>("matrix")
	              .def(constructor<size_t, size_t, size_t, double>())
	              .def("SetValues", &matrix_wrapper::SetValues)
	              .def("GetValues", &matrix_wrapper::GetValues)
	          ,
	          class_<param>("param")
	              .def(constructor<const std::string&>())
	              .def("ClassName", &param::ClassName)
	              .def("GetName", LUA_CMEMFN(std::string, param, Name, void))
	              .def("SetName", LUA_MEMFN(void, param, Name, const std::string&))
	              .def("GetGrib2Number", LUA_CMEMFN(long, param, GribParameter, void))
	              .def("SetGrib2Number", LUA_MEMFN(void, param, GribParameter, long))
	              .def("GetGrib2Discipline", LUA_CMEMFN(long, param, GribDiscipline, void))
	              .def("SetGrib2Discipline", LUA_MEMFN(void, param, GribDiscipline, long))
	              .def("GetGrib2Category", LUA_CMEMFN(long, param, GribCategory, void))
	              .def("SetGrib2Category", LUA_MEMFN(void, param, GribCategory, long))
	              .def("GetGrib1Parameter", LUA_CMEMFN(long, param, GribIndicatorOfParameter, void))
	              .def("SetGrib1Parameter", LUA_MEMFN(void, param, GribIndicatorOfParameter, long))
	              .def("GetGrib1TableVersion", LUA_CMEMFN(long, param, GribTableVersion, void))
	              .def("SetGrib1TableVersion", LUA_MEMFN(void, param, GribTableVersion, long))
	              .def("GetUnivId", LUA_CMEMFN(unsigned long, param, UnivId, void))
	              .def("SetUnivId", LUA_MEMFN(void, param, UnivId, unsigned long))
	              .def("GetAggregation", LUA_CMEMFN(const aggregation&, param, Aggregation, void))
	              .def("SetAggregation", LUA_MEMFN(void, param, Aggregation, const aggregation&)),
	          class_<level>("level")
	              .def(constructor<HPLevelType, double>())
	              .def(constructor<HPLevelType, double, double>())
	              .def("ClassName", &level::ClassName)
	              .def(tostring(self))
	              .def("GetType", LUA_CMEMFN(HPLevelType, level, Type, void))
	              .def("SetType", LUA_MEMFN(void, level, Type, HPLevelType))
	              .def("GetValue", LUA_CMEMFN(double, level, Value, void))
	              .def("SetValue", LUA_MEMFN(void, level, Value, double)),
	          class_<raw_time>("raw_time")
	              .def(constructor<const std::string&>())
	              .def("ClassName", &raw_time::ClassName)
	              .def("String", LUA_CMEMFN(std::string, raw_time, String, const std::string&))
	              .def("Adjust", &raw_time::Adjust)
	              .def("Empty", &raw_time::Empty),
	          class_<forecast_time>("forecast_time")
	              .def(constructor<const raw_time&, const raw_time&>())
	              .def("ClassName", &forecast_time::ClassName)
	              .def("GetOriginDateTime", LUA_MEMFN(raw_time&, forecast_time, OriginDateTime, void))
	              .def("GetValidDateTime", LUA_MEMFN(raw_time&, forecast_time, ValidDateTime, void))
	              .def("SetOriginDateTime",
	                   LUA_MEMFN(void, forecast_time, OriginDateTime, const std::string&, const std::string&))
	              .def("SetValidDateTime",
	                   LUA_MEMFN(void, forecast_time, ValidDateTime, const std::string&, const std::string&))
	              .def("GetStep", LUA_CMEMFN(int, forecast_time, Step, void))
	              .def("GetStepResolution", LUA_CMEMFN(HPTimeResolution, forecast_time, StepResolution, void))
	              .def("SetStepResolution", LUA_MEMFN(void, forecast_time, StepResolution, HPTimeResolution)),
	          class_<forecast_type>("forecast_type")
	              .def(constructor<HPForecastType, double>())
	              .def("ClassName", &forecast_type::ClassName)
	              .def("GetType", LUA_CMEMFN(HPForecastType, forecast_type, Type, void))
	              .def("SetType", LUA_MEMFN(void, forecast_type, Type, HPForecastType))
	              .def("GetValue", LUA_CMEMFN(double, forecast_type, Value, void))
	              .def("SetValue", LUA_MEMFN(void, forecast_type, Value, double)),
	          class_<point>("point")
	              .def(constructor<double, double>())
	              .def("ClassName", &point::ClassName)
	              .def("SetX", LUA_MEMFN(void, point, X, double))
	              .def("SetY", LUA_MEMFN(void, point, Y, double))
	              .def("GetX", LUA_CMEMFN(double, point, X, void))
	              .def("GetY", LUA_CMEMFN(double, point, Y, void)),
	          class_<producer>("producer")
	              .def(constructor<>())
	              .def("ClassName", &producer::ClassName)
	              .def("SetName", LUA_MEMFN(void, producer, Name, const std::string&))
	              .def("GetName", LUA_CMEMFN(std::string, producer, Name, void))
	              .def("SetId", LUA_MEMFN(void, producer, Id, long))
	              .def("GetId", LUA_CMEMFN(long, producer, Id, void))
	              .def("SetProcess", LUA_MEMFN(void, producer, Process, long))
	              .def("GetProcess", LUA_CMEMFN(long, producer, Process, void))
	              .def("SetCentre", LUA_MEMFN(void, producer, Centre, long))
	              .def("GetCentre", LUA_CMEMFN(long, producer, Centre, void))
	          // TableVersion intentionally left out since in RADON it will be only
	          // a parameter property
	          ,
	          class_<logger>("logger")
	              .def(constructor<>())
	              .def("Trace", &logger::Trace)
	              .def("Debug", &logger::Debug)
	              .def("Info", &logger::Info)
	              .def("Warning", &logger::Warning)
	              .def("Error", &logger::Error)
	              .def("Fatal", &logger::Fatal),
	          class_<aggregation>("aggregation")
	              .def(constructor<HPAggregationType, HPTimeResolution, int>())
	              .def("ClassName", &aggregation::ClassName)
	              .def("GetType", LUA_CMEMFN(HPAggregationType, aggregation, Type, void))
	              .def("SetType", LUA_MEMFN(void, aggregation, Type, HPAggregationType))
	              .def("GetTimeResolution", LUA_CMEMFN(HPTimeResolution, aggregation, TimeResolution, void))
	              .def("SetTimeResolution", LUA_MEMFN(void, aggregation, TimeResolution, HPTimeResolution))
	              .def("GetTimeResolutionValue", LUA_CMEMFN(int, aggregation, TimeResolutionValue, void))
	              .def("SetTimeResolutionValue", LUA_MEMFN(void, aggregation, TimeResolutionValue, int)),
	          class_<configuration, std::shared_ptr<configuration>>("configuration")
	              .def(constructor<>())
	              .def("ClassName", &configuration::ClassName)
	              .def("GetOutputFileType", LUA_CMEMFN(HPFileType, configuration, OutputFileType, void))
	              .def("GetSourceProducer", LUA_CMEMFN(const producer&, configuration, SourceProducer, size_t))
	              .def("GetTargetProducer", LUA_CMEMFN(const producer&, configuration, TargetProducer, void))
	              .def("GetForecastStep", &configuration::ForecastStep)
	              ,
	          class_<plugin_configuration, configuration, std::shared_ptr<plugin_configuration>>("plugin_configuration")
	              .def(constructor<>())
	              .def("ClassName", &plugin_configuration::ClassName)
	              .def("GetValue", &plugin_configuration::GetValue)
	              .def("GetValueList", &plugin_configuration::GetValueList)
	              .def("Exists", &plugin_configuration::Exists),
	          class_<lcl_t>("lcl_t")
	              .def(constructor<>())
	              .def_readwrite("T", &lcl_t::T)
	              .def_readwrite("P", &lcl_t::P)
	              .def_readwrite("Q", &lcl_t::Q),
	          class_<write_options>("write_options")
	              .def(constructor<>())
	              .def_readwrite("use_bitmap", &write_options::use_bitmap),
		  class_<himan::ensemble, std::shared_ptr<himan::ensemble>>("ensemble")
		      .def(constructor<param, int>())
		      .def("ClassName", &ensemble::ClassName)
		      .def("Fetch", &ensemble::Fetch)
		      .def("Values", &ensemble_wrapper::Values)
		      .def("SortedValues", &ensemble_wrapper::SortedValues)
		      .def("ResetLocation", &ensemble::ResetLocation)
		      .def("FirstLocation", &ensemble::FirstLocation)
		      .def("NextLocation", &ensemble::NextLocation)
		      .def("Value", &ensemble::Value)
		      .def("Mean", &ensemble::Mean)
		      .def("Variance", &ensemble::Variance)
		      .def("CentralMoment", LUA_CMEMFN(double,ensemble,CentralMoment, int))
		      .def("Size", &ensemble::Size)
		      .def("ExpectedSize", &ensemble::ExpectedSize)
		      .def("SetMaximumMissingForecasts", LUA_MEMFN(void, ensemble, MaximumMissingForecasts, int))
		      .def("GetMaximumMissingForecasts", LUA_CMEMFN(int, ensemble, MaximumMissingForecasts, void)),
		  class_<himan::lagged_ensemble, std::shared_ptr<himan::lagged_ensemble>>("lagged_ensemble")
		      .def(constructor<param, size_t, HPTimeResolution, int, size_t>())
		      .def("ClassName", &lagged_ensemble::ClassName)
		      .def("Fetch", &lagged_ensemble::Fetch)
		      .def("Value", &lagged_ensemble::Value)
		      .def("Values", &ensemble_wrapper::Values)
		      .def("SortedValues", &ensemble_wrapper::SortedValues)
		      .def("ResetLocation", &lagged_ensemble::ResetLocation)
		      .def("FirstLocation", &lagged_ensemble::FirstLocation)
		      .def("NextLocation", &lagged_ensemble::NextLocation)
		      .def("LagResolution", &lagged_ensemble::LagResolution)
		      .def("Lag", &lagged_ensemble::Lag)
		      .def("Size", &lagged_ensemble::Size)
		      .def("ExpectedSize", &lagged_ensemble::ExpectedSize)
		      .def("NumberOfSteps", &lagged_ensemble::NumberOfSteps)
		      .def("VerifyValidForecastCount", &lagged_ensemble::VerifyValidForecastCount)
		      .def("SetMaximumMissingForecasts", LUA_MEMFN(void, lagged_ensemble, MaximumMissingForecasts, int))
		      .def("GetMaximumMissingForecasts", LUA_CMEMFN(int, lagged_ensemble, MaximumMissingForecasts, void)),
		  class_<modifier>("modifier")
		      .def("Result", &modifier_wrapper::Result)
		      .def("CalculationFinished", &modifier::CalculationFinished)
		      .def("GetLowerHeightGrid", &modifier_wrapper::GetLowerHeightGrid)
		      .def("SetLowerHeightGrid", &modifier_wrapper::SetLowerHeightGrid)
		      .def("GetUpperHeightGrid", &modifier_wrapper::GetUpperHeightGrid)
		      .def("SetUpperHeightGrid", &modifier_wrapper::SetUpperHeightGrid)
		      .def("GetFindValue", &modifier_wrapper::GetFindValueGrid)
		      .def("SetFindValue", &modifier_wrapper::SetFindValueGrid)
		      .def("SetFindNth", LUA_MEMFN(void, modifier, FindNth, size_t))
		      .def("GetFindNth", LUA_CMEMFN(size_t, modifier, FindNth, void)),
		  class_<modifier_findvalue, modifier>("modifier_findvalue")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::findvalue::Process),
		  class_<modifier_findheight, modifier>("modifier_findheight")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::findheight::Process),
		  class_<modifier_findheight_lt, modifier>("modifier_findheight_lt")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::findheight_lt::Process),
		  class_<modifier_findheight_gt, modifier>("modifier_findheight_gt")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::findheight_gt::Process),
		  class_<modifier_max, modifier>("modifier_max")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::max::Process),
		  class_<modifier_min, modifier>("modifier_min")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::min::Process),
		  class_<modifier_maxmin, modifier>("modifier_maxmin")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::maxmin::Process),
		  class_<modifier_count, modifier>("modifier_count")
		      .def(constructor<>())
		      .def("ClassName", &modifier_findvalue::ClassName)
		      .def("Process", &modifier_wrapper::count::Process),
		  class_<modifier_mean, modifier>("modifier_mean")
		      .def(constructor<>())
		      .def("ClassName", &modifier_mean::ClassName)
		      .def("Process", &modifier_wrapper::mean::Process)
		      .def("Result", &modifier_wrapper::mean::Result),
	          // numerical_functions namespace
	          def("Filter2D", &numerical_functions::Filter2D),
	          def("Max2D", &numerical_functions::Max2D),
	          def("Min2D", &numerical_functions::Min2D),
	          // metutil namespace
	          def("LCL_", &metutil::LCL_), def("Es_", &metutil::Es_), def("Gammas_", &metutil::Gammas_),
	          def("Gammaw_", &metutil::Gammaw_), def("MixingRatio_", &metutil::MixingRatio_),
	          def("MoistLift_", &metutil::MoistLift_), def("DryLift_", &metutil::DryLift_),
		  def("FlightLevel_", &metutil::FlightLevel_),
		  // himan namespace
		  def("IsMissing", static_cast<bool(*)(double)>(&::IsMissing)),
		  def("IsValid", static_cast<bool(*)(double)>(&::IsValid))];
}

void BindPlugins(lua_State* L)
{
	module(L)[class_<compiled_plugin_base>("compiled_plugin_base")
	              .def(constructor<>())
	              .def("WriteToFile", LUA_MEMFN(void, luatool, WriteToFile, const info_t& targetInfo)),
	          class_<luatool, compiled_plugin_base>("luatool")
	              .def(constructor<>())
	              .def("ClassName", &luatool::ClassName)
	              .def("FetchInfo", &luatool::FetchInfo)
	              .def("Fetch", LUA_CMEMFN(object, luatool, Fetch, const forecast_time&, const level&, const param&))
	              .def("FetchWithType", LUA_CMEMFN(object, luatool, Fetch, const forecast_time&, const level&,
	                                               const param&, const forecast_type&)),
	          class_<hitool, std::shared_ptr<hitool>>("hitool")
	              .def(constructor<>())
	              .def("ClassName", &hitool::ClassName)
	              // Local functions to luatool
	              .def("VerticalMaximumGrid", &hitool_wrapper::VerticalMaximumGrid)
	              .def("VerticalMaximum", &hitool_wrapper::VerticalMaximum)
	              .def("VerticalMinimumGrid", &hitool_wrapper::VerticalMinimumGrid)
	              .def("VerticalMinimum", &hitool_wrapper::VerticalMinimum)
	              .def("VerticalSumGrid", &hitool_wrapper::VerticalSumGrid)
	              .def("VerticalSum", &hitool_wrapper::VerticalSum)
	              .def("VerticalAverageGrid", &hitool_wrapper::VerticalAverageGrid)
	              .def("VerticalAverage", &hitool_wrapper::VerticalAverage)
	              .def("VerticalCountGrid", &hitool_wrapper::VerticalCountGrid)
	              .def("VerticalCount", &hitool_wrapper::VerticalCount)
	              .def("VerticalHeightGrid", &hitool_wrapper::VerticalHeightGrid)
	              .def("VerticalHeight", &hitool_wrapper::VerticalHeight)
	              .def("VerticalHeightGreaterThanGrid", &hitool_wrapper::VerticalHeightGreaterThanGrid)
	              .def("VerticalHeightGreaterThan", &hitool_wrapper::VerticalHeightGreaterThan)
	              .def("VerticalHeightLessThanGrid", &hitool_wrapper::VerticalHeightLessThanGrid)
	              .def("VerticalHeightLessThan", &hitool_wrapper::VerticalHeightLessThan)
	              .def("VerticalValueGrid", &hitool_wrapper::VerticalValueGrid)
	              .def("VerticalValue", &hitool_wrapper::VerticalValue)
	              .def("VerticalPlusMinusAreaGrid", &hitool_wrapper::VerticalPlusMinusAreaGrid)
	              .def("VerticalPlusMinusArea", &hitool_wrapper::VerticalPlusMinusArea)
	              .def("SetHeightUnit", &hitool_wrapper::SetHeightUnit)
	              .def("GetHeightUnit", &hitool_wrapper::GetHeightUnit),
	          class_<neons, std::shared_ptr<neons>>("neons")
	              .def(constructor<>())
	              .def("ClassName", &neons::ClassName)
	              .def("GetProducerMetaData", &neons_wrapper::GetProducerMetaData),
	          class_<radon, std::shared_ptr<radon>>("radon")
	              .def(constructor<>())
	              .def("GetProducerMetaData", &radon_wrapper::GetProducerMetaData)];
}

// clang-format on

void luatool::Run(info_t myTargetInfo, unsigned short threadIndex)
{
	while (Next(*myTargetInfo))
	{
		Calculate(myTargetInfo, threadIndex);

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
		}
	}
}

void luatool::Finish()
{
	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer.Stop();
		itsConfiguration->Statistics()->AddToProcessingTime(itsTimer.GetTime());
	}
}

std::shared_ptr<info> luatool::FetchInfo(const forecast_time& theTime, const level& theLevel,
                                         const param& theParam) const
{
	return compiled_plugin_base::Fetch(theTime, theLevel, theParam, forecast_type(kDeterministic), false);
}

luabind::object luatool::Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam) const
{
	return luatool::Fetch(theTime, theLevel, theParam, forecast_type(kDeterministic));
}

luabind::object luatool::Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
                               const forecast_type& theType) const
{
	auto x = compiled_plugin_base::Fetch(theTime, theLevel, theParam, theType, false);

	if (!x)
	{
		return object();
	}

	return VectorToTable(x->Data().Values());
}

object VectorToTable(const std::vector<double>& vec)
{
	assert(myL.get());

	object ret = newtable(myL.get());

	size_t i = 0;
	for (const double& val : vec)
	{
		ret[++i] = val;

		/*		"Lua tables make no distinction between a table value being nil and the corresponding key not existing
		   in
		   the table"
		        if (val == MissingDouble())
		        {
		            ret[++i] = nil;
		        }
		        else
		        {
		            ret[++i] = val;
		        }*/
	}

	return ret;
}

std::vector<double> TableToVector(const object& table)
{
	assert(table.is_valid());

	if (type(table) == 0)
	{
		// Input argument is nil (lua.h)
		return std::vector<double>();
	}

	luabind::iterator iter(table), end;

	auto size = std::distance(iter, end);
	std::vector<double> ret(size, himan::MissingDouble());

	size_t i = 0;
	for (; iter != end; ++iter, i++)
	{
		try
		{
			ret[i] = object_cast<double>(*iter);
		}
		catch (cast_failed& e)
		{
			ret[i] = himan::MissingDouble();
		}
	}
	return ret;
}

void luatool::WriteToFile(const info& targetInfo, write_options writeOptions)
{
	// Do nothing, override is needed to prevent double write
}

void luatool::WriteToFile(const info_t& targetInfo) { compiled_plugin_base::WriteToFile(*targetInfo, itsWriteOptions); }
#endif  // __clang_analyzer__
