#include "luatool.h"
#include "ensemble.h"
#include "forecast_time.h"
#include "hitool.h"
#include "lagged_ensemble.h"
#include "latitude_longitude_grid.h"
#include "lift.h"
#include "logger.h"
#include "metutil.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "radon.h"
#include "statistics.h"
#include "stereographic_grid.h"
#include <boost/filesystem.hpp>
#include <thread>

#ifndef __clang_analyzer__

extern "C" {
#include <lualib.h>
}

#include <luabind/adopt_policy.hpp>
#include <luabind/iterator_policy.hpp>
#include <luabind/luabind.hpp>
#include <luabind/operator.hpp>

using namespace himan;
using namespace himan::plugin;
using namespace luabind;

#define LUA_MEMFN(r, t, m, ...) static_cast<r (t::*)(__VA_ARGS__)>(&t::m)
#define LUA_CMEMFN(r, t, m, ...) static_cast<r (t::*)(__VA_ARGS__) const>(&t::m)

void BindEnum(lua_State* L);
int BindErrorHandler(lua_State* L);
void BindPlugins(lua_State* L);
void BindLib(lua_State* L);

template <typename T>
object VectorToTable(const std::vector<T>& vec);

template <typename T>
std::vector<T> TableToVector(const object& table);

namespace
{
thread_local lua_State* myL;
bool myUseCuda;
}

luatool::luatool() : itsWriteOptions()
{
	itsLogger = logger("luatool");
	myL = 0;
}

luatool::~luatool()
{
}
void luatool::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("DUMMY")});

	if (!itsConfiguration->GetValue("ThreadDistribution").empty())
	{
		if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForAny")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForAny;
		}
		else if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForForecastTypeAndTime")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForForecastTypeAndTime;
		}
		else if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForForecastTypeAndLevel")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForForecastTypeAndLevel;
		}
		else if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForTimeAndLevel")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForTimeAndLevel;
		}
		else if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForForecastType")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForForecastType;
		}
		else if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForTime")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForTime;
		}
		else if (itsConfiguration->GetValue("ThreadDistribution") == "kThreadForLevel")
		{
			itsThreadDistribution = ThreadDistribution::kThreadForLevel;
		}
	}

	myUseCuda = itsConfiguration->UseCuda();

	Start();
}

void luatool::Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("luatoolThread #" + std::to_string(threadIndex));

	InitLua();

	ASSERT(myL);
	myThreadedLogger.Info("Calculating time " + static_cast<std::string>(myTargetInfo->Time().ValidDateTime()) +
	                      " level " + static_cast<std::string>(myTargetInfo->Level()));

	globals(myL)["logger"] = myThreadedLogger;

	for (const std::string& luaFile : itsConfiguration->GetValueList("luafile"))
	{
		if (luaFile.empty())
		{
			continue;
		}

		myThreadedLogger.Info("Starting script " + luaFile);

		ResetVariables(myTargetInfo);
		ReadFile(luaFile);
	}

	lua_close(myL);
	myL = 0;
}

void luatool::InitLua()
{
	lua_State* L = luaL_newstate();

	ASSERT(L);

	luaL_openlibs(L);

	open(L);

	set_pcall_callback(&BindErrorHandler);

	BindEnum(L);
	BindLib(L);
	BindPlugins(L);

	myL = L;
}

void luatool::ResetVariables(info_t myTargetInfo)
{
	// Set some variable that are needed in luatool calculations
	// but are too hard or complicated to create in the lua side

	const auto L = myL;

	globals(L)["luatool"] = boost::ref(*this);
	globals(L)["result"] = myTargetInfo;
	globals(L)["configuration"] = itsConfiguration;
	globals(L)["write_options"] = boost::ref(itsWriteOptions);

	// Useful variables
	globals(L)["current_time"] = forecast_time(myTargetInfo->Time());
	globals(L)["current_level"] = level(myTargetInfo->Level());
	globals(L)["current_forecast_type"] = forecast_type(myTargetInfo->ForecastType());
	globals(L)["missing"] = MissingDouble();
	globals(L)["missingf"] = MissingFloat();
	globals(L)["kHPMissingValue"] = kHPMissingValue;  // todo: remove this constant altogether

	globals(L)["kKelvin"] = constants::kKelvin;

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(forecast_time(myTargetInfo->Time()));
	h->ForecastType(forecast_type(myTargetInfo->ForecastType()));

	auto r = GET_PLUGIN(radon);

	// Useful plugins
	globals(L)["hitool"] = h;
	globals(L)["radon"] = r;
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
		timer t(true);
		ASSERT(myL);
		if (luaL_dofile(myL, luaFile.c_str()))
		{
			itsLogger.Error(lua_tostring(myL, -1));
			return false;
		}
		t.Stop();
		itsLogger.Debug("Script " + luaFile + " executed in " + std::to_string(t.GetTime()) + " ms");
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
	         .enum_("constants")[
				 value("kUnknownTimeResolution", kUnknownTimeResolution),
				 value("kHourResolution", kHourResolution),
				 value("kMinuteResolution", kMinuteResolution),
				 value("kDayResolution", kDayResolution),
				 value("kMonthResolution", kMonthResolution)],
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
	     class_<HPProcessingType>("HPProcessingType")
	         .enum_("constants")[
				 value("kUnknownProcessingType", kUnknownProcessingType),
				 value("kProbabilityGreaterThan", kProbabilityGreaterThan),
				 value("kProbabilityLessThan", kProbabilityLessThan),
				 value("kProbabilityBetween", kProbabilityBetween),
				 value("kProbabilityEquals", kProbabilityEquals),
				 value("kProbabilityNotEquals", kProbabilityNotEquals),
				 value("kProbabilityEqualsIn", kProbabilityEqualsIn),
				 value("kFractile", kFractile),
				 value("kEnsembleMean", kEnsembleMean),
				 value("kSpread", kSpread),
				 value("kStandardDeviation", kStandardDeviation),
				 value("kEFI", kEFI)],
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
	         .enum_("constants")[
				 value("kUnknownGridType", kUnknownGridType),
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
				 value("kEpsPerturbation", kEpsPerturbation),
				 value("kStatisticalProcessing", kStatisticalProcessing)]];
}

// clang-format on

namespace info_wrapper
{
// These are convenience functions for accessing info class contents

template <typename T>
void SetValue(std::shared_ptr<info<T>>& anInfo, int index, double value)
{
	anInfo->Data().Set(--index, static_cast<T>(value));
}
template <typename T>
double GetValue(std::shared_ptr<info<T>>& anInfo, int index)
{
	return anInfo->Data().At(--index);
}
template <typename T>
size_t GetLocationIndex(std::shared_ptr<info<T>> anInfo)
{
	return anInfo->LocationIndex() + 1;
}
template <typename T>
size_t GetTimeIndex(std::shared_ptr<info<T>> anInfo)
{
	return anInfo->template Index<forecast_time>() + 1;
}
template <typename T>
size_t GetParamIndex(std::shared_ptr<info<T>> anInfo)
{
	return anInfo->template Index<param>() + 1;
}
template <typename T>
size_t GetLevelIndex(std::shared_ptr<info<T>> anInfo)
{
	return anInfo->template Index<level>() + 1;
}
template <typename T>
size_t GetForecastTypeIndex(std::shared_ptr<info<T>> anInfo)
{
	return anInfo->template Index<forecast_type>() + 1;
}

template <typename T>
void SetLocationIndex(std::shared_ptr<info<T>> anInfo, size_t theIndex)
{
	anInfo->template LocationIndex(--theIndex);
}
template <typename T>
void SetTimeIndex(std::shared_ptr<info<T>> anInfo, size_t theIndex)
{
	anInfo->template Index<forecast_time>(--theIndex);
}
template <typename T>
void SetParamIndex(std::shared_ptr<info<T>> anInfo, size_t theIndex)
{
	anInfo->template Index<param>(--theIndex);
}
template <typename T>
void SetLevelIndex(std::shared_ptr<info<T>> anInfo, size_t theIndex)
{
	anInfo->template Index<level>(--theIndex);
}
template <typename T>
void SetForecastTypeIndex(std::shared_ptr<info<T>> anInfo, size_t theIndex)
{
	anInfo->template Index<forecast_type>(--theIndex);
}
template <typename T>
void SetValues(std::shared_ptr<info<T>>& anInfo, const object& table)
{
	std::vector<T> vals = TableToVector<T>(table);

	if (vals.empty())
	{
		return;
	}

	if (vals.size() != anInfo->Data().Size())
	{
		std::cerr << "Error::luatool input table size is not the same as grid size: " << vals.size() << " vs "
		          << anInfo->Data().Size() << std::endl;
		return;
	}

	// Reset data here also. If we would not reset, a single info *within in a script* would
	// be recycled all along, which would mean that newer data would overwrite older data
	// (that's already in cache!).

	auto g = std::shared_ptr<grid>(anInfo->Grid()->Clone());
	matrix<T> d(anInfo->Data().SizeX(), anInfo->Data().SizeY(), 1, anInfo->Data().MissingValue(), vals);

	anInfo->Base(std::make_shared<base<T>>(g, d));
}
template <typename T>
void SetValuesFromMatrix(std::shared_ptr<info<T>>& anInfo, const matrix<T>& mat)
{
	if (mat.Size() != anInfo->Data().Size())
	{
		std::cerr << "Error::luatool input table size is not the same as grid size: " << mat.Size() << " vs "
		          << anInfo->Data().Size() << std::endl;
	}
	else
	{
		anInfo->Data().Set(mat.Values());
	}
}
template <typename T>
object GetValues(info_t& anInfo)
{
	return VectorToTable<double>(VEC(anInfo));
}
template <typename T>
point GetLatLon(std::shared_ptr<info<T>>& anInfo, size_t theIndex)
{
	return anInfo->Grid()->LatLon(--theIndex);
}
template <typename T>
double GetMissingValue(std::shared_ptr<info<T>>& anInfo)
{
	return anInfo->Data().MissingValue();
}
template <typename T>
void SetMissingValue(std::shared_ptr<info<T>>& anInfo, T missingValue)
{
	anInfo->Data().MissingValue(missingValue);
}
template <typename T>
matrix<T> GetData(std::shared_ptr<info<T>>& anInfo)
{
	return anInfo->Data();
}
template <typename T>
void SetParam(std::shared_ptr<info<T>>& anInfo, const param& par)
{
	auto r = GET_PLUGIN(radon);

	param newpar(par);

	const auto lvl = anInfo->template Peek<level>(0);
	auto paramInfo =
	    r->RadonDB().GetParameterFromDatabaseName(anInfo->Producer().Id(), par.Name(), lvl.Type(), lvl.Value());

	if (!paramInfo.empty())
	{
		newpar = param(paramInfo);

		if (par.Aggregation().Type() != kUnknownAggregationType)
		{
			newpar.Aggregation(par.Aggregation());
		}
		if (par.ProcessingType().Type() != kUnknownProcessingType)
		{
			newpar.ProcessingType(par.ProcessingType());
		}
	}

	anInfo->template Set<param>(newpar);
}
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
		return VectorToTable<double>(h->VerticalMaximum<double>(theParam, TableToVector<double>(firstLevelValue),
		                                                        TableToVector<double>(lastLevelValue)));
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
		return VectorToTable<double>(h->VerticalMaximum<double>(theParam, firstLevelValue, lastLevelValue));
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
		return VectorToTable<double>(h->VerticalMinimum<double>(theParam, TableToVector<double>(firstLevelValue),
		                                                        TableToVector<double>(lastLevelValue)));
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
		return VectorToTable<double>(h->VerticalMinimum<double>(theParam, firstLevelValue, lastLevelValue));
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
		return VectorToTable<double>(h->VerticalSum<double>(theParam, TableToVector<double>(firstLevelValue),
		                                                    TableToVector<double>(lastLevelValue)));
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
		return VectorToTable<double>(h->VerticalSum<double>(theParam, firstLevelValue, lastLevelValue));
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
		return VectorToTable<double>(h->VerticalAverage<double>(theParam, TableToVector<double>(firstLevelValue),
		                                                        TableToVector<double>(lastLevelValue)));
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
		return VectorToTable<double>(h->VerticalAverage<double>(theParams, firstLevelValue, lastLevelValue));
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
		return VectorToTable<double>(h->VerticalCount<double>(theParams, TableToVector<double>(firstLevelValue),
		                                                      TableToVector<double>(lastLevelValue),
		                                                      TableToVector<double>(findValue)));
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
		return VectorToTable<double>(h->VerticalCount<double>(theParams, firstLevelValue, lastLevelValue, findValue));
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
                          const object& lastLevelValue, const object& findValue, int findNth)
{
	try
	{
		return VectorToTable<double>(h->VerticalHeight<double>(theParam, TableToVector<double>(firstLevelValue),
		                                                       TableToVector<double>(lastLevelValue),
		                                                       TableToVector<double>(findValue), findNth));
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
                      double findValue, int findNth)
{
	try
	{
		return VectorToTable<double>(
		    h->VerticalHeight<double>(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
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
                                     const object& lastLevelValue, const object& findValue, int findNth)
{
	try
	{
		return VectorToTable<double>(h->VerticalHeightGreaterThan<double>(
		    theParam, TableToVector<double>(firstLevelValue), TableToVector<double>(lastLevelValue),
		    TableToVector<double>(findValue), findNth));
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
                                 double lastLevelValue, double findValue, int findNth)
{
	try
	{
		return VectorToTable<double>(
		    h->VerticalHeightGreaterThan<double>(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
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
                                  const object& lastLevelValue, const object& findValue, int findNth)
{
	try
	{
		return VectorToTable<double>(h->VerticalHeightLessThan<double>(theParam, TableToVector<double>(firstLevelValue),
		                                                               TableToVector<double>(lastLevelValue),
		                                                               TableToVector<double>(findValue), findNth));
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
                              double lastLevelValue, double findValue, int findNth)
{
	try
	{
		return VectorToTable<double>(
		    h->VerticalHeightLessThan<double>(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
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
		return VectorToTable<double>(h->VerticalValue<double>(theParam, TableToVector<double>(findValue)));
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
		return VectorToTable<double>(h->VerticalValue<double>(theParam, findValue));
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
		return VectorToTable<double>(h->PlusMinusArea<double>(theParams, TableToVector<double>(firstLevelValue),
		                                                      TableToVector<double>(lastLevelValue)));
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
		return VectorToTable<double>(h->PlusMinusArea<double>(theParams, firstLevelValue, lastLevelValue));
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

void Time(std::shared_ptr<hitool> h, const forecast_time& theTime)
{
	h->Time(theTime);
}
void SetHeightUnit(std::shared_ptr<hitool> h, HPParameterUnit theHeightUnit)
{
	h->HeightUnit(theHeightUnit);
}
HPParameterUnit GetHeightUnit(std::shared_ptr<hitool> h)
{
	return h->HeightUnit();
}
}  // namespace hitool_wrapper

namespace modifier_wrapper
{
void SetLowerHeightGrid(modifier& mod, const object& lowerHeight)
{
	mod.LowerHeight(TableToVector<double>(lowerHeight));
}
object GetLowerHeightGrid(modifier& mod)
{
	return VectorToTable<double>(mod.LowerHeight());
}
void SetUpperHeightGrid(modifier& mod, const object& upperHeight)
{
	mod.UpperHeight(TableToVector<double>(upperHeight));
}
object GetUpperHeightGrid(modifier& mod)
{
	return VectorToTable<double>(mod.UpperHeight());
}
void SetFindValueGrid(modifier& mod, const object& findValue)
{
	mod.FindValue(TableToVector<double>(findValue));
}
object GetFindValueGrid(modifier& mod)
{
	return VectorToTable<double>(mod.FindValue());
}
object Result(modifier& mod)
{
	return VectorToTable<double>(mod.Result());
}
namespace findvalue
{
void Process(modifier_findvalue& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace findvalue

namespace findheight
{
void Process(modifier_findheight& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace findheight

namespace findheight_gt
{
void Process(modifier_findheight_gt& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace findheight_gt

namespace findheight_lt
{
void Process(modifier_findheight_lt& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace findheight_lt

namespace max
{
void Process(modifier_max& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace max

namespace min
{
void Process(modifier_min& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace min

namespace maxmin
{
void Process(modifier_maxmin& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace maxmin

namespace count
{
void Process(modifier_count& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace count

namespace mean
{
object Result(modifier_mean& mod)
{
	return VectorToTable<double>(mod.Result());
}
void Process(modifier_mean& mod, const object& data, const object& height)
{
	mod.Process(TableToVector<double>(data), TableToVector<double>(height));
}
}  // namespace mean

}  // namespace modifier_wrapper

namespace radon_wrapper
{
std::string GetProducerMetaData(std::shared_ptr<radon> r, const producer& prod, const std::string& attName)
{
	return r->RadonDB().GetProducerMetaData(prod.Id(), attName);
}

}  // namespace radon_wrapper

namespace ensemble_wrapper
{
object Values(const ensemble& ens)
{
	return VectorToTable<float>(ens.Values());
}
object SortedValues(const ensemble& ens)
{
	return VectorToTable<float>(ens.SortedValues());
}
}  // ensemble_wrapper

namespace lagged_ensemble_wrapper
{
object Values(const lagged_ensemble& ens)
{
	return VectorToTable<float>(ens.Values());
}
object SortedValues(const lagged_ensemble& ens)
{
	return VectorToTable<float>(ens.SortedValues());
}
}  // lagged_ensemble_wrapper

namespace matrix_wrapper
{
template <typename T>
void SetValues(matrix<T>& mat, const object& values)
{
	mat.Set(TableToVector<T>(values));
}
template <typename T>
object GetValues(matrix<T>& mat)
{
	return VectorToTable<T>(std::vector<T>(mat.Values()));
}
template <typename T>
void Fill(matrix<T>& mat, T value)
{
	mat.Fill(value);
}
}  // matrix_wrapper

namespace luabind_workaround
{
template <typename T>
matrix<T> ProbLimitGt2D(const matrix<T>& A, const matrix<T>& B, T limit)
{
#ifdef HAVE_CUDA
	if (myUseCuda)
	{
		return numerical_functions::ProbLimitGt2DGPU<T>(A, B, limit);
	}
#endif
	return numerical_functions::Prob2D<T>(A, B, [=](const T& val1) { return val1 > limit; });
}

template <typename T>
matrix<T> ProbLimitGe2D(const matrix<T>& A, const matrix<T>& B, T limit)
{
#ifdef HAVE_CUDA
	if (myUseCuda)
	{
		return numerical_functions::ProbLimitGe2DGPU<T>(A, B, limit);
	}
#endif
	return numerical_functions::Prob2D<T>(A, B, [=](const T& val) { return val >= limit; });
}

template <typename T>
matrix<T> ProbLimitLt2D(const matrix<T>& A, const matrix<T>& B, T limit)
{
#ifdef HAVE_CUDA
	if (myUseCuda)
	{
		return numerical_functions::ProbLimitLt2DGPU<T>(A, B, limit);
	}
#endif
	return numerical_functions::Prob2D<T>(A, B, [=](const T& val) { return val < limit; });
}

template <typename T>
matrix<T> ProbLimitLe2D(const matrix<T>& A, const matrix<T>& B, T limit)
{
#ifdef HAVE_CUDA
	if (myUseCuda)
	{
		return numerical_functions::ProbLimitLe2DGPU<T>(A, B, limit);
	}
#endif
	return numerical_functions::Prob2D<T>(A, B, [=](const T& val) { return val <= limit; });
}

template <typename T>
matrix<T> ProbLimitEq2D(const matrix<T>& A, const matrix<T>& B, T limit)
{
#ifdef HAVE_CUDA
	if (myUseCuda)
	{
		return numerical_functions::ProbLimitEq2DGPU<T>(A, B, limit);
	}
#endif
	return numerical_functions::Prob2D<T>(A, B, [=](const T& val) { return val == limit; });
}
}

// clang-format off

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

void BindLib(lua_State* L)
{
	module(L)[class_<himan::info<double>, std::shared_ptr<himan::info<double>>>("info")
	              .def(constructor<>())
	              .def("ClassName", &info<double>::ClassName)
	              .def("ResetParam", &info<double>::Reset<param>)
	              .def("FirstParam", &info<double>::First<param>)
	              .def("NextParam", &info<double>::Next<param>)
	              .def("ResetLevel", &info<double>::Reset<level>)
	              .def("FirstLevel", &info<double>::First<level>)
	              .def("NextLevel", &info<double>::Next<level>)
	              .def("ResetTime", &info<double>::Reset<forecast_time>)
	              .def("FirstTime", &info<double>::First<forecast_time>)
	              .def("NextTime", &info<double>::Next<forecast_time>)
	              .def("ResetForecastType", &info<double>::Reset<forecast_type>)
	              .def("FirstForecastType", &info<double>::First<forecast_type>)
	              .def("NextForecastType", &info<double>::Next<forecast_type>)
	              .def("SizeLocations", LUA_CMEMFN(size_t, info<double>, SizeLocations, void))
	              .def("SizeTimes", LUA_CMEMFN(size_t, info<double>, Size<forecast_time>, void))
	              .def("SizeParams", LUA_CMEMFN(size_t, info<double>, Size<param>, void))
	              .def("SizeLevels", LUA_CMEMFN(size_t, info<double>, Size<level>, void))
	              .def("SizeForecastTypes", LUA_CMEMFN(size_t, info<double>, Size<forecast_type>, void))
	              .def("GetLevel", LUA_CMEMFN(const level&, info<double>, Level, void))
	              .def("GetTime", LUA_CMEMFN(const forecast_time&, info<double>, Time, void))
	              .def("GetParam", LUA_CMEMFN(const param&, info<double>, Param, void))
	              .def("GetForecastType", LUA_CMEMFN(const forecast_type&, info<double>, ForecastType, void))
	              .def("GetGrid", LUA_CMEMFN(std::shared_ptr<grid>, info<double>, Grid, void))
	              .def("SetTime", LUA_MEMFN(void, info<double>, Set<forecast_time>, const forecast_time&))
	              .def("SetLevel", LUA_MEMFN(void, info<double>, Set<level>, const level&))
	              .def("SetForecastType", LUA_MEMFN(void, info<double>, Set<forecast_type>, const forecast_type&))
	              //.def("SetParam", LUA_MEMFN(void, info, SetParam, const param&))
	              // These are local functions to luatool
	              .def("SetParam", &info_wrapper::SetParam<double>)
	              .def("SetIndexValue", &info_wrapper::SetValue<double>)
	              .def("GetIndexValue", &info_wrapper::GetValue<double>)
	              .def("GetTimeIndex", &info_wrapper::GetTimeIndex<double>)
	              .def("GetParamIndex", &info_wrapper::GetParamIndex<double>)
	              .def("GetLevelIndex", &info_wrapper::GetLevelIndex<double>)
	              .def("GetForecastTypeIndex", &info_wrapper::GetForecastTypeIndex<double>)
	              .def("SetTimeIndex", &info_wrapper::SetTimeIndex<double>)
	              .def("SetParamIndex", &info_wrapper::SetParamIndex<double>)
	              .def("SetLevelIndex", &info_wrapper::SetLevelIndex<double>)
	              .def("SetForecastTypeIndex", &info_wrapper::SetForecastTypeIndex<double>)
	              .def("SetValues", &info_wrapper::SetValues<double>)
	              .def("SetValuesFromMatrix", &info_wrapper::SetValuesFromMatrix<double>)
	              .def("GetValues", &info_wrapper::GetValues<double>)
	              .def("GetLatLon", &info_wrapper::GetLatLon<double>)
	              .def("GetMissingValue", &info_wrapper::GetMissingValue<double>)
	              .def("SetMissingValue", &info_wrapper::SetMissingValue<double>)
	              .def("GetData", &info_wrapper::GetData<double>),
	          class_<himan::info<float>, std::shared_ptr<himan::info<float>>>("infof")
	              .def(constructor<>())
	              .def("ClassName", &info<float>::ClassName)
	              .def("ResetParam", &info<float>::Reset<param>)
	              .def("FirstParam", &info<float>::First<param>)
	              .def("NextParam", &info<float>::Next<param>)
	              .def("ResetLevel", &info<float>::Reset<level>)
	              .def("FirstLevel", &info<float>::First<level>)
	              .def("NextLevel", &info<float>::Next<level>)
	              .def("ResetTime", &info<float>::Reset<forecast_time>)
	              .def("FirstTime", &info<float>::First<forecast_time>)
	              .def("NextTime", &info<float>::Next<forecast_time>)
	              .def("ResetForecastType", &info<float>::Reset<forecast_type>)
	              .def("FirstForecastType", &info<float>::First<forecast_type>)
	              .def("NextForecastType", &info<float>::Next<forecast_type>)
	              .def("SizeLocations", LUA_CMEMFN(size_t, info<float>, SizeLocations, void))
	              .def("SizeTimes", LUA_CMEMFN(size_t, info<float>, Size<forecast_time>, void))
	              .def("SizeParams", LUA_CMEMFN(size_t, info<float>, Size<param>, void))
	              .def("SizeLevels", LUA_CMEMFN(size_t, info<float>, Size<level>, void))
	              .def("SizeForecastTypes", LUA_CMEMFN(size_t, info<float>, Size<forecast_type>, void))
	              .def("GetLevel", LUA_CMEMFN(const level&, info<float>, Level, void))
	              .def("GetTime", LUA_CMEMFN(const forecast_time&, info<float>, Time, void))
	              .def("GetParam", LUA_CMEMFN(const param&, info<float>, Param, void))
	              .def("GetForecastType", LUA_CMEMFN(const forecast_type&, info<float>, ForecastType, void))
	              .def("GetGrid", LUA_CMEMFN(std::shared_ptr<grid>, info<float>, Grid, void))
	              .def("SetTime", LUA_MEMFN(void, info<float>, Set<forecast_time>, const forecast_time&))
	              .def("SetLevel", LUA_MEMFN(void, info<float>, Set<level>, const level&))
	              .def("SetParam", &info_wrapper::SetParam<float>)
	              .def("SetForecastType", LUA_MEMFN(void, info<float>, Set<forecast_type>, const forecast_type&))
	              .def("SetIndexValue", &info_wrapper::SetValue<float>)
	              .def("GetIndexValue", &info_wrapper::GetValue<float>)
	              .def("GetTimeIndex", &info_wrapper::GetTimeIndex<float>)
	              .def("GetParamIndex", &info_wrapper::GetParamIndex<float>)
	              .def("GetLevelIndex", &info_wrapper::GetLevelIndex<float>)
	              .def("GetForecastTypeIndex", &info_wrapper::GetForecastTypeIndex<float>)
	              .def("SetTimeIndex", &info_wrapper::SetTimeIndex<float>)
	              .def("SetParamIndex", &info_wrapper::SetParamIndex<float>)
	              .def("SetLevelIndex", &info_wrapper::SetLevelIndex<float>)
	              .def("SetForecastTypeIndex", &info_wrapper::SetForecastTypeIndex<float>)
	              .def("SetValues", &info_wrapper::SetValues<float>)
	              .def("SetValuesFromMatrix", &info_wrapper::SetValuesFromMatrix<float>)
	              .def("GetValues", &info_wrapper::GetValues<float>)
	              .def("GetLatLon", &info_wrapper::GetLatLon<float>)
	              .def("GetMissingValue", &info_wrapper::GetMissingValue<float>)
	              .def("SetMissingValue", &info_wrapper::SetMissingValue<float>)
	              .def("GetData", &info_wrapper::GetData<float>),
	          class_<grid, std::shared_ptr<grid>>("grid")
	              .def("ClassName", &grid::ClassName)
	              //.def("GetScanningMode", LUA_CMEMFN(HPScanningMode, grid, ScanningMode, void))
	              .def("GetGridType", LUA_CMEMFN(HPGridType, grid, Type, void))
	              .def("GetGridClass", LUA_CMEMFN(HPGridClass, grid, Class, void))
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
#if 0
	          class_<reduced_gaussian_grid, grid, std::shared_ptr<reduced_gaussian_grid>>("reduced_gaussian_grid")
	              .def(constructor<>())
	              .def("ClassName", &reduced_gaussian_grid::ClassName)
	              .def("GetN", LUA_CMEMFN(int, reduced_gaussian_grid, N, void))
	              .def("SetN", LUA_MEMFN(void, reduced_gaussian_grid, N, int))
	              .def("GetFirstPoint", LUA_CMEMFN(point, reduced_gaussian_grid, FirstPoint, void))
	              .def("GetLastPoint", LUA_CMEMFN(point, reduced_gaussian_grid, LastPoint, void))
	          ,
#endif
	          class_<matrix<double>>("matrix")
	              .def(constructor<size_t, size_t, size_t, double>())
	              .def("SetValues", &matrix_wrapper::SetValues<double>)
	              .def("GetValues", &matrix_wrapper::GetValues<double>)
	              .def("Fill", &matrix_wrapper::Fill<double>),
	          class_<matrix<float>>("matrixf")
	              .def(constructor<size_t, size_t, size_t, float>())
	              .def("SetValues", &matrix_wrapper::SetValues<float>)
	              .def("GetValues", &matrix_wrapper::GetValues<float>)
	              .def("Fill", &matrix_wrapper::Fill<float>),
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
	              .def("SetAggregation", LUA_MEMFN(void, param, Aggregation, const aggregation&))
	              .def("GetProcessingType", LUA_CMEMFN(const processing_type&, param, ProcessingType, void))
	              .def("SetProcessingType", LUA_MEMFN(void, param, ProcessingType, const processing_type&)),
	          class_<level>("level")
	              .def(constructor<HPLevelType, double>())
	              .def(constructor<HPLevelType, double, double>())
	              .def("ClassName", &level::ClassName)
	              .def(tostring(self))
	              .def("GetAB", LUA_CMEMFN(std::vector<double>, level, AB, void))
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
	              .def("GetStep", LUA_CMEMFN(time_duration, forecast_time, Step, void)),
	          class_<forecast_type>("forecast_type")
	              .def(constructor<HPForecastType>())
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
		  class_<time_duration>("time_duration")
		      .def(constructor<std::string>())
		      .def(constructor<HPTimeResolution, int>())
	              .def(tostring(self))
		      .def("Hours", &time_duration::Hours)
		      .def("Minutes", &time_duration::Minutes)
		      .def("Seconds", &time_duration::Seconds)
		      .def("Empty", &time_duration::Empty),
	          class_<aggregation>("aggregation")
	              .def(constructor<HPAggregationType, const time_duration&>())
	              .def(constructor<HPAggregationType, const time_duration&, const time_duration&>())
	              .def("ClassName", &aggregation::ClassName)
	              .def("GetType", LUA_CMEMFN(HPAggregationType, aggregation, Type, void))
	              .def("SetType", LUA_MEMFN(void, aggregation, Type, HPAggregationType))
	              .def("GetTimeDuration", LUA_CMEMFN(time_duration, aggregation, TimeDuration, void))
	              .def("SetTimeDuration", LUA_MEMFN(void, aggregation, TimeDuration, const time_duration&))
	              .def("GetTimeOffset", LUA_CMEMFN(time_duration, aggregation, TimeOffset, void))
	              .def("SetTimeOffset", LUA_MEMFN(void, aggregation, TimeOffset, const time_duration&)),
	          class_<processing_type>("processing_type")
	              .def(constructor<HPProcessingType, double, double>())
	              .def("ClassName", &processing_type::ClassName)
	              .def("GetType", LUA_CMEMFN(HPProcessingType, processing_type, Type, void))
	              .def("SetType", LUA_MEMFN(void, processing_type, Type, HPProcessingType))
	              .def("GetValue", LUA_CMEMFN(double, processing_type, Value, void))
	              .def("SetValue", LUA_MEMFN(void, processing_type, Value, double))
	              .def("GetValue2", LUA_CMEMFN(double, processing_type, Value2, void))
	              .def("SetValue2", LUA_MEMFN(void, processing_type, Value2, double))
	              .def("GetNumberOfEnsembleMembers", LUA_CMEMFN(int, processing_type, NumberOfEnsembleMembers, void))
	              .def("SetNumberOfEnsembleMembers", LUA_MEMFN(void, processing_type, NumberOfEnsembleMembers, int)),
	          class_<configuration, std::shared_ptr<configuration>>("configuration")
	              .def(constructor<>())
	              .def("ClassName", &configuration::ClassName)
	              .def("GetOutputFileType", LUA_CMEMFN(HPFileType, configuration, OutputFileType, void))
	              .def("GetSourceProducer", LUA_CMEMFN(const producer&, configuration, SourceProducer, size_t))
	              .def("GetTargetProducer", LUA_CMEMFN(const producer&, configuration, TargetProducer, void))
	              .def("GetForecastStep", LUA_CMEMFN(time_duration, configuration, ForecastStep, void))
	              .def("GetUseCuda", LUA_CMEMFN(bool, configuration, UseCuda, void))
	              ,
	          class_<plugin_configuration, configuration, std::shared_ptr<plugin_configuration>>("plugin_configuration")
	              .def(constructor<>())
	              .def("ClassName", &plugin_configuration::ClassName)
	              .def("GetValue", &plugin_configuration::GetValue)
	              .def("GetValueList", &plugin_configuration::GetValueList)
	              .def("Exists", &plugin_configuration::Exists),
	          class_<himan::metutil::lcl_t<double>>("lcl_t")
	              .def(constructor<>())
	              .def_readwrite("T", &himan::metutil::lcl_t<double>::T)
	              .def_readwrite("P", &himan::metutil::lcl_t<double>::P)
	              .def_readwrite("Q", &himan::metutil::lcl_t<double>::Q),
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
		      .def("CentralMoment", LUA_CMEMFN(float,ensemble,CentralMoment, int))
		      .def("Size", &ensemble::Size)
		      .def("ExpectedSize", &ensemble::ExpectedSize)
		      .def("SetMaximumMissingForecasts", LUA_MEMFN(void, ensemble, MaximumMissingForecasts, int))
		      .def("GetMaximumMissingForecasts", LUA_CMEMFN(int, ensemble, MaximumMissingForecasts, void))
		      .def("GetForecast", &ensemble::Forecast),
		  class_<himan::lagged_ensemble, ensemble, std::shared_ptr<himan::lagged_ensemble>>("lagged_ensemble")
		      .def(constructor<param, size_t, const time_duration&, size_t>())
		      .def(constructor<param, const std::string&>())
		      .def("ClassName", &lagged_ensemble::ClassName)
		      .def("Fetch", &lagged_ensemble::Fetch)
		      .def("Value", &lagged_ensemble::Value)
		      .def("Values", &lagged_ensemble_wrapper::Values)
		      .def("SortedValues", &lagged_ensemble_wrapper::SortedValues)
		      .def("ResetLocation", &lagged_ensemble::ResetLocation)
		      .def("FirstLocation", &lagged_ensemble::FirstLocation)
		      .def("NextLocation", &lagged_ensemble::NextLocation)
		      .def("Size", &lagged_ensemble::Size)
		      .def("ExpectedSize", &lagged_ensemble::ExpectedSize)
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
		      .def("SetFindNth", LUA_MEMFN(void, modifier, FindNth, int))
		      .def("GetFindNth", LUA_CMEMFN(int, modifier, FindNth, void)),
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
	          def("Filter2D", &numerical_functions::Filter2D<double>),
	          def("Filter2D", &numerical_functions::Filter2D<float>),
	          def("Max2D", &numerical_functions::Max2D<double>),
	          def("Max2D", &numerical_functions::Max2D<float>),
	          def("Min2D", &numerical_functions::Min2D<double>),
	          def("Min2D", &numerical_functions::Min2D<float>),
                  def("ProbLimitGt2D", &luabind_workaround::ProbLimitGt2D<double>),
                  def("ProbLimitGt2D", &luabind_workaround::ProbLimitGt2D<float>),
                  def("ProbLimitGe2D", &luabind_workaround::ProbLimitGe2D<double>),
                  def("ProbLimitGe2D", &luabind_workaround::ProbLimitGe2D<float>),
                  def("ProbLimitLt2D", &luabind_workaround::ProbLimitLt2D<double>),
                  def("ProbLimitLt2D", &luabind_workaround::ProbLimitLt2D<float>),
                  def("ProbLimitLe2D", &luabind_workaround::ProbLimitLe2D<double>),
                  def("ProbLimitLe2D", &luabind_workaround::ProbLimitLe2D<float>),
                  def("ProbLimitEq2D", &luabind_workaround::ProbLimitEq2D<double>),
                  def("ProbLimitEq2D", &luabind_workaround::ProbLimitEq2D<float>),
	          // metutil namespace
	          def("LCL_", &metutil::LCL_<double>), 
	          def("Es_", &metutil::Es_<double>), 
	          def("Gammas_", &metutil::Gammas_<double>),
	          def("Gammaw_", &metutil::Gammaw_<double>), 
	          def("MixingRatio_", &metutil::MixingRatio_<double>),
	          def("MoistLift_", &metutil::MoistLift_<double>), 
	          def("DryLift_", &metutil::DryLift_<double>),
		  def("FlightLevel_", &metutil::FlightLevel_),
		  // himan namespace
		  def("IsMissing", static_cast<bool(*)(double)>(&::IsMissing)),
		  def("IsValid", static_cast<bool(*)(double)>(&::IsValid))];
}

#pragma GCC diagnostic pop

void BindPlugins(lua_State* L)
{
	module(L)[class_<compiled_plugin_base>("compiled_plugin_base")
	              .def(constructor<>())
	              .def("WriteToFile", LUA_MEMFN(void, luatool, WriteToFile, const info_t targetInfo)),
	          class_<luatool, compiled_plugin_base>("luatool")
	              .def(constructor<>())
	              .def("ClassName", &luatool::ClassName)
	              .def("FetchInfo", LUA_CMEMFN(std::shared_ptr<himan::info<double>>, luatool, FetchInfo, const forecast_time&, const level&, const param&))
                      .def("FetchInfoWithType", LUA_CMEMFN(std::shared_ptr<himan::info<double>>, luatool, FetchInfo, const forecast_time&, const level&,
                                                           const param&, const forecast_type&))
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
	          class_<radon, std::shared_ptr<radon>>("radon")
	              .def(constructor<>())
	              .def("GetProducerMetaData", &radon_wrapper::GetProducerMetaData)];
}

// clang-format on

std::shared_ptr<info<double>> luatool::FetchInfo(const forecast_time& theTime, const level& theLevel,
                                                 const param& theParam) const
{
	return compiled_plugin_base::Fetch(theTime, theLevel, theParam, forecast_type(kDeterministic), false);
}

std::shared_ptr<info<double>> luatool::FetchInfo(const forecast_time& theTime, const level& theLevel,
                                                 const param& theParam, const forecast_type& theType) const
{
	return compiled_plugin_base::Fetch(theTime, theLevel, theParam, theType, false);
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
	return VectorToTable<double>(x->Data().Values());
}

template <typename T>
object VectorToTable(const std::vector<T>& vec)
{
	ASSERT(myL);

	object ret = newtable(myL);

	size_t i = 0;
	for (const T& val : vec)
	{
		ret[++i] = val;

		// "Lua tables make no distinction between a table value being nil and
		// the corresponding key not existing in the table"
	}

	return ret;
}

template object VectorToTable(const std::vector<double>&);
template object VectorToTable(const std::vector<float>&);

template <typename T>
std::vector<T> TableToVector(const object& table)
{
	ASSERT(table.is_valid());

	if (type(table) == 0)
	{
		// Input argument is nil (lua.h)
		return std::vector<T>();
	}

	luabind::iterator iter(table), end;

	auto size = std::distance(iter, end);

	std::vector<T> ret(size, himan::MissingValue<T>());

	size_t i = 0;
	for (; iter != end; ++iter, i++)
	{
		try
		{
			ret[i] = object_cast<T>(*iter);
		}
		catch (cast_failed& e)
		{
		}
	}
	return ret;
}

void luatool::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	// Do nothing, override is needed to prevent double write
}

void luatool::WriteToFile(const info_t targetInfo)
{
	compiled_plugin_base::WriteToFile(targetInfo, itsWriteOptions);
}
#endif  // __clang_analyzer__
