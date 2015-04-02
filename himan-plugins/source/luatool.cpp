#include "luatool.h"
#include <boost/filesystem.hpp>
#include "logger_factory.h"
#include "forecast_time.h"
#include "util.h"
#include "metutil.h"
#include "regular_grid.h"
#include "plugin_factory.h"
#include <boost/foreach.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

extern "C"
{
#include <lualib.h>
}

#include <luabind/luabind.hpp>
#include <luabind/iterator_policy.hpp>
#include <luabind/adopt_policy.hpp>
#include <luabind/iterator_policy.hpp>
#include <luabind/operator.hpp>
//#include <boost/get_pointer.hpp>

using namespace himan;
using namespace himan::plugin;
using namespace luabind;

#define LUA_MEMFN(r, t,  m, ...) static_cast<r(t::*)(__VA_ARGS__)>(&t::m)
#define LUA_CMEMFN(r, t, m, ...) static_cast<r(t::*)(__VA_ARGS__) const>(&t::m)

void BindEnum(lua_State* L);
int BindErrorHandler(lua_State* L);
void BindPlugins(lua_State* L);
void BindLib(lua_State* L);

object VectorToTable(const std::vector<double>& vec);
std::vector<double> TableToVector(const object& table);

boost::thread_specific_ptr <lua_State> myL;

luatool::luatool()
{
	itsClearTextFormula = "<interpreted>";
	itsLogger = logger_factory::Instance()->GetLog("luatool");
	myL.reset();
}

luatool::~luatool()
{
}

void luatool::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("DUMMY")});

	Start();
}

void luatool::Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger_factory::Instance()->GetLog("luatoolThread #" + boost::lexical_cast<std::string> (threadIndex));

	InitLua(myTargetInfo);

	assert(myL.get());
	myThreadedLogger->Info("Calculating time " + static_cast<std::string> (myTargetInfo->Time().ValidDateTime()) + " level " + static_cast<std::string> (myTargetInfo->Level()));

	globals(myL.get())["logger"] = myThreadedLogger.get();

	BOOST_FOREACH(const std::string& luaFile, itsConfiguration->GetValueList("luafile"))
	{
		if (luaFile.empty())
		{
			continue;
		}

		myThreadedLogger->Info("Starting script " + luaFile);

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

	globals(L)["luatool"] = boost::ref(this);
	globals(L)["result"] = myTargetInfo;

	globals(L)["current_time"] = forecast_time(myTargetInfo->Time());
	globals(L)["current_level"] = level(myTargetInfo->Level());

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(forecast_time(myTargetInfo->Time()));

	globals(L)["hitool"] = h;
	globals(L)["kFloatMissing"] = kFloatMissing;

	itsLogger->Trace("luabind finished");

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
			itsLogger->Error(lua_tostring(myL.get(), -1));
			return false;
		}
	}
	catch (const error& e)
	{
		return false;
	}
	catch (const std::exception& e)
	{
		itsLogger->Error(e.what());
		return false;
	}

	return true;

}

int BindErrorHandler(lua_State* L)
{
    // log the error message
    luabind::object msg(luabind::from_stack( L, -1 ));
    std::ostringstream str;
    str << "lua> run-time error: " << msg;
    std::cout << str.str() << std::endl;

    // log the callstack
    std::string traceback = luabind::call_function<std::string>( luabind::globals(L)["debug"]["traceback"] );
    traceback = std::string("lua> ") + traceback;
    std::cout << traceback.c_str() << std::endl;

    // return unmodified error object
    return 1;
}

void BindEnum(lua_State* L)
{

	module(L)
	[
		class_<HPLevelType>("HPLevelType")
			.enum_("constants")
		[
			value("kUnknownLevel", kUnknownLevel),
			value("kGround", kHeight),
			value("kTopOfAtmosphere", kHeight),
			value("kPressure", kHeight),
			value("kMeanSea", kHeight),
			value("kAltitude", kHeight),
			value("kHeight", kHeight),
			value("kHybrid", kHybrid),
			value("kGndLayer", kGndLayer),
			value("kDepth", kDepth),
			value("kEntireAtmosphere", kEntireAtmosphere),
			value("kEntireOcean", kEntireOcean)
		]
		,
		class_<HPTimeResolution>("HPTimeResolution")
			.enum_("constants")
		[
			value("kUnknownTimeResolution", kUnknownTimeResolution),
			value("kHourResolution", kHourResolution),
			value("kMinuteResolution", kMinuteResolution)
		]
		,
		class_<HPFileType>("HPFileType")
			.enum_("constants")
		[
			value("kUnknownFile", kUnknownFile),
			value("kGRIB1", kGRIB1),
			value("kGRIB2", kGRIB2),
			value("kGRIB", kGRIB),
			value("kQueryData", kQueryData),
			value("kNetCDF", kNetCDF)
		]
		,
		class_<HPProjectionType>("HPProjectionType")
			.enum_("constants")
		[
			value("kUnknownProjection", kUnknownProjection),
			value("kLatLonProjection", kLatLonProjection),
			value("kRotatedLatLonProjection", kRotatedLatLonProjection),
			value("kStereographicProjection", kStereographicProjection)
		]
		,
		class_<HPScanningMode>("HPScanningMode")
			.enum_("constants")
		[
			value("kUnknownScanningMode", kUnknownScanningMode),
			value("kTopLeft", kTopLeft),
			value("kTopRight", kTopRight),
			value("kBottomLeft", kBottomLeft),
			value("kBottomRight", kBottomRight)
		]
		,
		class_<HPAggregationType>("HPAggregationType")
			.enum_("constants")
		[
			value("kUnknownAggregationType", kUnknownAggregationType),
			value("kAverage", kAverage),
			value("kAccumulation", kAccumulation),
			value("kMaximum", kMaximum),
			value("kMinimum", kMinimum),
			value("kDifference", kDifference)
		],
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
			value("kPlusMinusAreaModifier", kPlusMinusAreaModifier)
		],
		class_<HPGridType>("HPGridType")
			.enum_("constants")
		[
			value("kUnknownGridType", kUnknownGridType),
			value("kRegularGrid", kRegularGrid),
			value("kIrregularGrid", kIrregularGrid)
		],
		class_<HPParameterUnit>("HPParameterUnit")
			.enum_("constants")
		[
			value("kM", kM),
			value("kHPa", kHPa)
		]
	];
}

namespace info_wrapper
{
// These are convenience functions for accessing info class contents

bool SetValue(std::shared_ptr<info>& anInfo, int index, double value)
{
	return anInfo->Grid()->Value(--index, value);
}

double GetValue(std::shared_ptr<info>& anInfo, int index)
{
	return anInfo->Grid()->Value(--index);
}

size_t GetLocationIndex(std::shared_ptr<info> anInfo)
{
	return anInfo->LocationIndex()+1;
}

size_t GetTimeIndex(std::shared_ptr<info> anInfo)
{
	return anInfo->TimeIndex()+1;
}

size_t GetParamIndex(std::shared_ptr<info> anInfo)
{
	return anInfo->ParamIndex()+1;
}

size_t GetLevelIndex(std::shared_ptr<info> anInfo)
{
	return anInfo->LevelIndex()+1;
}

void SetLocationIndex(std::shared_ptr<info> anInfo, size_t theIndex)
{
	anInfo->LocationIndex(--theIndex);
}

void SetTimeIndex(std::shared_ptr<info> anInfo, size_t theIndex)
{
	anInfo->TimeIndex(--theIndex);
}

void SetParamIndex(std::shared_ptr<info> anInfo, size_t theIndex)
{
	anInfo->ParamIndex(--theIndex);
}

void SetLevelIndex(std::shared_ptr<info> anInfo, size_t theIndex)
{
	anInfo->LevelIndex(--theIndex);
}

void SetValues(info_t& anInfo, const object& table)
{
	std::vector<double> vals = TableToVector(table);
	
	if (vals.size() > 0 && vals.size() != anInfo->Data().Size())
	{
		std::cerr << "Error::luatool input table size is not the same as grid size: " << vals.size() << " vs " << anInfo->Data().Size() << std::endl;
	}
	else
	{
		anInfo->Data().Set(vals);
	}
}

std::vector<double> GetValues(info_t& anInfo)
{
	return anInfo->Data().Values();
}

point GetLatLon(info_t& anInfo, size_t theIndex)
{
	return anInfo->Grid()->LatLon(--theIndex);
}

void SetParam(info_t& anInfo, const param& par)
{

	param p = par;

	auto n = GET_PLUGIN(neons);

	long tableVersion = anInfo->Producer().TableVersion();

	if (tableVersion == kHPMissingInt)
	{
		auto prodinfo = n->NeonsDB().GetProducerDefinition(anInfo->Producer().Id());
		tableVersion = boost::lexical_cast<long> (prodinfo["no_vers"]);
	}

	p.GribTableVersion(tableVersion);

	long paramId = n->NeonsDB().GetGridParameterId(tableVersion, p.Name());

	if (paramId != -1)
	{
		p.GribIndicatorOfParameter(paramId);
	}

	anInfo->SetParam(p);

}


} // namespace info_wrapper

namespace hitool_wrapper
{
// The following functions are all wrappers for hitool:
// we cannot specify hitool functions directly in the lua binding
// because that leads to undefined symbols in plugins and that
// forces us to link luatool with hitool which is not nice!

object VerticalMaximumGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue, const object& lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalMaximum(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
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
			throw e;
		}
	}

	return object();
}

object VerticalMinimumGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue, const object& lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalMinimum(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
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
			throw e;
		}
	}

	return object();
}

object VerticalSumGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue, const object& lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalSum(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
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
			throw e;
		}
	}

	return object();
}

object VerticalAverageGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue, const object& lastLevelValue)
{
	try
	{
		return VectorToTable(h->VerticalAverage(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
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
			throw e;
		}
	}

	return object();
}

object VerticalCountGrid(std::shared_ptr<hitool> h, const param& theParams, const object& firstLevelValue, const object& lastLevelValue, const object& findValue)
{
	try
	{
		return VectorToTable(h->VerticalCount(theParams, TableToVector(firstLevelValue), TableToVector(lastLevelValue), TableToVector(findValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
		}
	}

	return object();
}

object VerticalCount(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue, double lastLevelValue, double findValue)
{
	try
	{
		return VectorToTable(h->VerticalCount(theParams, firstLevelValue, lastLevelValue, findValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
		}
	}

	return object();
}

object VerticalHeightGrid(std::shared_ptr<hitool> h, const param& theParam, const object& firstLevelValue, const object& lastLevelValue, const object& findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeight(theParam, TableToVector(firstLevelValue), TableToVector(lastLevelValue), TableToVector(findValue), findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
		}
	}

	return object();
}

object VerticalHeight(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue, double lastLevelValue, double findValue, size_t findNth)
{
	try
	{
		return VectorToTable(h->VerticalHeight(theParams, firstLevelValue, lastLevelValue, findValue, findNth));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
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
			throw e;
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
			throw e;
		}
	}

	return object();
}

object VerticalPlusMinusAreaGrid(std::shared_ptr<hitool> h, const param& theParams, const object& firstLevelValue, const object& lastLevelValue)
{
	try
	{
		return VectorToTable(h->PlusMinusArea(theParams, TableToVector(firstLevelValue), TableToVector(lastLevelValue)));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
		}
	}

	return object();
}

object VerticalPlusMinusArea(std::shared_ptr<hitool> h, const param& theParams, double firstLevelValue, double lastLevelValue)
{
	try
	{
		return VectorToTable(h->PlusMinusArea(theParams, firstLevelValue, lastLevelValue));
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
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

} // namespace hitool_wrapper

void BindLib(lua_State* L)
{
	module(L)
	[
		class_<himan::info, std::shared_ptr<himan::info>>("info")
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
			// These are local functions to luatool
			.def("SetParam", &info_wrapper::SetParam)
			.def("SetIndexValue", &info_wrapper::SetValue)
			.def("GetIndexValue", &info_wrapper::GetValue)
			.def("GetTimeIndex", &info_wrapper::GetTimeIndex)
			.def("GetParamIndex", &info_wrapper::GetParamIndex)
			.def("GetLevelIndex", &info_wrapper::GetLevelIndex)
			.def("SetTimeIndex", &info_wrapper::SetTimeIndex)
			.def("SetParamIndex", &info_wrapper::SetParamIndex)
			.def("SetLevelIndex", &info_wrapper::SetLevelIndex)
			.def("SetValues", &info_wrapper::SetValues)
			.def("GetValues", &info_wrapper::GetValues, return_stl_iterator)
			.def("GetLatLon", &info_wrapper::GetLatLon)
		,
		class_<grid, std::shared_ptr<grid>>("grid")
		,
		class_<regular_grid, grid, std::shared_ptr<regular_grid>>("regular_grid")
			.def(constructor<>())
			.def("ClassName", &grid::ClassName)
			.def("GetSize", &grid::Size)
			.def("GetNi", LUA_CMEMFN(size_t, regular_grid, Ni, void))
			.def("GetNj", LUA_CMEMFN(size_t, regular_grid, Nj, void))
			.def("GetDi", LUA_CMEMFN(double, regular_grid, Di, void))
			.def("GetDj", LUA_CMEMFN(double, regular_grid, Dj, void))
			.def("GetScanningMode", LUA_CMEMFN(HPScanningMode, regular_grid, ScanningMode, void))
			.def("GetProjection", LUA_CMEMFN(HPProjectionType, regular_grid, Projection, void))
			.def("GetAB", LUA_CMEMFN(std::vector<double>, regular_grid, AB, void))
			//.def("SetAB", LUA_MEMFN(void, regular_grid, AB, const std::vector<double>&))
			.def("GetBottomLeft", LUA_CMEMFN(point, regular_grid, BottomLeft, void))
			.def("SetBottomLeft", LUA_MEMFN(void, regular_grid, BottomLeft, const point&))
			.def("GetTopRight", LUA_CMEMFN(point, regular_grid, TopRight, void))
			.def("SetTopRight", LUA_MEMFN(void, regular_grid, BottomLeft, const point&))
			.def("GetFirstGridPoint", LUA_CMEMFN(point, regular_grid, FirstGridPoint, void))
			.def("GetLastGridPoint", LUA_CMEMFN(point, regular_grid, LastGridPoint, void))
		,
		class_<matrix<double>>("matrix")
			.def(constructor<size_t,size_t,size_t,double>())
			/*.def("Size", &matrix<double>::Size)
			.def("ClassName", &matrix<double>::ClassName)
			.def("Resize", LUA_MEMFN(void, matrix<double>, Resize, size_t, size_t, size_t))
			.def("GetValue", LUA_CMEMFN(double, matrix<double>, At, size_t))
			.def("GetValues", &matrix<double>::Values)
			.def("SetValue", LUA_MEMFN(bool, matrix<double>, Set, size_t, double))
			.def("SetValues", LUA_MEMFN(bool, matrix<double>, Set, const std::vector<double>&))
			.def("Fill", &matrix<double>::Fill)
			.def("GetMissingValue", LUA_CMEMFN(double, matrix<double>, MissingValue, void))
			.def("SetMissingValue", LUA_MEMFN(void, matrix<double>, MissingValue, double))
			.def("Clear", &matrix<double>::Clear)
			.def("IsMissing", LUA_CMEMFN(bool, matrix<double>, IsMissing, size_t))
			.def("MissingCount", &matrix<double>::MissingCount)*/
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
			.def("SetAggregation", LUA_MEMFN(void, param, Aggregation, const aggregation&))
		,
		class_<level>("level")
			.def(constructor<HPLevelType, double>())
			.def("ClassName", &level::ClassName)
			.def(tostring(self))
			.def("GetType", LUA_CMEMFN(HPLevelType, level, Type, void))
			.def("SetType", LUA_MEMFN(void, level, Type, HPLevelType))
			.def("GetValue", LUA_CMEMFN(double, level, Value, void))
			.def("SetValue", LUA_MEMFN(void, level, Value, double))
		,
		class_<raw_time>("raw_time")
			.def(constructor<const std::string&>())
			.def("ClassName", &raw_time::ClassName)
			.def("String", LUA_CMEMFN(std::string, raw_time, String, const std::string&))
			.def("Adjust", &raw_time::Adjust)
			.def("Empty", &raw_time::Empty)
		,
		class_<forecast_time>("forecast_time")
			.def(constructor<const raw_time&, const raw_time&>())
			.def("ClassName", &forecast_time::ClassName)
			.def("GetOriginDateTime", LUA_MEMFN(raw_time&, forecast_time, OriginDateTime, void))
			.def("GetValidDateTime", LUA_MEMFN(raw_time&, forecast_time, ValidDateTime, void))
			.def("SetOriginDateTime", LUA_MEMFN(void, forecast_time, OriginDateTime, const std::string&, const std::string&))
			.def("SetValidDateTime", LUA_MEMFN(void, forecast_time, ValidDateTime, const std::string&, const std::string&))
			.def("GetStep", LUA_CMEMFN(int, forecast_time, Step, void))
			.def("GetStepResolution", LUA_CMEMFN(HPTimeResolution, forecast_time, StepResolution, void))
			.def("SetStepResolution", LUA_MEMFN(void, forecast_time, StepResolution, HPTimeResolution))
		,
		class_<point>("point")
			.def(constructor<double, double>())
			.def("ClassName", &point::ClassName)
			.def("SetX", LUA_MEMFN(void, point, X, double))
			.def("SetY", LUA_MEMFN(void, point, Y, double))
			.def("GetX", LUA_CMEMFN(double, point, X, void))
			.def("GetY", LUA_CMEMFN(double, point, Y, void))
		,
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
			.def("Fatal", &logger::Fatal)
		,
		class_<aggregation>("aggregation")
			.def(constructor<HPAggregationType, HPTimeResolution, int>())
			.def("ClassName", &aggregation::ClassName)
			.def("GetType", LUA_CMEMFN(HPAggregationType, aggregation, Type, void))
			.def("SetType", LUA_MEMFN(void, aggregation, Type, HPAggregationType))
			.def("GetTimeResolution", LUA_CMEMFN(HPTimeResolution, aggregation, TimeResolution, void))
			.def("SetTimeResolution", LUA_MEMFN(void, aggregation, TimeResolution, HPTimeResolution))
			.def("GetTimeResolutionValue", LUA_CMEMFN(int, aggregation, TimeResolutionValue, void))
			.def("SetTimeResolutionValue", LUA_MEMFN(void, aggregation, TimeResolutionValue, int))
		,
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
			.def("GetValue", &plugin_configuration::GetValueList)
			.def("Exists", &plugin_configuration::Exists)
		,
		class_<lcl_t>("lcl_t")
			.def(constructor<>())
			.def_readwrite("T", &lcl_t::T)
			.def_readwrite("P", &lcl_t::P)
			.def_readwrite("Q", &lcl_t::Q)
		,
		// util namespace
		def("Filter2D", &util::Filter2D)
		,
		// metutil namespace
		def("LCL_", &metutil::LCL_)
		,
		def("Es_", &metutil::Es_)
		,
		def("Gammas_", &metutil::Gammas_)
		,
		def("Gammaw_", &metutil::Gammaw_)
		,
		def("MixingRatio_", &metutil::MixingRatio_)
		,
		def("MoistLift_", &metutil::MoistLift_)
		,
		def("DryLift_", &metutil::DryLift_)
	];

}

void BindPlugins(lua_State* L)
{
	module(L) [
		class_<compiled_plugin_base>("compiled_plugin_base")
			.def(constructor<>())
			.def("WriteToFile", LUA_CMEMFN(void, compiled_plugin_base, WriteToFile, const info&))
		,
		class_<luatool, compiled_plugin_base>("luatool")
			.def(constructor<>())
			.def("ClassName", &luatool::ClassName)
			.def("FetchRaw", &luatool::FetchRaw)
			.def("Fetch", &luatool::Fetch)
		,
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
			.def("VerticalValueGrid", &hitool_wrapper::VerticalValueGrid)
			.def("VerticalValue", &hitool_wrapper::VerticalValue)
			.def("VerticalPlusMinusAreaGrid", &hitool_wrapper::VerticalPlusMinusAreaGrid)
			.def("VerticalPlusMinusArea", &hitool_wrapper::VerticalPlusMinusArea)
			.def("SetHeightUnit", &hitool_wrapper::SetHeightUnit)
			.def("GetHeightUnit", &hitool_wrapper::GetHeightUnit)
	];
}

void luatool::Run(info_t myTargetInfo, unsigned short threadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		ResetNonLeadingDimension(myTargetInfo);

		while (AdjustNonLeadingDimension(myTargetInfo))
		{
			Calculate(myTargetInfo, threadIndex);

			if (itsConfiguration->StatisticsEnabled())
			{
				itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data().MissingCount());
				itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data().Size());
			}
		}
	}
}

void luatool::Finish() const
{

	if (itsConfiguration->StatisticsEnabled())
	{
		itsTimer->Stop();
		itsConfiguration->Statistics()->AddToProcessingTime(itsTimer->GetTime());
	}
}

std::shared_ptr<info> luatool::FetchRaw(const forecast_time& theTime, const level& theLevel, const param& theParam) const
{
	return compiled_plugin_base::Fetch(theTime,theLevel,theParam,false);
}

luabind::object luatool::Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam) const
{
	auto x = compiled_plugin_base::Fetch(theTime,theLevel,theParam,false);

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
	BOOST_FOREACH(const double& val, vec)
	{
		ret[++i] = val;
		
/*		"Lua tables make no distinction between a table value being nil and the corresponding key not existing in the table"
		if (val == kFloatMissing)
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
	
	// TODO: preallocate vector to correct size, but from where can I have
	// table size!?
	
	std::vector<double> ret;

	for (luabind::iterator iter(table), end; iter != end; ++iter)
	{
		try
		{
			ret.push_back(object_cast<double>(*iter));
		}
		catch (cast_failed& e)
		{
			ret.push_back(kFloatMissing);
		}
	}
	
	return ret;
}
