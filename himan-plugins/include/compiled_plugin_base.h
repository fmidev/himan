/**
 * @file compiled_plugin_base.h
 *
 */

#ifndef COMPILED_PLUGIN_BASE_H
#define COMPILED_PLUGIN_BASE_H

#include "compiled_plugin.h"
#include "info.h"
#include "plugin_configuration.h"
#include "timer.h"
#include <boost/iterator/zip_iterator.hpp>
#include <write_options.h>

template <class... Conts>
auto zip_range(Conts&... conts)
    -> decltype(boost::make_iterator_range(boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
                                           boost::make_zip_iterator(boost::make_tuple(conts.end()...))))
{
	return {boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
	        boost::make_zip_iterator(boost::make_tuple(conts.end()...))};
}

/*
 * Really nice pre-processor macros here
 * What all this does is it'll change this
 *
 * LOCKSTEP(info1,info2,...)
 *
 * to this
 *
 * assert(info1); assert(info2);
 * for (info1->ResetLocation(), info2->ResetLocation(); info->NextLocation() && info2->NextLocation();)
 *
 * The current maximum number of infos is 14 (thanks preform_pressure).
 */

#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define FULLY_EXPAND_PROPERTIES(count, ...) \
	ASSERT##count(__VA_ARGS__) for (RESET##count(__VA_ARGS__); NEXT##count(__VA_ARGS__);)

#define SEMI_EXPAND_PROPERTIES(count, ...) FULLY_EXPAND_PROPERTIES(count, __VA_ARGS__)

// this is the actual function called to expand properties
#define LOCKSTEP(...) SEMI_EXPAND_PROPERTIES(VA_NARGS(__VA_ARGS__), __VA_ARGS__)

// Here are the different expansions of MACA and MACB (the names are arbitrary)
#define ACTUAL_MACRO_A(x) x->NextLocation()
#define NEXT1(a) ACTUAL_MACRO_A(a)
#define NEXT2(a, b) NEXT1(a) && ACTUAL_MACRO_A(b)
#define NEXT3(a, b, c) NEXT2(a, b) && ACTUAL_MACRO_A(c)
#define NEXT4(a, b, c, d) NEXT3(a, b, c) && ACTUAL_MACRO_A(d)
#define NEXT5(a, b, c, d, e) NEXT4(a, b, c, d) && ACTUAL_MACRO_A(e)
#define NEXT6(a, b, c, d, e, f) NEXT5(a, b, c, d, e) && ACTUAL_MACRO_A(f)
#define NEXT7(a, b, c, d, e, f, g) NEXT6(a, b, c, d, e, f) && ACTUAL_MACRO_A(g)
#define NEXT8(a, b, c, d, e, f, g, h) NEXT7(a, b, c, d, e, f, g) && ACTUAL_MACRO_A(h)
#define NEXT9(a, b, c, d, e, f, g, h, i) NEXT8(a, b, c, d, e, f, g, h) && ACTUAL_MACRO_A(i)
#define NEXT10(a, b, c, d, e, f, g, h, i, j) NEXT9(a, b, c, d, e, f, g, h, i) && ACTUAL_MACRO_A(j)
#define NEXT11(a, b, c, d, e, f, g, h, i, j, k) NEXT10(a, b, c, d, e, f, g, h, i, j) && ACTUAL_MACRO_A(k)
#define NEXT12(a, b, c, d, e, f, g, h, i, j, k, l) NEXT11(a, b, c, d, e, f, g, h, i, j, k) && ACTUAL_MACRO_A(l)
#define NEXT13(a, b, c, d, e, f, g, h, i, j, k, l, m) NEXT12(a, b, c, d, e, f, g, h, i, j, k, l) && ACTUAL_MACRO_A(m)
#define NEXT14(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
	NEXT13(a, b, c, d, e, f, g, h, i, j, k, l, m) && ACTUAL_MACRO_A(n)

#define ACTUAL_MACRO_B(x) x->ResetLocation()
#define RESET1(a) ACTUAL_MACRO_B(a)
#define RESET2(a, b) RESET1(a), ACTUAL_MACRO_B(b)
#define RESET3(a, b, c) RESET2(a, b), ACTUAL_MACRO_B(c)
#define RESET4(a, b, c, d) RESET3(a, b, c), ACTUAL_MACRO_B(d)
#define RESET5(a, b, c, d, e) RESET4(a, b, c, d), ACTUAL_MACRO_B(e)
#define RESET6(a, b, c, d, e, f) RESET5(a, b, c, d, e), ACTUAL_MACRO_B(f)
#define RESET7(a, b, c, d, e, f, g) RESET6(a, b, c, d, e, f), ACTUAL_MACRO_B(g)
#define RESET8(a, b, c, d, e, f, g, h) RESET7(a, b, c, d, e, f, g), ACTUAL_MACRO_B(h)
#define RESET9(a, b, c, d, e, f, g, h, i) RESET8(a, b, c, d, e, f, g, h), ACTUAL_MACRO_B(i)
#define RESET10(a, b, c, d, e, f, g, h, i, j) RESET9(a, b, c, d, e, f, g, h, i), ACTUAL_MACRO_B(j)
#define RESET11(a, b, c, d, e, f, g, h, i, j, k) RESET10(a, b, c, d, e, f, g, h, i, j), ACTUAL_MACRO_B(k)
#define RESET12(a, b, c, d, e, f, g, h, i, j, k, l) RESET11(a, b, c, d, e, f, g, h, i, j, k), ACTUAL_MACRO_B(l)
#define RESET13(a, b, c, d, e, f, g, h, i, j, k, l, m) RESET12(a, b, c, d, e, f, g, h, i, j, k, l), ACTUAL_MACRO_B(m)
#define RESET14(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
	RESET13(a, b, c, d, e, f, g, h, i, j, k, l, m), ACTUAL_MACRO_B(n)

#define ACTUAL_MACRO_C(x) ASSERT(x);
#define ASSERT1(a) ACTUAL_MACRO_C(a)
#define ASSERT2(a, b) ASSERT1(a) ACTUAL_MACRO_C(b)
#define ASSERT3(a, b, c) ASSERT2(a, b) ACTUAL_MACRO_C(c)
#define ASSERT4(a, b, c, d) ASSERT3(a, b, c) ACTUAL_MACRO_C(d)
#define ASSERT5(a, b, c, d, e) ASSERT4(a, b, c, d) ACTUAL_MACRO_C(e)
#define ASSERT6(a, b, c, d, e, f) ASSERT5(a, b, c, d, e) ACTUAL_MACRO_C(f)
#define ASSERT7(a, b, c, d, e, f, g) ASSERT6(a, b, c, d, e, f) ACTUAL_MACRO_C(g)
#define ASSERT8(a, b, c, d, e, f, g, h) ASSERT7(a, b, c, d, e, f, g) ACTUAL_MACRO_C(h)
#define ASSERT9(a, b, c, d, e, f, g, h, i) ASSERT8(a, b, c, d, e, f, g, h) ACTUAL_MACRO_C(i)
#define ASSERT10(a, b, c, d, e, f, g, h, i, j) ASSERT9(a, b, c, d, e, f, g, h, i) ACTUAL_MACRO_C(j)
#define ASSERT11(a, b, c, d, e, f, g, h, i, j, k) ASSERT10(a, b, c, d, e, f, g, h, i, j) ACTUAL_MACRO_C(k)
#define ASSERT12(a, b, c, d, e, f, g, h, i, j, k, l) ASSERT11(a, b, c, d, e, f, g, h, i, j, k) ACTUAL_MACRO_C(l)
#define ASSERT13(a, b, c, d, e, f, g, h, i, j, k, l, m) ASSERT12(a, b, c, d, e, f, g, h, i, j, k, l) ACTUAL_MACRO_C(m)
#define ASSERT14(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
	ASSERT13(a, b, c, d, e, f, g, h, i, j, k, l, m) ACTUAL_MACRO_C(n)

namespace himan
{
namespace plugin
{
// Decide the strategegy how work is distributed to threads. Basically the smaller the number, the less control
// there is to be done for the thread itself (most of the work is done in the functions of this superclass).
// So aim for smaller enum numbers unless a good reason to do otherwise.
// Enum value naming is such that the dimension(s) in the enum name is given to thread, and dimension(s) not
// mentioned are the responsibility of the thread.
enum class ThreadDistribution
{
	// Calculations between 'dimensions' are not dependent, each time/level/thread combination is given randomly to
	// requesting thread. Thread will controll no iterators by itself. This is the same as
	// "kThreadForForecastTypeAndTimeAndLevel"
	kThreadForAny = 0,
	// Each requesting thread will get their own forecast type and forecast time to process
	// (eg. hybrid_height, stability, cape). Thread will control level iterator.
	kThreadForForecastTypeAndTime,
	// Each requesting thread will get their own forecast type and level to process. Thread will control time
	// iterator.
	kThreadForForecastTypeAndLevel,
	// Each requesting thread will get their own forecast time and level to process. Thread will control forecast type
	// iterator.
	kThreadForTimeAndLevel,
	// Each requesting thread will get their own forecast type to process. Thread will control time and level
	// iterators.
	kThreadForForecastType,
	// Each requesting thread will get their own time to process. Thread will control forecast type and level
	// iterators.
	kThreadForTime,
	// Each requesting thread will get their own level to process. Thread will control forecast type and time
	// iterators.
	kThreadForLevel
};

class compiled_plugin_base
{
   public:
	compiled_plugin_base() = default;
	inline virtual ~compiled_plugin_base() = default;
	compiled_plugin_base(const compiled_plugin_base& other) = delete;
	compiled_plugin_base& operator=(const compiled_plugin_base& other) = delete;

	/**
	 * @brief Write plugin contents to file.
	 *
	 * Function will determine whether it needs to write whole info or just active
	 * parts of it. Function will preserve iterator positions.
	 *
	 * @param targetInfo info-class instance holding the data
	 */

	template <typename T>
	void WriteToFile(const std::shared_ptr<info<T>> targetInfo, write_options opts = write_options());

	virtual void WriteToFile(const std::shared_ptr<info<double>> targetInfo, write_options opts = write_options());
	virtual void WriteToFile(const std::shared_ptr<info<float>> targetInfo, write_options opts = write_options());

   protected:
	virtual std::string ClassName() const
	{
		return "himan::plugin::compiled_plugin_base";
	}

	/**
	 * @brief Copy AB values from source to dest info
	 */

	template <typename T>
	bool SetAB(const std::shared_ptr<info<T>>& myTargetInfo, const std::shared_ptr<info<T>>& sourceInfo);
	bool SetAB(const std::shared_ptr<info<double>>& myTargetInfo, const std::shared_ptr<info<double>>& sourceInfo);

	/**
	 * @brief Distribute work equally to all threads
	 * @param myTargetInfo
	 * @return
	 */

	template <typename T>
	bool Next(info<T>& myTargetInfo);

	/**
	 * @brief Entry point for threads.
	 *
	 * This function will handle jobs (ie. times, levels to process) to each thread.
	 *
	 * @param threadIndex
	 */

	template <typename T>
	void Run(std::shared_ptr<info<T>> myTargetInfo, unsigned short threadIndex);

	/**
	 * @brief Set target params
	 *
	 * Function will fetch parameter definitions from database if argument 'paramInformationExists'
	 * is false, and will create the data backend for the resulting info.
	 *
	 * @param params vector of target parameters
	 */

	void SetParams(std::vector<param>& params, bool paramInformationExists = false);
	void SetParams(std::vector<param>& params, const std::vector<level>& levels, bool paramInformationExists = false);

	/**
	 * @brief Set target params
	 *
	 * Syntactic sugar for SetParams(vector<param>&)
	 *
	 * @param list of params
	 */

	void SetParams(std::initializer_list<param> params, bool paramInformationExists = false);
	void SetParams(std::initializer_list<param> params, std::initializer_list<level> levels,
	               bool paramInformationExists = false);

	/**
	 * @brief Record timing info and write info contents to disk
	 */

	virtual void Finish();

	/**
	 * @brief Top level entry point for per-thread calculation
	 *
	 * This function will abort since the plugins must define the processing
	 * themselves.
	 *
	 * @param myTargetInfo A threads own info instance
	 * @param threadIndex
	 */

	virtual void Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex);
	virtual void Calculate(std::shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex);

	/**
	 * @brief Start threaded calculation
	 */

	template <typename T>
	void Start();

	virtual void Start();

	/**
	 * @brief Syntactic sugar: simple function to check if any of the arguments is a missing value
	 *
	 * @param values List of doubles
	 * @return True if any of the values is missing value, otherwise false
	 */

	template <typename T>
	bool IsMissingValue(std::initializer_list<T> values) const;
	bool IsMissingValue(std::initializer_list<double> values) const;

	/**
	 * @brief Fetch source data with given requirements
	 *
	 * Overcoat for Fetch(const forecast_time&, const level&, const param&, bool useCuda)
	 *
	 * @param theTime Wanted forecast time
	 * @param theLevel Wanted level
	 * @param theParams List of parameters (in a vector)
	 * @param returnPacked Flag for returning data either packed or unpacked
	 * @param theType
	 * @return std::shared_ptr<info> on success, null-pointer if data not found
	 */

	template <typename T>
	std::shared_ptr<info<T>> Fetch(const forecast_time& theTime, const level& theLevel, const himan::params& theParams,
	                               const forecast_type& theType = forecast_type(kDeterministic),
	                               bool returnPacked = false) const;

	virtual std::shared_ptr<info<double>> Fetch(const forecast_time& theTime, const level& theLevel,
	                                            const himan::params& theParams,
	                                            const forecast_type& theType = forecast_type(kDeterministic),
	                                            bool returnPacked = false) const;

	/**
	 * @brief Fetch source data with given requirements
	 *
	 * Will throw if an error occurs in data retrieval (file not found does not count as an
	 * error in this case).
	 *
	 * Data can be fetched and returned in four different ways
	 *
	 *                        FETCH
	 *                    packed unpacked
	 * RETURN   packed       1      X
	 * RETURN unpacked       2      3
	 *
	 * Case 1)
	 * Data is fetched packed, and is returned packed. This is most useful for cuda-plugins which unpack the data in
	 * their cuda kernel.
	 *
	 * Command line option --no-cuda-packing is NOT set, and flag returnPacked = true
	 *
	 * Case 2)
	 * Data is fetched packed but it is unpacked by himan in cuda before returning. This is most useful for regular
	 * plugins which need the
	 * data unpacked. Unpacking is done in cuda since it is faster that CPU -based unpacking.
	 *
	 * Command line option --no-cuda-packing is NOT set, and flag returnPacked = false
	 *
	 * Case 3)
	 * Data is fetched unpacked and returned unpacked. This is how himan used to function with CPU-plugins. This mode
	 * has been superseded
	 * by Case 2)
	 *
	 * Command line option --no-cuda-packing is set, flag returnPacked is ignored
	 *
	 * Case X)
	 * This case is actually impossible and not realistic so it is not implemented, and Fetch() will fallback to case 1)
	 *
	 * @param theTime Wanted forecast time
	 * @param theLevel Wanted level
	 * @param theParams List of parameters (in a vector)
	 * @param returnPacked Flag for returning data either packed or unpacked
	 * @param theType
	 * @return std::shared_ptr<info> on success, un-initialized std::shared_ptr if data not found
	 */

	template <typename T>
	std::shared_ptr<info<T>> Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                               const forecast_type& theType = forecast_type(kDeterministic),
	                               bool returnPacked = false) const;

	virtual std::shared_ptr<info<double>> Fetch(const forecast_time& theTime, const level& theLevel,
	                                            const param& theParam,
	                                            const forecast_type& theType = forecast_type(kDeterministic),
	                                            bool returnPacked = false) const;

	template <typename T>
	std::shared_ptr<info<T>> Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                               const forecast_type& theType, const std::vector<std::string>& geomNames,
	                               const producer& sourceProd, bool returnPacked = false) const;

	std::shared_ptr<info<double>> Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                                    const forecast_type& theType, const std::vector<std::string>& geomNames,
	                                    const producer& sourceProd, bool returnPacked = false) const;

	/**
	 * @brief Initialize compiled_plugin_base and set internal state.
	 *
	 * @param conf
	 */

	virtual void Init(const std::shared_ptr<const plugin_configuration> conf);

	template <typename T>
	void AllocateMemory(info<T> myTargetInfo);

	template <typename T>
	void DeallocateMemory(info<T> myTargetInfo);

   protected:
	void SetInitialIteratorPositions();
	void SetThreadCount();

	std::shared_ptr<const plugin_configuration> itsConfiguration;
	timer itsTimer = timer();
	short itsThreadCount = -1;
	bool itsDimensionsRemaining = true;
	param_iter itsParamIterator;
	level_iter itsLevelIterator;
	time_iter itsTimeIterator;
	forecast_type_iter itsForecastTypeIterator;
	ThreadDistribution itsThreadDistribution = ThreadDistribution::kThreadForAny;

	std::vector<std::pair<std::string, HPWriteStatus>> itsWriteStatuses;

   private:
	logger itsBaseLogger = logger("compiled_plugin_base");
	bool itsPluginIsInitialized = false;
	/**
	 * Variable will hold the actual level-param pairs that plugin is using.
	 * Because Himan will create a cartesian product of all dimensions, in
	 * some cases (like stability plugin) we only want to calculate certain
	 * parameters for certain level. We do not want to allocate memory for
	 * those level-param combinations that we don't actually calculate.
	 */
	std::vector<std::pair<level, param>> itsLevelParams;
};

}  // namespace plugin
}  // namespace himan

#endif /* COMPILED_PLUGIN_BASE_H */
