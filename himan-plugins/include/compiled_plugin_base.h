/**
 * @file compiled_plugin_base.h
 *
 */

#ifndef COMPILED_PLUGIN_BASE_H
#define COMPILED_PLUGIN_BASE_H

#include "compiled_plugin.h"
#include "plugin_configuration.h"
#include <boost/iterator/zip_iterator.hpp>
#include <mutex>
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

#define ACTUAL_MACRO_C(x) assert(x);
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
class compiled_plugin_base
{
   public:
	compiled_plugin_base();
	inline virtual ~compiled_plugin_base() {}
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

	virtual void WriteToFile(const info& targetInfo, write_options opts = write_options());

   protected:
	virtual std::string ClassName() const { return "himan::plugin::compiled_plugin_base"; }
	/**
	 * @brief Set primary dimension
	 *
	 * Functionality of this function could be replaced just by exposing the
	 * variables to all child classes but as all other access to this
	 * variable is through functions (ie adjusting the dimensions), it is
	 * better not to allow direct access to have some consistency.
	 */

	void PrimaryDimension(HPDimensionType thePrimaryDimension);
	HPDimensionType PrimaryDimension() const;

	/**
	 * @brief Copy AB values from source to dest info
	 */

	bool SetAB(const info_t& myTargetInfo, const info_t& sourceInfo);

	/**
	 * @brief Distribute work equally to all threads
	 * @param myTargetInfo
	 * @return
	 */

	virtual bool Next(info& myTargetInfo);

	/**
	 * @brief Distribute work equally to all threads so that each calling
	 * thread will have access to all levels.
	 *
	 * @param myTargetInfo
	 * @return
	 */

	virtual bool NextExcludingLevel(info& myTargetInfo);

	/**
	 * @brief Entry point for threads.
	 *
	 * This function will handle jobs (ie. times, levels to process) to each thread.
	 *
	 * @param threadIndex
	 */

	virtual void Run(unsigned short threadIndex);

	/**
	 * @brief Set target params
	 *
	 * Function will fetch grib1 definitions from neons if necessary, and will
	 * create the data backend for the resulting info.
	 *
	 * @param params vector of target parameters
	 */

	void SetParams(std::vector<param>& params);

	/**
	 * @brief Set target params
	 *
	 * Syntactic sugar for SetParams(vector<param>&)
	 *
	 * @param list of params
	 */

	void SetParams(std::initializer_list<param> params);

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

	virtual void Calculate(info_t myTargetInfo, unsigned short threadIndex);

	/**
	 * @brief Start threaded calculation
	 */

	virtual void Start();

#ifdef HAVE_CUDA
	/**
	 * @brief Unpack grib data
	 *
	 * This function should be called if the source data is packed but cuda cannot
	 * be used in calculation. If the calculation is done with cuda, the unpacking
	 * is also made there.
	 *
	 * @param infos List of shared_ptr<info> 's that have packed data
	 */

	void Unpack(std::initializer_list<info_t> infos);

	/**
	 * @brief Copy data from info_simple to actual info, clear memory and
	 * put the result to cache (optionally).
	 *
	 * Function has two slightly different calling types:
	 * 1) A parameter has been calculated on GPU and the results have been stored
	 *	to info_simple. This function will copy data to info and release the
	 *	page-locked memory of info_simple. In this calling type the resulting
	 *	data is not written to cache at this point, because it will be written
	 *	to cache when it is written to disk.
	 *
	 * 2) A source parameter for a calculation has been read in packed format from
	 *	grib and has been unpacked at GPU. This function will copy the unpacked
	 *	source data from info_simple to info, release page-locked memory of
	 *	info_simple and clear the packed data array from info. Then it will also
	 *	write the source data to cache since it might be needed by some other
	 *	plugin.
	 *
	 * @param anInfo Target info
	 * @param aSimpleInfo Source info_simple
	 * @param writeToCache If true info will be written to cache
	 */

	void CopyDataFromSimpleInfo(const info_t& anInfo, info_simple* aSimpleInfo, bool writeToCache);

#endif

	/**
	 * @brief Compare a number of grids to see if they are equal.
	 *
	 * @param grids List of grids
	 * @return True if all are equal, else false
	 */

	bool CompareGrids(std::initializer_list<std::shared_ptr<grid>> grids) const;

	/**
	 * @brief Syntactic sugar: simple function to check if any of the arguments is a missing value
	 *
	 * @param values List of doubles
	 * @return True if any of the values is missing value, otherwise false
	 */

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
	 * @return shared_ptr<info> on success, null-pointer if data not found
	 */

	virtual info_t Fetch(const forecast_time& theTime, const level& theLevel, const himan::params& theParams,
	                     const forecast_type& theType = forecast_type(kDeterministic), bool returnPacked = false) const;

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
	 * @return shared_ptr<info> on success, un-initialized shared_ptr if data not found
	 */

	virtual info_t Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                     const forecast_type& theType = forecast_type(kDeterministic), bool returnPacked = false) const;

	/**
	 * @brief Initialize compiled_plugin_base and set internal state.
	 *
	 * @param conf
	 */

	virtual void Init(const std::shared_ptr<const plugin_configuration> conf);

	/**
	 * @brief Run threads through all dimensions in the most effective way.
	 */

	void RunAll(info_t myTargetInfo, unsigned short threadIndex);

	/**
	 * @brief Run threads so that each thread will get one time step.
	 *
	 * This limits the number of threads to the number of time steps, but it is
	 * the preferred way when f.ex. levels need to be accessed sequentially (hybrid_height).
	 */

	void RunTimeDimension(info_t myTargetInfo, unsigned short threadIndex);

	virtual void AllocateMemory(info myTargetInfo);
	virtual void DeallocateMemory(info myTargetInfo);

   protected:
	info_t itsInfo;
	std::shared_ptr<const plugin_configuration> itsConfiguration;
	timer itsTimer;
	short itsThreadCount;
	bool itsDimensionsRemaining;

   private:
	logger itsBaseLogger;
	bool itsPluginIsInitialized;
	HPDimensionType itsPrimaryDimension;
};

}  // namespace plugin
}  // namespace himan

#endif /* COMPILED_PLUGIN_BASE_H */
