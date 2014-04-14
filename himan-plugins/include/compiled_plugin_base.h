/**
 * @file compiled_plugin_base.h
 *
 * @date Jan 15, 2013
 * @author partio
 */

#ifndef COMPILED_PLUGIN_BASE_H
#define COMPILED_PLUGIN_BASE_H

//#include <NFmiGrid.h>
#include "compiled_plugin.h"
#include "plugin_configuration.h"
#include <mutex>

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

protected:

	virtual std::string ClassName() const { return "himan::plugin::compiled_plugin_base"; }

	/**
	 * @brief Interpolates value to point, or gets the value directly if grids are equal
	 *
	 * Function will try get the value without interpolation: if parameter gridsAreEqual is true,
	 * the value at given grid point is returned instantly. If not, function will check if the
	 * point projected from targetGrid to sourceGrid is actually the same point. If so, the
	 * value of that gridpoint is returned. As a last resort to do the interpolation.
	 *
	 * @param targetGrid Target grid where the point-to-interpolated is picked
	 * @param sourceGrid Source grid where the latlon point is projected, and the value fetched
	 * @param gridsAreEqual If the definitions of these two grids are equal, do no interpolation
	 * @param value The interpolated value
	 *
	 * @return Always true if no interpolation is made, otherwise returns the value given by newbase interpolation function
	 */

	bool InterpolateToPoint(std::shared_ptr<const NFmiGrid> targetGrid, std::shared_ptr<NFmiGrid> sourceGrid, bool gridsAreEqual, double& value);

	/**
	 * @brief Set leading dimension
	 *
	 * Functionality of this function could be replaced just by exposing the variable
	 * itsLeadingDimension to all child classes but as all other access to this
	 * variable is through functions (ie adjusting the dimensions), it is
	 * better not to allow direct access to have some consistency.
	 */

	void Dimension(HPDimensionType theLeadingDimension)
	{
		itsLeadingDimension = theLeadingDimension;
	}

	HPDimensionType Dimension() const
	{
		return itsLeadingDimension;
	}

	/**
	 * @brief Advance leading dimension (time or level) by one, called by threads
	 *
	 * This function is protected with a mutex as it is responsible for distributing
	 * time steps or levels for processing to all calling threads.
	 *
     * @param myTargetInfo Threads own copy of target info
     * @return True if thread has more items to process
     */

	bool AdjustLeadingDimension(std::shared_ptr<info> myTargetInfo);

	/**
	 * @brief Adjust non-leading dimension (time of level) by one, called by threads
	 *
	 * This function is not protected with a mutex since all threads have exclusive
	 * access to their own info class instances' non-leading dimension. It is
	 * implemented in base class however because the information what is the
	 * leading dimension is located here.
	 *
     * @param myTargetInfo Threads own copy of target info
     * @return
     */

	bool AdjustNonLeadingDimension(std::shared_ptr<info> myTargetInfo);

	
	void ResetNonLeadingDimension(std::shared_ptr<info> myTargetInfo);
 
	/**
	 * @brief Copy AB values from source to dest info
	 */

	bool SetAB(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> sourceInfo);

	/**
	 * @brief Swap data ordering in a grid
	 *
	 */

	bool SwapTo(std::shared_ptr<info> myTargetInfo, HPScanningMode targetScanningMode);

	/**
	 * @brief Write plugin contents to file.
	 *
	 * Function will determine whether it needs to write whole info or just active
	 * parts of it. Function will preserve iterator positions.
	 *
	 * @param targetInfo info-class instance holding the data
	 */

	void WriteToFile(std::shared_ptr<const info> targetInfo);

	/**
	 * @brief Determine if cuda can be used in this thread, and if so
	 * set the environment.
	 *
	 * @param conf Plugin configuration
	 * @param threadIndex Thread index number, starting from 1
	 */
	
	bool GetAndSetCuda(std::shared_ptr<const configuration> conf, int threadIndex);

	/**
	 * @brief Reset GPU card state
     */
	
	void ResetCuda() const;

	/**
	 * @brief Entry point for threads.
	 *
	 * This function will handle jobs (ie. times, levels to process) to each thread.
	 * 
     * @param myTargetInfo A threads own info instance
     * @param threadIndex 
     */
	
	virtual void Run(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);

	/**
	 * @brief Initialize compiled_plugin_base and set internal state.
	 *
     * @param conf
     */

	virtual void Init(std::shared_ptr<const plugin_configuration> conf);

	/**
	 * @brief Set target params
	 *
	 * Function will fetch grib1 definitions from neons if necessary, and will
	 * create the data backend for the resulting info.
	 *
     * @param params vector of target parameters
     */

	virtual void SetParams(std::vector<param>& params);


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
	
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex) ;

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

	void Unpack(std::initializer_list<std::shared_ptr<info>> infos);

	/**
	 * @brief Copy data from info_simple to actual info, clear memory and 
	 * put the result to cache (optionally).
	 *
	 * Function has two slightly different calling types:
	 * 1) A parameter has been calculated on GPU and the results have been stored
	 *    to info_simple. This function will copy data to info and release the
	 *    page-locked memory of info_simple. In this calling type the resulting
	 *    data is not written to cache at this point, because it will be written
	 *    to cache when it is written to disk.
	 *
	 * 2) A source parameter for a calculation has been read in packed format from
	 *    grib and has been unpacked at GPU. This function will copy the unpacked
	 *    source data from info_simple to info, release page-locked memory of
	 *    info_simple and clear the packed data array from info. Then it will also
	 *    write the source data to cache since it might be needed by some other
	 *    plugin.
	 * 
     * @param anInfo Target info
     * @param aSimpleInfo Source info_simple
	 * @param writeToCache If true info will be written to cache
     */
	
	void CopyDataFromSimpleInfo(std::shared_ptr<info> anInfo, info_simple* aSimpleInfo, bool writeToCache);

#endif

	/**
	 * @brief Compare a number of grids to see if they are equal.
	 *
     * @param grids List of grids
     * @return True if all are equal, else false
     */
	
	bool CompareGrids(std::initializer_list<std::shared_ptr<grid>> grids);

	/**
	 * @brief Syntactic sugar: simple function to check if any of the arguments is a missing value
	 *
	 * @param values List of doubles
	 * @return True if any of the values is missing value (kFloatMissing), otherwise false
	 */

	bool IsMissingValue(std::initializer_list<double> values);

	std::shared_ptr<info> itsInfo;
	std::shared_ptr<const plugin_configuration> itsConfiguration;
	short itsThreadCount;

private:
	std::unique_ptr<timer> itsTimer;
	std::unique_ptr<logger> itsBaseLogger;
	bool itsPluginIsInitialized;
	HPDimensionType itsLeadingDimension;

};

} // namespace plugin
} // namespace himan

#endif /* COMPILED_PLUGIN_BASE_H */
