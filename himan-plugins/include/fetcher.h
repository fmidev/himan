/**
 * @file fetcher.h
 *
 * @date Nov 21, 2012
 * @author partio
 *
 * Class purpose is to server as an intermediator between a
 * compiled plugin asking for data and the different plugins
 * serving the different aspects of the data.
 *
 * In other words when a plugin asks for data for a parameter,
 * this plugin will return the data.
 *
 * 1) check cache if the data exists there; if so return it
 * 2) check auxiliary files specified at command line for data
 * 3) check neons for data
 * 4) fetch data if found
 * 5) store data to cache
 * 6) return data to caller
 *
 */

#ifndef FETCHER_H
#define FETCHER_H

#include "auxiliary_plugin.h"
#include "plugin_configuration.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{

class fetcher : public auxiliary_plugin
{
public:
	fetcher();

	virtual ~fetcher() {}

	fetcher(const fetcher& other) = delete;
	fetcher& operator=(const fetcher& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::plugin::fetcher";
	}

	virtual HPPluginClass PluginClass() const
	{
		return kAuxiliary;
	}

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(1, 1);
	}

	/**
	 * @brief Multi-param overcoat for the other Fetch() function
	 *
	 * Will return the first param found.
	 *
	 * @param config Plugin configuration
	 * @param requestedValidTime
	 * @param requestedLevel
	 * @param requestedParams List of wanted params
	 * @param readPackedData
	 * @return Data for first param found.
	 */

	std::shared_ptr<info> Fetch(std::shared_ptr<const plugin_configuration> config, 
		forecast_time requestedValidTime, 
		level requestedLevel, 
		const params& requestedParams, 
		forecast_type requestedType = forecast_type(kDeterministic),
		bool readPackedData = false);

	/**
	 * @brief Fetch data based on given arguments.
	 * 
 	 * Data can be defined in command line when himan started, or it can be already
	 * in cache or it can be fetched from neons.
	 *
	 * Will throw kFileDataNotFound if data is not found.
	 *
	 * @param config Plugin configuration
	 * @param requestedValidTime
	 * @param requestedLevel
	 * @param requestedParam
	 * @param readPackedData Whether to read unpacked data (from grib only!). Caller must do unpacking.
	 * @param controlWaitTime Whether this function should control wait times if they are specified.
	 * Default is true, will be set to false if this function is called from multi-param Fetch()
	 * @return shared_ptr to info-instance
	 */

	std::shared_ptr<info> Fetch(std::shared_ptr<const plugin_configuration> config, 
		forecast_time requestedValidTime, 
		level requestedLevel, 
		param requestedParam, 
		forecast_type requestedType = forecast_type(kDeterministic),
		bool readPackedData = false, 
		bool controlWaitTime = true);

	/**
	 * @brief Set flag for level transform
	 *
	 * @param theDoLevelTransform If false, level transform is not made
	 */

	void DoLevelTransform(bool theDoLevelTransform);
	bool DoLevelTransform() const;
	void DoInterpolation(bool theDoInterpolation);
	bool DoInterpolation() const;

	void UseCache(bool theUseCache);
	bool UseCache() const;

	void ApplyLandSeaMask(bool theApplyLandSeaMask);
	bool ApplyLandSeaMask() const;

	void LandSeaMaskThreshold(double theLandSeaMaskThreshold);
	double LandSeaMaskThreshold() const;
	
private:

	/**
	 * @brief Apply land-sea mask to requested data.
	 * 
	 * Threshold value can range from -1 to 1.
	 * If threshold < 0, masking will be done so that land is masked missing
	 * If threshold > 0, masking will be done so that sea is masked missing
	 * 
	 * @param theInfo Info that's masked
	 * @param requestedTime 
	 * @return True if masking is successful
	 */
	
	bool ApplyLandSeaMask(std::shared_ptr<const plugin_configuration> config, info& theInfo, forecast_time& requestedTime, forecast_type& requestedType);

	std::vector<std::shared_ptr<info>> FromCache(search_options& options);

	/**
	 * @brief Get data and metadata from a file.
	 * 
	 * Returns a vector of infos, mainly because one grib file can contain many
	 * grib messages. If read file is querydata, the vector size is always one
	 * (or zero if the read fails)
	 *
	 * Function will call FromGrib() or FromQueryData()
	 *
	 * @param files The files that are read
	 * @param options A struct holding the search criteria
	 * @param readContents Specify if data should also be read (and not only metadata)
	 * @param readPackedData Whether to read packed data. Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	std::vector<std::shared_ptr<info>> FromFile(const std::vector<std::string>& files,
												search_options& options,
												bool readContents = true,
												bool readPackedData = false,
												bool forceCaching = false);

	/**
	 * @brief Return all data from a grib file, overcoat for himan::plugin::grib::FromFile().
	 * @see himan::plugin::grib::FromFile()
	 *
	 * @param inputFile Input file name
	 * @param options Search options (param, level, time, prod, config)
	 * @param readContents Specify if data should also be read (and not only metadata)
	 * @param readPackedData Whether to read packed data. Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	std::vector<std::shared_ptr<info>> FromGrib(const std::string& inputFile, search_options& options, bool readContents = true, bool readPackedData = false, bool forceCaching = false);
	
	/**
	 * @brief Return all data from a querydata file, overcoat for himan::plugin::querydata::FromFile().
	 * @see himan::plugin::querydata::FromFile()
	 *
	 * @param inputFile Input file name
	 * @param options Search options (param, level, time, prod, config)
	 * @param readContents Specify if data should also be read (and not only metadata)
	 *
	 * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
	 */

	std::vector<std::shared_ptr<info>> FromQueryData(const std::string& inputFile, search_options& options, bool readContents = true);

	/**
	 * @brief Return all data from a CSV file, overcoat for himan::plugin::csv::FromFile().
	 * @see himan::plugin::csv::FromFile()
	 *
	 * @param inputFile Input file name
	 * @param options Search options (param, level, time, prod, config)
	 *
	 * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
	 */

	std::vector<std::shared_ptr<info>> FromCSV(const std::string& inputFile, search_options& options);

	/**
	 * @brief Map level definitions between models and code tables
	 *
	 * Ie. Fetch level that matches level 'targetLevel' for producer 'sourceProducer' from neons.
	 * 
	 * @param sourceProducer Id of source producer
	 * @param targetParam
	 * @param targetLevel
	 * @return New level. If mapping is not found, new level == targetLevel.
	 */
	
	level LevelTransform(const producer& sourceProducer, const param& targetParam, const level& targetLevel) const;

	/**
	 * @brief Try to fetch data from a single producer
	 *
	 * @param opts Search options
	 * @param readPackedData Determine whether to read packed data from grib
	 * @param fetchFromAuxiliaryFiles Determine whether to read data from aux files
	 * @return Vector of infos, zero sized if none found
	 */

	std::vector<std::shared_ptr<info>> FetchFromProducer(search_options& opts, bool readPackedData, bool fetchFromAuxiliaryFiles);

	/**
	 * @brief Function to perform area interpolation
	 *
	 * If target area is not identical to source area, perform area interpolation. Newbase
	 * is used to do the actual job. If data is packed, it will be unpacked before interpolation.
	 *
	 * Only the active part of infos is accessed; function does not move iterator positions.
	 *
	 * @param targetInfo Target info (ie. area)
	 * @param infos List of source infos, all elements of list will be interpolated.
	 * @return True if interpolation succeeds for all infos
	 */

	bool InterpolateArea(const plugin_configuration& conf, info& targetInfo, std::vector<info_t> infos) const;
	bool Interpolate(const plugin_configuration& conf, info& baseInfo, std::vector<info_t>& infos) const;
	bool ReorderPoints(info& base, std::vector<info_t> infos) const;
	bool InterpolateAreaCuda(info& base, info& target, unpacked& targetData) const;
	bool InterpolateAreaNewbase(info& base, info& source, unpacked& targetData) const;



	/**
	 * @brief Swap scanning mode if needed
	 *
	 * If source area != target area and interpolation is done, we might need to swap. This is because
	 * newbase is used to do the interpolation, and newbase will always normalize the scanning mode to
	 * bottom left. So if our target area is top left, we need to swap.
	 *
	 * Function will be a no-op if target grid scanning mode == wanted scanning mode.
	 *
	 * If target grid data is packed, function will unpack it.
	 * 
	 * @param targetGrid grid that's swapped
	 * @param targetScanningMode The scanning mode that the grid is swapped to
	 * @return True if swapping succeeds
	 */

	bool SwapTo(const std::shared_ptr<grid>& targetGrid, HPScanningMode targetScanningMode);

	HPFileType FileType(const std::string& theInputFile);
	bool itsDoLevelTransform; //<! Default true
	bool itsDoInterpolation; //<! Default true
	bool itsUseCache;
	bool itsApplyLandSeaMask;
	double itsLandSeaMaskThreshold;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<fetcher> ();
}

//typedef std::shared_ptr<fetcher> fetcher_t;

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* FETCHER_H */
