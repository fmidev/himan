/**
 * @file fetcher.h
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
#include "file_information.h"
#include "info.h"
#include "plugin_configuration.h"
#include "search_options.h"
#include "timer.h"

namespace himan
{
namespace plugin
{
enum class HPDataFoundFrom : int
{
	kCache = 0,
	kAuxFile,
	kDatabase
};

class fetcher : public auxiliary_plugin
{
   public:
	fetcher();
	virtual ~fetcher() = default;
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

	template <typename T>
	std::shared_ptr<info<T>> Fetch(std::shared_ptr<const plugin_configuration> config, forecast_time requestedValidTime,
	                               level requestedLevel, const params& requestedParams,
	                               forecast_type requestedType = forecast_type(kDeterministic),
	                               bool readPackedData = false);
	std::shared_ptr<info<double>> Fetch(std::shared_ptr<const plugin_configuration> config,
	                                    forecast_time requestedValidTime, level requestedLevel,
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
	 * @return shared_ptr to info-instance
	 */

	template <typename T>
	std::shared_ptr<info<T>> Fetch(std::shared_ptr<const plugin_configuration> config, forecast_time requestedValidTime,
	                               level requestedLevel, param requestedParam,
	                               forecast_type requestedType = forecast_type(kDeterministic),
	                               bool readPackedData = false, bool suppressLogging = false);
	std::shared_ptr<info<double>> Fetch(std::shared_ptr<const plugin_configuration> config,
	                                    forecast_time requestedValidTime, level requestedLevel, param requestedParam,
	                                    forecast_type requestedType = forecast_type(kDeterministic),
	                                    bool readPackedData = false, bool suppressLogging = false);

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

	void DoVectorComponentRotation(bool theDoVectorComponentRotation);
	bool DoVectorComponentRotation() const;

   private:
	template <typename T>
	void RotateVectorComponents(std::vector<std::shared_ptr<info<T>>>& components, const grid* target,
	                            std::shared_ptr<const plugin_configuration> conf, const producer& sourceProd);

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

	template <typename T>
	bool ApplyLandSeaMask(std::shared_ptr<const plugin_configuration> config, std::shared_ptr<info<T>> theInfo,
	                      const forecast_time& requestedTime, const forecast_type& requestedType);

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FromCache(search_options& options);

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
	 * @param readPackedData Whether to read packed data. Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FromFile(const std::vector<himan::file_information>& files,
	                                               search_options& options, bool readPackedData = false,
	                                               bool forceCaching = false);

	/**
	 * @brief Map level definitions between models and our expected levels.
	 *
	 * For example we expect normal surface temperature to be found at level
	 * height/2, whereas ECMWF provides it at level ground/0.
	 *
	 * @param conf
	 * @param sourceProducer Source producer (in the example case ECMWF)
	 * @param targetParam Target param (in the example case T-K)
	 * @param targetLevel Target level (in the example case height/2)
	 * @return New level. If mapping is not found, new level == targetLevel.
	 */

	level LevelTransform(const std::shared_ptr<const configuration>& conf, const producer& sourceProducer,
	                     const param& targetParam, const level& targetLevel) const;

	/**
	 * @brief Try to fetch data from a single producer
	 *
	 * @param opts Search options
	 * @param readPackedData Determine whether to read packed data from grib
	 * @return Vector of infos, zero sized if none found
	 */

	template <typename T>
	std::pair<HPDataFoundFrom, std::vector<std::shared_ptr<info<T>>>> FetchFromAllSources(search_options& opts,
	                                                                                      bool readPackedData);

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FetchFromCache(search_options& opts);

	std::pair<HPDataFoundFrom, std::vector<std::shared_ptr<info<double>>>> FetchFromAuxiliaryFiles(search_options& opts,
	                                                                                               bool readPackedData);
	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FetchFromDatabase(search_options& opts, bool readPackedData);

	/**
	 * @brief Rotate and interpolate infos. Function is called when auxiliary files
	 *        are "batch processed". This data is always read as double.
	 *
	 * Processing is threaded.
	 */

	void AuxiliaryFilesRotateAndInterpolate(const search_options& opts, std::vector<info_t>& infos);

	template <typename T>
	std::shared_ptr<himan::info<T>> FetchFromProducer(search_options& opts, bool readPackedData, bool suppressLogging);
	template <typename T>
	std::shared_ptr<himan::info<T>> FetchFromProducerSingle(search_options& opts, bool readPackedData,
	                                                        bool suppressLogging);

	HPFileType FileType(const std::string& theInputFile);
	bool itsDoLevelTransform;           //<! Default true
	bool itsDoInterpolation;            //<! Default true
	bool itsDoVectorComponentRotation;  //<! Default false
	bool itsUseCache;
	bool itsApplyLandSeaMask;
	double itsLandSeaMaskThreshold;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<fetcher>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* FETCHER_H */
