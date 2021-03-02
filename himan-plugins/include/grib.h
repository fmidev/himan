/**
 * @file grib.h
 *
 * @brief Class to implement grib writing and reading. Actual grib opening and reading is done by fmigrib library.
 */

#ifndef GRIB_H
#define GRIB_H

#include "auxiliary_plugin.h"
#include "file_information.h"
#include "info.h"
#include <NFmiGribMessage.h>

namespace himan
{
namespace plugin
{
class grib : public io_plugin
{
   public:
	grib();
	virtual ~grib() = default;

	grib(const grib& other) = delete;
	grib& operator=(const grib& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::grib";
	};
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	};

	/**
	 * @brief Return all data from a grib file.
	 *
	 * This function reads a grib file and returns the metadata+data (if specified) in a one or
	 * more info class instance(s).
	 *
	 * Function returns a vector because unlike with querydata, one grib file can contain many messages
	 * with totally different areas and projections. A single info-class instance can handle different times,
	 * levels, params and even changing grid size but it cannot handle different sized areas. Therefore from
	 * this function we need to return a vector.
	 *
	 * @param file Input file name
	 * @param options Search options (param, level, time)
	 * @param readPackedData Whether to read packed data (from grib). Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of std::shared_ptr'd infos.
	 */

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FromFile(const file_information& inputFile, const search_options& options,
	                                               bool readPackedData, bool forceCaching) const;
	std::vector<std::shared_ptr<info<double>>> FromFile(const file_information& inputFile,
	                                                    const search_options& options, bool readPackedData,
	                                                    bool forceCaching) const;

	template <typename T>
	std::pair<HPWriteStatus, file_information> ToFile(info<T>& anInfo);
	std::pair<HPWriteStatus, file_information> ToFile(info<double>& anInfo);

	template <typename T>
	bool CreateInfoFromGrib(const search_options& options, bool readPackedData, bool validate,
	                        std::shared_ptr<info<T>> newInfo, const NFmiGribMessage& message,
	                        bool readData = true) const;

	template <typename T>
	std::pair<himan::file_information, NFmiGribMessage> CreateGribMessage(info<T>& anInfo);
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<grib>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* GRIB_H */
