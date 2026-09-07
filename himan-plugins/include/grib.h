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
/**
 * @brief How the grib message that is being read was selected.
 *
 * The metadata of a message has to be handled differently depending on how the message was
 * found: only when the location of the message is known beforehand, we can trust that it
 * contains the data that was requested.
 */

enum class message_selection
{
	kExactMessage,  // location of the message is known: file, offset and message number
	kSearchFile,    // all messages of the file are searched for the requested data
	kAllMessages    // all messages of the file are read, each with their own metadata
};

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

	/**
	 * @brief Create an info from a single grib message.
	 *
	 * The metadata of the message is validated against the search options, unless the user
	 * has disabled validation. In that case the amount of metadata that is read from the
	 * message depends on how the message was selected.
	 *
	 * @param selection How the message that is being read was selected
	 */

	template <typename T>
	bool CreateInfoFromGrib(const search_options& options, bool readPackedData, message_selection selection,
	                        std::shared_ptr<info<T>> newInfo, const NFmiGribMessage& message,
	                        bool readData = true) const;

	/**
	 * @brief Create an info from a single grib message, when the location of the message is
	 * not known beforehand.
	 *
	 * All metadata is read from the message itself.
	 *
	 * @param forceCaching Force caching of data even if it does not match searched data
	 */

	template <typename T>
	bool CreateInfoFromGrib(const search_options& options, bool readPackedData, bool forceCaching,
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
