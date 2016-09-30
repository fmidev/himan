/**
 * @file grib.h
 *
 * @brief Class to implement grib writing and reading. Actual grib opening and reading is done by fmigrib library.
 *
 * @date Nov 20, 2012
 * @author partio
 */

#ifndef GRIB_H
#define GRIB_H

#include "NFmiGrib.h"
#include "auxiliary_plugin.h"

namespace himan
{
namespace plugin
{
class grib : public io_plugin
{
   public:
	grib();

	virtual ~grib() {}
	grib(const grib& other) = delete;
	grib& operator=(const grib& other) = delete;

	virtual std::string ClassName() const { return "himan::plugin::grib"; };
	virtual HPPluginClass PluginClass() const { return kAuxiliary; };
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
	std::shared_ptr<NFmiGrib> Reader();

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
	 * @param readContents Specify if data should also be read (and not only metadata)
	 * @param readPackedData Whether to read packed data (from grib). Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	std::vector<std::shared_ptr<info>> FromFile(const std::string& inputFile, const search_options& options,
	                                            bool readContents, bool readPackedData, bool forceCaching) const;

	/**
	 * @brief Return selected data from a grib index file.
	 *
	 * This function reads a grib index file and returns the metadata+data (if specified) in a one or
	 * more info class instance(s).
	 *
	 * @param file Input file name
	 * @param options Search options (param, level, time)
	 * @param readContents Specify if data should also be read (and not only metadata)
	 * @param readPackedData Whether to read packed data (from grib). Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	std::vector<std::shared_ptr<info>> FromIndexFile(const std::string& inputFile, const search_options& options,
	                                                 bool readContents, bool readPackedData, bool forceCaching) const;

	bool ToFile(info& anInfo, std::string& outputFile, bool appendToFile = false);

   private:
	void WriteAreaAndGrid(info& anInfo);
	void WriteTime(info& anInfo);
	void WriteParameter(info& anInfo);
	bool CreateInfoFromGrib(const search_options& options, bool readContents, bool readPackedData, bool forceCaching,
	                        std::shared_ptr<info> newInfo) const;

	/**
	 * @brief OptionsToKeys
	 *
 * Converts the search options struct into a map of key-value pairs that are used to select a grib message from a grib
 * index file
 * @param options Search options (param, level, time)
 */

	std::map<std::string, long> OptionsToKeys(const search_options& options) const;

	std::unique_ptr<grid> ReadAreaAndGrid() const;

	/**
	 * @brief UnpackBitmap
	 *
	 * Transform regular bitmap (unsigned char) to a int-based bitmap where each array key represents
	 * an actual data value. If bitmap is zero for that key, zero is also put to the int array. If bitmap
	 * is set for that key, the value is one.
	 *
	 * TODO: Change int-array to unpacked unsigned char array (reducing size 75%) or even not unpack bitmap beforehand
	 * but do it
	 * while computing stuff with the data array.
	 *
	 * @param bitmap Original bitmap read from grib
	 * @param unpacked Unpacked bitmap where number of keys is the same as in the data array
	 * @param len Length of original bitmap
	 */

	void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* __restrict__ unpacked, size_t len,
	                  size_t unpackedLen) const;

	std::shared_ptr<NFmiGrib> itsGrib;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<grib>(); }
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* GRIB_H */
