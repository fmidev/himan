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

class NFmiGrib;

namespace himan
{
namespace plugin
{
class grib : public io_plugin
{
   public:
	grib();

	virtual ~grib()
	{
	}
	grib(const grib& other) = delete;
	grib& operator=(const grib& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::plugin::grib";
	};
	virtual HPPluginClass PluginClass() const
	{
		return kAuxiliary;
	};

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
	 * @param readPackedData Whether to read packed data (from grib). Caller must do unpacking.
	 * @param forceCaching Force caching of data even if it does not match searched data
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FromFile(const file_information& inputFile, const search_options& options,
	                                               bool readPackedData, bool forceCaching) const;
	std::vector<std::shared_ptr<info<double>>> FromFile(const file_information& inputFile,
	                                                    const search_options& options, bool readPackedData,
	                                                    bool forceCaching) const;

	template <typename T>
	file_information ToFile(info<T>& anInfo);
	file_information ToFile(info<double>& anInfo);

   private:
	void WriteAreaAndGrid(const std::shared_ptr<himan::grid>& grid, const producer& prod);
	void WriteTime(const forecast_time& ftime, const producer& prod, const param& par);
	void WriteParameter(const param& par, const producer& prod, const forecast_type& ftype);
	void WriteLevel(const level& lev);
	void WriteForecastType(const forecast_type& forecastType, const producer& prod);

	template <typename T>
	void WriteData(info<T>& anInfo);

	template <typename T>
	himan::file_information CreateGribMessage(info<T>& anInfo);

	template <typename T>
	bool CreateInfoFromGrib(const search_options& options, bool readPackedData, bool forceCaching,
	                        std::shared_ptr<info<T>> newInfo) const;

	std::unique_ptr<grid> ReadAreaAndGrid() const;
	himan::param ReadParam(const search_options& options, const producer& prod) const;
	himan::forecast_time ReadTime() const;
	himan::level ReadLevel(const search_options& options, const producer& prod) const;
	himan::producer ReadProducer(const search_options& options) const;

	template <typename T>
	void ReadData(std::shared_ptr<info<T>> newInfo, bool readPackedData) const;

	void DetermineMessageNumber(file_information& finfo);
	void WriteMessageToFile(const file_information& finfo);

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

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<grib>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* GRIB_H */
