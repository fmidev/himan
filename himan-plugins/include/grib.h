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

#include "auxiliary_plugin.h"
#include "NFmiGrib.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{

class grib : public auxiliary_plugin
{

public:

	grib();

	virtual ~grib() {}

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

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

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
	 *
	 * @return A vector of shared_ptr'd infos.
	 */

	std::vector<std::shared_ptr<info>> FromFile(const std::string& inputFile, const search_options& options, bool readContents = true, bool readPackedData = false);

	bool ToFile(std::shared_ptr<info> anInfo, const std::string& outputFile, HPFileType fileType, HPFileWriteOption fileWriteOption);

private:

	bool WriteGrib(std::shared_ptr<const info> anInfo, const std::string& outputFile, HPFileType fileType, bool appendToFile = false);

	std::shared_ptr<NFmiGrib> itsGrib;

};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<grib> (new grib());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* GRIB_H */
