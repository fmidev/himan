/**
 * @file csv.h
 *
 */

#ifndef CSV_H
#define CSV_H

#include "auxiliary_plugin.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{
class csv : public io_plugin
{
   public:
	csv();

	virtual ~csv() {}
	csv(const csv& other) = delete;
	csv& operator=(const csv& other) = delete;

	virtual std::string ClassName() const { return "himan::plugin::csv"; }
	virtual HPPluginClass PluginClass() const { return kAuxiliary; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
	/**
	 * @brief Return all data from a csv file.
	 *
	 * This function reads a csv file and returns the metadata+data (if specified) in a info
	 * class instance. Function returns a vector, but in reality the vector size is always zero
	 * (error reading file or no data matching search options was found) or one (data was found).
	 *
	 * @param file Input file name
	 * @param options Search options (param, level, time)
	 * @param readContents Specify if data should also be read (and not only metadata)
	 *
	 * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
	 */

	std::shared_ptr<info> FromFile(const std::string& inputFile, const search_options& options,
	                               bool readifNotMatching = false) const;

	/**
	 * @brief Write info contents to a csv file
	 *
	 * @param theInfo
	 * @param outputFile Name of output file
	 * @param fileWriteOption Determine whether to write whole contents or just the active part
	 * @return True if writing succeeds
	 */

	bool ToFile(info& theInfo, std::string& outputFile);
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<csv>(); }
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* CSV_H */
