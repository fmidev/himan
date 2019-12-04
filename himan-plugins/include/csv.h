/**
 * @file csv.h
 *
 */

#ifndef CSV_H
#define CSV_H

#include "auxiliary_plugin.h"
#include "file_information.h"
#include "info.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{
class csv : public io_plugin
{
   public:
	csv();
	virtual ~csv() = default;

	csv(const csv& other) = delete;
	csv& operator=(const csv& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::plugin::csv";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kAuxiliary;
	}
	/**
	 * @brief Return all data from a csv file.
	 *
	 * This function reads a csv file and returns the metadata+data (if specified) in a info
	 * class instance. Function returns a vector, but in reality the vector size is always zero
	 * (error reading file or no data matching search options was found) or one (data was found).
	 *
	 * @param file Input file name
	 * @param options Search options (param, level, time)
	 *
	 * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
	 */

	template <typename T>
	std::shared_ptr<info<T>> FromFile(const std::string& inputFile, const search_options& options,
	                                  bool readifNotMatching = false) const;

	std::shared_ptr<info<double>> FromFile(const std::string& inputFile, const search_options& options,
	                                       bool readifNotMatching = false) const;

	/**
	 * @brief Write info contents to a csv file
	 *
	 * @param theInfo
	 * @param outputFile Name of output file
	 * @param fileWriteOption Determine whether to write whole contents or just the active part
	 * @return True if writing succeeds
	 */

	template <typename T>
	file_information ToFile(info<T>& theInfo);

	file_information ToFile(info<double>& theInfo);
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<csv>();
}
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* CSV_H */
