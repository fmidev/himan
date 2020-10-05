#ifndef GEOTIFF_H
#define GEOTIFF_H

#include "auxiliary_plugin.h"
#include "file_information.h"
#include "info.h"

namespace himan
{
namespace plugin
{
class geotiff : public io_plugin
{
   public:
	geotiff();

	virtual ~geotiff() = default;
	geotiff(const geotiff& other) = delete;
	geotiff& operator=(const geotiff& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::geotiff";
	};
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	};

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> FromFile(const file_information& inputFile, const search_options& options,
	                                               bool validate = true, bool readData = true) const;
	std::vector<std::shared_ptr<info<double>>> FromFile(const file_information& inputFile,
	                                                    const search_options& options, bool validate = true,
	                                                    bool readData = true) const;

	template <typename T>
	file_information ToFile(info<T>& anInfo);
	file_information ToFile(info<double>& anInfo);

   private:
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<geotiff>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* GEOTIFF_H */
