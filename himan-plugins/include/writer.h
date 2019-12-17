/*
 * writer.h
 *
 */

#ifndef WRITER_H
#define WRITER_H

#include "auxiliary_plugin.h"
#include "file_information.h"
#include "info.h"
#include "plugin_configuration.h"
#include "write_options.h"

namespace himan
{
namespace plugin
{
class writer : public auxiliary_plugin
{
   public:
	writer();

	virtual ~writer() = default;
	writer(const writer& other) = delete;
	writer& operator=(const writer& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::plugin::writer";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kAuxiliary;
	}
	template <typename T>
	bool ToFile(std::shared_ptr<info<T>> theInfo, std::shared_ptr<const plugin_configuration> conf);
	bool ToFile(std::shared_ptr<info<double>> theInfo, std::shared_ptr<const plugin_configuration> conf);

	write_options WriteOptions() const;
	void WriteOptions(const write_options& theWriteOptions);

   private:
	template <typename T>
	file_information CreateFile(info<T>& theInfo, std::shared_ptr<const plugin_configuration> conf);

	write_options itsWriteOptions;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factories

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<writer>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* WRITER_H */
