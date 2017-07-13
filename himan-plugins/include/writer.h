/*
 * writer.h
 *
 */

#ifndef WRITER_H
#define WRITER_H

#include "auxiliary_plugin.h"
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

	virtual ~writer() {}
	writer(const writer& other) = delete;
	writer& operator=(const writer& other) = delete;

	virtual std::string ClassName() const { return "himan::plugin::writer"; }
	virtual HPPluginClass PluginClass() const { return kAuxiliary; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 2); }
	bool ToFile(info& theInfo, std::shared_ptr<const plugin_configuration> conf, const std::string& theFileName = "");

	write_options WriteOptions() const;
	void WriteOptions(const write_options& theWriteOptions);

   private:
	bool CreateFile(info& theInfo, std::shared_ptr<const plugin_configuration> conf, std::string& theOutputFile);

	write_options itsWriteOptions;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factories

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<writer>(); }
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* WRITER_H */
