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
#include "radon_record.h"
#include "write_options.h"

namespace himan
{
namespace plugin
{

typedef std::tuple<HPWriteStatus, file_information, std::shared_ptr<himan::info<double>>> write_information;

class writer : public auxiliary_plugin
{
   public:
	writer();

	virtual ~writer() = default;
	writer(const writer& other) = delete;
	writer& operator=(const writer& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::writer";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	}
	template <typename T>
	HPWriteStatus ToFile(std::shared_ptr<info<T>> theInfo, std::shared_ptr<const plugin_configuration> conf);
	HPWriteStatus ToFile(std::shared_ptr<info<double>> theInfo, std::shared_ptr<const plugin_configuration> conf);

	write_options WriteOptions() const;
	void WriteOptions(const write_options& theWriteOptions);
	static void ClearPending();
	void WritePendingInfos(std::shared_ptr<const plugin_configuration> conf);

   private:
	template <typename T>
	std::pair<HPWriteStatus, file_information> CreateFile(info<T>& theInfo);

	template <typename T>
	std::pair<bool, radon_record> WriteToRadon(std::shared_ptr<const plugin_configuration> conf,
	                                           const file_information& finfo, std::shared_ptr<himan::info<T>> info);

	std::vector<write_information> WritePendingGribs(const std::vector<std::shared_ptr<himan::info<double>>>& infos);
	std::vector<write_information> WritePendingGeotiffs(const std::vector<std::shared_ptr<himan::info<double>>>& infos);
	void WritePendingToRadon(std::vector<write_information>& list);

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
