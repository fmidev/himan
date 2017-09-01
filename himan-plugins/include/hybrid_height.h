/**
 * @file hybrid_height.h
 *
 */

#ifndef HYBRID_HEIGHT_H
#define HYBRID_HEIGHT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <boost/thread.hpp>

namespace himan
{
namespace plugin
{
class hybrid_height : public compiled_plugin, private compiled_plugin_base
{
   public:
	hybrid_height();

	virtual ~hybrid_height();

	hybrid_height(const hybrid_height& other) = delete;
	hybrid_height& operator=(const hybrid_height& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::hybrid_height"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 2); }
	virtual void WriteToFile(const info& targetInfo, write_options opts = write_options()) override;

   private:
	void Write(himan::info targetInfo);
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);
	bool WithIteration(info_t& myTargetInfo);
	bool WithGeopotential(info_t& myTargetInfo);
	void Prefetch(info_t myTargetInfo);

	int itsBottomLevel;
	bool itsUseGeopotential;
	bool itsUseWriterThreads;
	mutable boost::thread_group itsWriterGroup;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<hybrid_height>(); }
}  // namespace plugin
}  // namespace himan

#endif /* HYBRID_HEIGHT_H */
