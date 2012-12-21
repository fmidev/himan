/*
 * vvmms.h
 *
 *  Created on: Dec 21, 2012
 *      Author: partio
 */

#ifndef VVMMS_H
#define VVMMS_H

#include "compiled_plugin.h"
#include <boost/thread.hpp>

namespace hilpee
{
namespace plugin
{

class vvmms : public compiled_plugin
{
	public:
		vvmms();

		inline virtual ~vvmms() {}

		virtual void Process(std::shared_ptr<configuration> theConfiguration);

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::vvmms";
		}

		virtual HPPluginClass PluginClass() const
		{
			return kCompiled;
		}

		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

	private:

		bool AdjustParams(std::shared_ptr<info> myTargetInfo);
		virtual void Run(std::shared_ptr<info>, const configuration& theConfiguration, unsigned short theThreadIndex);
		void Calculate(std::shared_ptr<info> theTargetInfo, const configuration& theConfiguration, unsigned short theThreadIndex);

		std::shared_ptr<info> itsMetaTargetInfo;
		boost::mutex itsMetaMutex;

};

// the class factory

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<vvmms> (new vvmms());
}

} // namespace plugin
} // namespace hilpee


#endif /* VVMMS_H */
