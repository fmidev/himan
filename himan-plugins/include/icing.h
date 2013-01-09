/*
 * @file icing.h
 * 
 * @brief A plugin to calculate sea spray icing.
 *
 * @date Jan 03, 2013
 * 
 * @author aaltom
 * 
 */

#ifndef ICING_H
#define ICING_H

#include "compiled_plugin.h"

namespace himan
{
namespace plugin
{

class icing : public compiled_plugin
{
	public:
		icing();

		inline virtual ~icing() {}

		icing(const icing& other) = delete;
		icing& operator=(const icing& other) = delete;

		virtual void Process(std::shared_ptr<configuration> theConfiguration);

		virtual std::string ClassName() const
		{
			return "himan::plugin::icing";
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

		void Run(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);
		void Calculate(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);

		bool itsUseCuda;
};

// the class factory

#ifndef HIMAN_AUXILIARY_INCLUDE

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<icing> (new icing());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan


#endif /* ICING_H */
