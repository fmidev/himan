/*
 * compiled_plugin.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef COMPILED_PLUGIN_H
#define COMPILED_PLUGIN_H

#include "himan_plugin.h"
#include "configuration.h"

namespace himan
{
namespace plugin
{

class compiled_plugin : public himan_plugin
{
	public:

		compiled_plugin() {}

		virtual ~compiled_plugin() {}

		virtual void Process(std::shared_ptr<configuration> theConfiguration) = 0;

		virtual std::string Formula()
		{
			return itsClearTextFormula;
		}
		virtual void Formula(std::string theClearTextFormula)
		{
			itsClearTextFormula = theClearTextFormula;
		}

	protected:

		virtual void Run(std::shared_ptr<info> myTargetInfo,
		                 std::shared_ptr<const configuration> theConfiguration,
		                 unsigned short theThreadIndex) = 0;

		virtual bool AdjustParams(std::shared_ptr<info> myTargetInfo) = 0;

		virtual void Calculate(std::shared_ptr<info> theTargetInfo,
		                       std::shared_ptr<const configuration> theConfiguration,
		                       unsigned short theThreadIndex) = 0;

		std::string itsClearTextFormula;
		boost::mutex itsAdjustParamMutex;
		std::shared_ptr<info> itsFeederInfo;

};

} // namespace plugin
} // namespace himan

#endif /* COMPILED_PLUGIN_H */
