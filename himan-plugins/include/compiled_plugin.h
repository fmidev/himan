/*
 * compiled_plugin.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef COMPILED_PLUGIN_H
#define COMPILED_PLUGIN_H

#include "hilpee_plugin.h"
#include "configuration.h"

namespace hilpee
{
namespace plugin
{

class compiled_plugin : public hilpee_plugin
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
		                 const configuration& theConfiguration,
		                 unsigned short theThreadIndex) = 0;


		std::string itsClearTextFormula;

};

}
}

#endif /* COMPILED_PLUGIN_H */
