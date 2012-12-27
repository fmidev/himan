/*
 * tpot.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef TPOT_H
#define TPOT_H

#include "compiled_plugin.h"
//#include <boost/thread.hpp>

namespace himan
{
namespace plugin
{

class tpot : public compiled_plugin
{
	public:
		tpot();

		inline virtual ~tpot() {}

		tpot(const tpot& other) = delete;
		tpot& operator=(const tpot& other) = delete;

		virtual void Process(std::shared_ptr<configuration> theConfiguration);

		virtual std::string ClassName() const
		{
			return "himan::plugin::tpot";
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
		bool AdjustParams(std::shared_ptr<info> myTargetInfo);
		void Calculate(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<tpot> (new tpot());
}

} // namespace plugin
} // namespace himan


#endif /* TPOT_H */
