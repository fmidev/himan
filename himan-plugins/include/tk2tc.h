/*
 * tk2tc.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef TK2TC_H
#define TK2TC_H

#include "compiled_plugin.h"

namespace hilpee
{
namespace plugin
{

class tk2tc : public compiled_plugin
{
	public:
		tk2tc();

		inline virtual ~tk2tc() {}

		virtual void Process(std::shared_ptr<configuration> theConfiguration);

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::tk2tc";
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

		virtual void Run(std::shared_ptr<info>, const configuration& theConfiguration, unsigned short theThreadIndex);
		bool AdjustParams(std::shared_ptr<info> myTargetInfo);
		void Calculate(std::shared_ptr<info> theTargetInfo, const configuration& theConfiguration, unsigned short theThreadIndex);

		std::shared_ptr<info> itsMetaTargetInfo;
		boost::mutex itsMetaMutex;

};

// the class factory

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<tk2tc> (new tk2tc());
}

} // namespace plugin
} // namespace hilpee

#endif /* TK2TC */
