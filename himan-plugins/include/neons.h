/*
 * neons.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 *
 * Access to neons database.
 *
 * A note on compiling and linking:
 *
 * This class is only an overcoat to fmidb. fmidb-library is just a bunch of objects
 * and it contains no linking, so we have to link here. Now, the problem is that
 * oracle instant client provides only shared library version of libclntsh. This
 * means that we have to link every library/executable that used this class against
 * libclntsh. And libclntsh want libaio etc.
 *
 *  One option to solve this would be to link this class statically with libclntsh,
 *  but to do so we'd need the full version of oracle client. Also, the .a version
 *  of oracle client library is HUGE, nearly 100M.
 */

#ifndef NEONS_H
#define NEONS_H

#include "auxiliary_plugin.h"
#include "NFmiNeonsDB.h"

namespace hilpee
{
namespace plugin
{

class neons : public auxiliary_plugin
{

	public:
		neons();

		virtual ~neons() {}

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::neons";
		}

		virtual HPPluginClass PluginClass() const
		{
			return kAuxiliary;
		}

		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		std::shared_ptr<NFmiNeonsDB> Neons();

	private:
		std::shared_ptr<NFmiNeonsDB> itsNeonsDB;

};

#ifndef HILPEE_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<neons> (new neons());
}

#endif /* HILPEE_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace hilpee

#endif /* NEONS_H */
