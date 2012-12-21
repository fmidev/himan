/*
 * writer.h
 *
 *  Created on: Nov 26, 2012
 *      Author: partio
 *
 *
 */

#ifndef WRITER_H
#define WRITER_H

#include "auxiliary_plugin.h"

namespace hilpee
{
namespace plugin
{

class writer : public auxiliary_plugin
{
	public:
		writer();

		virtual ~writer() {}

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::writer";
		}

		virtual HPPluginClass PluginClass() const
		{
			return kAuxiliary;
		}

		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		bool ToFile(std::shared_ptr<info> theInfo,
		            const std::string& theFileName,
		            HPFileType theFileType,
		            bool theActiveOnly = false);

	private:

};

#ifndef HILPEE_AUXILIARY_INCLUDE

// the class factories

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<writer> (new writer());
}

#endif /* HILPEE_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace hilpee

#endif /* WRITER_H */
