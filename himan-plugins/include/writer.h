/*
 * writer.h
 *
 *  Created on: Nov 26, 2012
 *	  Author: partio
 *
 *
 */

#ifndef WRITER_H
#define WRITER_H

#include "auxiliary_plugin.h"
#include "plugin_configuration.h"

namespace himan
{
namespace plugin
{

class writer : public auxiliary_plugin
{
public:
	writer();

	virtual ~writer() {}

	writer(const writer& other) = delete;
	writer& operator=(const writer& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::plugin::writer";
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
				std::shared_ptr<const plugin_configuration> conf,
				const std::string& theFileName = "");

private:

};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factories

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<writer> (new writer());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* WRITER_H */
