/**
 * @file cache.h
 *
 */

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "auxiliary_plugin.h"
#include "area_interpolation.h"

#include <mutex>

namespace himan
{
namespace plugin
{

class interpolator : public auxiliary_plugin
{
   public:
        virtual std::string ClassName() const
        {
                return "himan::plugin::interpolator";
        }
        virtual HPPluginClass PluginClass() const
        {
                return kAuxiliary;
        }
        virtual HPVersionNumber Version() const
        {
                return HPVersionNumber(1, 1);
        }

	static bool Insert(grid* source, grid* target);
	bool Interpolate(grid* source, grid* target);
   private:
	static std::map<std::string, interpolate::area_interpolation> cache;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
        return std::make_shared<interpolator>();
}
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* CACHE_H */
