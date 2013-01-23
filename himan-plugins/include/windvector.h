/*
 * @file windvector.h
 *
 * @date Jan 21, 2013
 * @author: Aalto
 */

#ifndef WINDVECTOR_H
#define WINDVECTOR_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "point.h"

namespace himan
{
namespace plugin
{

class windvector : public compiled_plugin, private compiled_plugin_base
{
public:
    windvector();

    inline virtual ~windvector() {}

    windvector(const windvector& other) = delete;
    windvector& operator=(const windvector& other) = delete;

    virtual void Process(std::shared_ptr<configuration> theConfiguration);

    virtual std::string ClassName() const
    {
        return "himan::plugin::windvector";
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

    void Run(std::shared_ptr<info>, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);
    void Calculate(std::shared_ptr<info> theTargetInfo, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);

    /**
     * @brief If U and V components of wind are grid relative, transform them to be earth-relative.
     *
     * Algorithm by J.E. HAUGEN (HIRLAM JUNE -92), modified by K. EEROLA
     * Algorithm originally defined in hilake/TURNDD.F
     *
     * All coordinate values are given in degrees N and degrees E (negative values for S and W)
     *
     * Function works only with rotated latlon projections.
     *
     * @param regPoint Latlon coordinates of the point in question in earth-relative form
     * @param rotPoint Latlon coordinates of the point in question in grid-relative form
     * @param southPole Latlon coordinates of south pole
     * @param UV U and V in grid-relative form
     * @return U and V in earth-relative form
     */

    himan::point UVToEarthRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV);

    /**
     * If U and V components of wind are earth relative, transform them to be grid-relative.
     *
     * Algorithm by J.E. HAUGEN (HIRLAM JUNE -92), modified by K. EEROLA
     * Algorithm originally defined in hilake/TURNDD.F
     *
     * All coordinate values are given in degrees N and degrees E (negative values for S and W)
     *
     * Function works only with rotated latlon projections.
     *
     * @param regPoint Latlon coordinates of the point in question in earth-relative form
     * @param rotPoint Latlon coordinates of the point in question in grid-relative form
     * @param southPole Latlon coordinates of south pole
     * @param UV U and V in earth-relative form
     * @return U and V in grid-relative form
     */

    himan::point UVToGridRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV);

    bool itsUseCuda;

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<windvector> (new windvector());
}

} // namespace plugin
} // namespace himan

#endif /* WINDVECTOR */
