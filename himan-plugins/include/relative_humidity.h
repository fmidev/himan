/**
 * @file relative_humidity.h
 *
 * @date Jan Oct 7, 2013
 * @author: partio
 */

#ifndef RELATIVE_HUMIDITY_H
#define RELATIVE_HUMIDITY_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "relative_humidity_cuda.h"
namespace himan
{
namespace plugin
{

/**
 * @class relative_humidity
 *
 */

class relative_humidity : public compiled_plugin, private compiled_plugin_base
{
public:
    relative_humidity();

    inline virtual ~relative_humidity() {}

    relative_humidity(const relative_humidity& other) = delete;
    relative_humidity& operator=(const relative_humidity& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::relative_humidity";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kCompiled;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(1, 0);
    }

private:
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);
	double WithQ(double T, double Q, double P);
	double WithTD(double T, double TD);
#ifdef HAVE_CUDA
	std::unique_ptr<relative_humidity_cuda::options> CudaPrepareTTD(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> TDInfo, double TDBase, double TBase);
	std::unique_ptr<relative_humidity_cuda::options> CudaPrepareTQP(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> QInfo, std::shared_ptr<info> PInfo, double PScale, double TBase);
	std::unique_ptr<relative_humidity_cuda::options> CudaPrepareTQ(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> QInfo, double P_level, double TBase);

	void CudaFinish(std::unique_ptr<relative_humidity_cuda::options> opts, std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> TDInfo);
	void CudaFinish(std::unique_ptr<relative_humidity_cuda::options> opts, std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> QInfo, std::shared_ptr<info> PInfo);

#endif

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<relative_humidity> (new relative_humidity());
}

} // namespace plugin
} // namespace himan


#endif /* RELATIVE_HUMIDITY_H */
