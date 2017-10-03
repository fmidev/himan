/**
 * @file tpot.h
 *
 */

#ifndef TPOT_H
#define TPOT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "tpot.cuh"

namespace himan
{
namespace plugin
{
class tpot : public compiled_plugin, private compiled_plugin_base
{
   public:
	tpot();

	inline virtual ~tpot() {}
	tpot(const tpot& other) = delete;
	tpot& operator=(const tpot& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::tpot"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
   private:
	/**
	 * @brief Calculating pseudo-adiabatic theta (ie thetaw).
	 *
	 * Method: numerical integration from LCL to 1000mb level along wet adiabatic line.
	 *
	 * Numerical integration method used: leapfrog starting with euler
	 *
	 * Original author AK Sarkanen / May 1985
	 *
	 * @param P Pressure in hPa
	 * @param T Temperature in C
	 * @param TD Dew point temperature in C
	 * @return Pseudo-adiabatic potential temperature in C
	 */

	double ThetaW(double P, double T, double TD);

	/**
	  * @brief Calculate equivalent potential temperature
	  *
	  * @param P Pressure in Pa
	  * @param T Temperature in K
	  * @param TD Dew point temperature in C
	  * @return Equivalent potential temperature in K
	  */

	double ThetaE(double P, double T, double TD);

	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short theThreadIndex);
#ifdef HAVE_CUDA
	std::unique_ptr<tpot_cuda::options> CudaPrepare(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo,
	                                                std::shared_ptr<info> PInfo, std::shared_ptr<info> TDInfo);
#endif

	bool itsThetaCalculation;
	bool itsThetaWCalculation;
	bool itsThetaECalculation;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<tpot>(new tpot()); }
}  // namespace plugin
}  // namespace himan

#endif /* TPOT_H */
