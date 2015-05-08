/**
 * @file tpot.h
 *
 * @date Nov 20, 2012
 * @author partio
 */

#ifndef TPOT_H
#define TPOT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "tpot_cuda.h"

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
		return HPVersionNumber(1, 0);
	}

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
	  * Method:
	  *
	  * The approximation given by Holton: Introduction to Dyn. Met.
	  * page 331 is used. If the air is not saturated, it is
	  * taken adiabatically to LCL.
	  *
	  * Original author K Eerola.
	  * @param P Pressure in hPa
	  * @param T Temperature in C
	  * @param TD Dew point temperature in C
	  * @param theta Dry potential temperature in C (if calculated)
	  * @return Equivalent potential temperature in C
	  */
	
	double ThetaE(double P, double T, double TD, double theta);

	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short theThreadIndex);
#ifdef HAVE_CUDA
	std::unique_ptr<tpot_cuda::options> CudaPrepare(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> PInfo, std::shared_ptr<info> TDInfo);
#endif
	
	bool itsThetaCalculation;
	bool itsThetaWCalculation;
	bool itsThetaECalculation;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<tpot> (new tpot());
}

} // namespace plugin
} // namespace himan


#endif /* TPOT_H */
