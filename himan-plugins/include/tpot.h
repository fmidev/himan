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
		return HPVersionNumber(0, 1);
	}

private:

	/**
	 * @brief Calculate "dry" potential temperature with poissons equation.
	 * 
	 * Note that input and output temperature to this function are in Celsius, but
	 * the formula itself requires Kelvins!
	 *
	 * http://san.hufs.ac.kr/~gwlee/session3/potential.html
	 *
	 * @param P Pressure in hPa
	 * @param T Temperature in C
	 * @return Potential temperature in C
	 */

	double Theta(double P, double T);

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

	void Run(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const plugin_configuration> theConfiguration, unsigned short theThreadIndex);
	void Calculate(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const plugin_configuration> theConfiguration, unsigned short theThreadIndex);

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
