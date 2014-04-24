/*
 * @file stability.h
 *
 * @date Jan 23, 2012
 * @author: Aalto, revised by partio
 */

#ifndef STABILITY_H
#define STABILITY_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class stability
 *
 * @brief Calculate k-index and other indexes that describe that stability of the atmosphere.
 *
 */

class stability : public compiled_plugin, private compiled_plugin_base
{
public:
	stability();

	inline virtual ~stability() {}

	stability(const stability& other) = delete;
	stability& operator=(const stability& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::stability";
	}

	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(2, 0);
	}

	/**
	 * @brief Showalter Index
	 *
	 * Will lift a parcel of air from 850 hPa to 500 hPa either dry or wet
	 * adiabatically.
	 *
	 * http://forecast.weather.gov/glossary.php?word=SHOWALTER+INDEX
	 * 
	 * @param T850 Temperature of 850 hPa isobar in Kelvins
	 * @param T500 Temperature of 500 hPa isobar in Kelvins
	 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
	 * @return Index value
	 */

	double SI(double T850, double T500, double TD850) const;

	/**
	 * @brief Cross Totals Index
	 *
	 * http://glossary.ametsoc.org/wiki/Stability_index
	 * 
	 * @param T500 Temperature of 500 hPa isobar in Kelvins
	 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
	 * @return Index value (TD850 - T500)
	 */

	double CTI(double T500, double TD850) const;

	/**
	 * @brief Vertical Totals Index
	 *
	 * http://glossary.ametsoc.org/wiki/Stability_index
	 *
	 * @param T850 Temperature of 850 hPa isobar in Kelvins
	 * @param T500 Temperature of 500 hPa isobar in Kelvins
	 * @return Index value (T850 - T500)
	 */

	double VTI(double T850, double T500) const;

	/**
	 * @brief Total Totals Index
	 *
	 * http://glossary.ametsoc.org/wiki/Stability_index
	 *
	 * @param T850 Temperature of 850 hPa isobar in Kelvins
	 * @param T500 Temperature of 500 hPa isobar in Kelvins
	 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
	 * @return Index value ( T850 - T500 ) + ( TD850 - T500 )
	 */

	double TTI(double T850, double T500, double TD850) const;

	/**
	 * @brief Lifted index
	 *
	 * http://en.wikipedia.org/wiki/Lifted_index
	 * 
	 * @param T500 Temperature of 500 hPa isobar in Kelvins
	 * @return Index value
	 */

	double LI(double T500) const;

	/**
	 * @brief K-Index
	 * 
	 * @param T500 Temperature of 500 hPa isobar in Kelvins
	 * @param T700 Temperature of 700 hPa isobar in Kelvins
	 * @param T850 Temperature of 850 hPa isobar in Kelvins
	 * @param TD700 Dewpoint temperature of 700 hPa isobar in Kelvins
	 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
	 * @return Index value
	 */
	
	double KI(double T500, double T700, double T850, double TD700, double TD850) const;

private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<stability> (new stability());
}

} // namespace plugin
} // namespace himan

#endif /* STABILITY_H */
