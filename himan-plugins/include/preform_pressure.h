/**
 * @file preform_pressure.h
 *
 *
 * @brief Precipitation form algorithm based on pressure fields, Korpela-Koistinen version
 *
 * Formula implementation in smarttool language by Simo Neiglick.
 *
 * From wiki: https://wiki.fmi.fi/pages/viewpage.action?pageId=21139101
 *
 * ===============================================================================
 *
 * Sateen olomuototulkinta _ilman mallipintadataa_ (6-10vrk)
 * "Kukkostulkintaa" ja Koistisen kaavaa soveltaen (käyttää vain yksinkertaistettua
 * "pilviohjelmaa" stratuksen, keskipilven ja jäätävän sateen tulkintaan).
 *
 * 1. Jäätävää tihkua, jos -10<T2m<0C, pakkasstratus (~-10...-0), sade heikkoa, ei keskipilveä
 * 2. Jäätävää vesisadetta, jos T2m<0C ja pinnan yläpuolella on T>0C kerros
 * 3a. Vettä, jos dz850-1000 > 1300m
 * 3b. Tihkua, jos stratusta pienellä sadeintensiteetillä, eikä keskipilveä
 * 4. Lunta, jos dz850-1000 < 1288m
 * 5. Räntää, jos 1288m < dz850-1000 < 1300m
 * 6. Lisäksi:
 *     - vesi/räntä/lumi-olomuotoa säädetään Koistisen kaavan perusteella
 *     - mallin lumisadetta (snowfall) ei (ainakaan toistaiseksi) käytetä
 *       (SmartMetin RR/Snowfall ristiriidoista johtuen)
 *
 * Huom. Vuoristossa (missä pintapaine<925 hPa) päättely ei aina toimi järkevästi!
 * (925/850hPa parametreilla on aina arvo, vaikka vuoristossa ne voivat olla pinnan alla)
 *
 * 05/2013 SN
 *
 * ===============================================================================
 *
 * Output is one of
 *
 * 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade
 *
 */

#ifndef PREFORM_PRESSURE_H
#define PREFORM_PRESSURE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class preform_pressure : public compiled_plugin, private compiled_plugin_base
{
   public:
	preform_pressure();

	inline virtual ~preform_pressure() {}
	preform_pressure(const preform_pressure& other) = delete;
	preform_pressure& operator=(const preform_pressure& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::preform_pressure"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<himan_plugin>(new preform_pressure()); }
}  // namespace plugin
}  // namespace himan

#endif /* PREFORM_PRESSURE_H */
