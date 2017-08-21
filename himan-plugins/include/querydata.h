/**
 * @file querydata.h
 *
 */

#ifndef QUERYDATA_H
#define QUERYDATA_H

#include "auxiliary_plugin.h"
#include "search_options.h"

class NFmiFastQueryInfo;
class NFmiQueryData;
class NFmiTimeDescriptor;
class NFmiParamDescriptor;
class NFmiHPlaceDescriptor;
class NFmiVPlaceDescriptor;

namespace himan
{
namespace plugin
{
class querydata : public io_plugin
{
   public:
	querydata();

	virtual ~querydata() {}
	querydata(const querydata& other) = delete;
	querydata& operator=(const querydata& other) = delete;

	virtual std::string ClassName() const { return "himan::plugin::querydata"; }
	virtual HPPluginClass PluginClass() const { return kAuxiliary; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
	/**
	 * @brief Return all data from a querydata file.
	 *
	 * This function reads a querydata file and returns the metadata+data (if specified) in a info
	 * class instance. Function returns a vector, but in reality the vector size is always zero
	 * (error reading file or no data matching search options was found) or one (data was found).
	 * As querydata data is always in same projection and area, we can fit all data in a single info.
	 * The function returns a vector just to preserve compatitibilty with FromGrib().
	 *
	 * @param file Input file name
	 * @param options Search options (param, level, time)
	 *
	 * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
	 */

	std::shared_ptr<info> FromFile(const std::string& inputFile, const search_options& options) const;

	/**
	 * @brief Write info contents to a querydata file
	 *
	 * @param theInfo
	 * @param outputFile Name of output file
	 * @param fileWriteOption Determine whether to write whole contents or just the active part
	 * @return True if writing succeeds
	 */

	bool ToFile(info& theInfo, std::string& outputFile);

	/**
	 * @brief Create in-memory querydata from given info-instance
	 *
	 * This function *MAY* modify data in info class while it's doing
	 * conversion: sometimes the data needs to swapped to a form understood
	 * by newbase. Function will eventually swap it back but if some other
	 * thread accessed this data while it's swapped strange thinga will happen.
	 *
	 * @param theInfo source data
	 * @param activeOnly If set only the active part (current iterator positions) are read
	 * @param applyScaleAndBase If data is written to querydata, scale and base from database
	 * are used.
	 * @return shared pointer to querydata instance
	 */

	std::shared_ptr<NFmiQueryData> CreateQueryData(const info& theInfo, bool activeOnly,
	                                               bool applyScaleAndBase = false);

	/**
	 * @brief Create info from a given querydata
	 * @param theData
	 * @return shared_ptr to info instance
	 */

	std::shared_ptr<info> CreateInfo(std::shared_ptr<NFmiQueryData> theData) const;

	NFmiTimeDescriptor CreateTimeDescriptor(info& info, bool activeOnly);
	NFmiParamDescriptor CreateParamDescriptor(info& info, bool activeOnly);
	NFmiHPlaceDescriptor CreateHPlaceDescriptor(info& info, bool activeOnly);
	NFmiVPlaceDescriptor CreateVPlaceDescriptor(info& info, bool activeOnly);

	void UseDatabase(bool theUseDatabase);
	bool UseDatabase() const;

   private:
	/**
	 * @brief Copy data from info to querydata
	 *
	 * @param theInfo
	 * @param theQueryData
	 * @return
	 */
	bool CopyData(info& theInfo, NFmiFastQueryInfo& qinfo, bool applyScaleAndBase) const;

	NFmiHPlaceDescriptor CreateGrid(info& info) const;
	NFmiHPlaceDescriptor CreatePoint(info& info) const;

	bool itsUseDatabase;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<querydata>(); }
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* QUERYDATA_H */
