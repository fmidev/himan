/**
 * @file querydata.h
 *
 */

#ifndef QUERYDATA_H
#define QUERYDATA_H

#include "auxiliary_plugin.h"
#include "file_information.h"
#include "info.h"
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
	virtual ~querydata() = default;
	querydata(const querydata& other) = delete;
	querydata& operator=(const querydata& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::plugin::querydata";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kAuxiliary;
	}

	/**
	 * @brief Write info contents to a querydata file
	 *
	 * @param theInfo
	 * @param outputFile Name of output file
	 * @param fileWriteOption Determine whether to write whole contents or just the active part
	 * @return True if writing succeeds
	 */

	template <typename T>
	file_information ToFile(info<T>& theInfo);

	file_information ToFile(info<double>& theInfo);

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

	template <typename T>
	std::shared_ptr<NFmiQueryData> CreateQueryData(const info<T>& theInfo, bool activeOnly,
	                                               bool applyScaleAndBase = false);

	//	std::shared_ptr<NFmiQueryData> CreateQueryData(const info<double>& theInfo, bool activeOnly,
	//	                                               bool applyScaleAndBase = false);

	/**
	 * @brief Create info from a given querydata
	 * @param theData
	 * @return shared_ptr to info instance
	 */

	template <typename T>
	std::shared_ptr<info<T>> CreateInfo(std::shared_ptr<NFmiQueryData> theData) const;

	void UseDatabase(bool theUseDatabase);
	bool UseDatabase() const;

   private:
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<querydata>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* QUERYDATA_H */
