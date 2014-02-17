/**
 * @file querydata.h
 *
 * @date Nov 27, 2012
 * @author partio
 */

#ifndef QUERYDATA_H
#define QUERYDATA_H

#include "auxiliary_plugin.h"
#include "search_options.h"

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#include "NFmiFastQueryInfo.h"

#pragma clang diagnostic pop

#else

#include "NFmiFastQueryInfo.h"

#endif

namespace himan
{
namespace plugin
{

class querydata : public auxiliary_plugin
{
public:
	querydata();

	virtual ~querydata() {}

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

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

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
	 * @param readContents Specify if data should also be read (and not only metadata)
	 *
	 * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
	 */

	std::shared_ptr<info> FromFile(const std::string& inputFile, const search_options& options, bool readContents);

	/**
	 * @brief Write info contents to a querydata file
	 *
	 * @param theInfo 
	 * @param outputFile Name of output file
	 * @param fileWriteOption Determine whether to write whole contents or just the active part
	 * @return True if writing succeeds
	 */

	bool ToFile(std::shared_ptr<info> theInfo, const std::string& outputFile, HPFileWriteOption fileWriteOption);

	/**
	 * @brief Create in-memory querydata from given info-instance
	 *
	 * @param theInfo
	 * @param activeOnly If set only the active part (current iterator positions) are read
	 * @return shared pointer to querydata instance
	 */
	
	std::shared_ptr<NFmiQueryData> CreateQueryData(std::shared_ptr<info> theInfo, bool activeOnly);

	/**
	 * @brief Create info from a given querydata
	 * @param theData
	 * @return shared_ptr to info instance
	 */
	
	std::shared_ptr<info> CreateInfo(std::shared_ptr<NFmiQueryData> theData) const;

private:
	NFmiTimeDescriptor CreateTimeDescriptor(std::shared_ptr<info> info, bool activeOnly);
	NFmiParamDescriptor CreateParamDescriptor(std::shared_ptr<info> info, bool activeOnly);
	NFmiHPlaceDescriptor CreateHPlaceDescriptor(std::shared_ptr<info> info, bool activeOnly);
	NFmiVPlaceDescriptor CreateVPlaceDescriptor(std::shared_ptr<info> info, bool activeOnly);
   

};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<querydata> (new querydata());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* QUERYDATA_H */
