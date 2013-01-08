/**
 * @file util.h
 *
 * @date Dec 28, 2012
 * @author partio
 *
 * @brief Utility namespace for general helper functions and classes
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "himan_common.h"
#include "info.h"
#include <mutex>

namespace himan
{

/**
 * @namespace himan::util 
 * @brief Namespace for all utility functions and classes
 */
namespace util
{

/**
 * @class thread_manager
 *
 * @brief Provide services for individual threads
 *
 * Will help threads to advance leading and non-leading dimensions
 *
 */

class thread_manager
{
public:
    thread_manager() {}
    ~thread_manager() {}

    std::string ClassName() const
    {
        return "himan::util::thread_manager";
    }

    HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

    void Dimension(HPDimensionType theLeadingDimension)
    {
        itsLeadingDimension = theLeadingDimension;
    }

    void FeederInfo(std::shared_ptr<info> theFeederInfo)
    {
        itsFeederInfo = theFeederInfo;
        itsFeederInfo->Reset();
    }

    std::shared_ptr<info> FeederInfo() const
    {
        return itsFeederInfo;
    }

    bool AdjustLeadingDimension(std::shared_ptr<info> myTargetInfo);
    bool AdjustNonLeadingDimension(std::shared_ptr<info> myTargetInfo);
    void ResetNonLeadingDimension(std::shared_ptr<info> myTargetInfo);

private:
    std::mutex itsAdjustDimensionMutex;
    HPDimensionType itsLeadingDimension;
    std::shared_ptr<info> itsFeederInfo;
};

// Regular functions in the namespace

/**
 * @brief Determine file type by looking at the first few bytes of the file
 * @return Filetype of given argument, one of: grib, netcdf, querydata (or unknown)
 */

HPFileType FileType(const std::string& theFile);

/**
 * @brief Creates a neons-style filename with path, but without file extension
 */

std::string MakeNeonsFileName(std::shared_ptr <const info> info);

/**
 * @brief Splits a string and fills the gaps if requested
 * @param s Input string that should be splitted
 * @param delims String containing the characters that are used in splitting. If string length > 1, all characters are used
 * in splitting
 * @param fill Specify if gaps should be filled. For example string 1-4 can be splitted at - and filled with 2 and 3.
 * @return Vector or strings
 */

std::vector<std::string> Split(const std::string& s, const std::string& delims, bool fill);

} // namespace util
} // namespace himan


#endif /* UTIL_H_ */
