/*
 * util.h
 *
 *  Created on: Dec 30, 2012
 *      Author: partio
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "himan_common.h"
#include "info.h"
#include <mutex>

namespace himan
{
namespace util
{

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

HPFileType FileType(const std::string& theFile);
std::string MakeNeonsFileName(std::shared_ptr <const info> info);
std::vector<std::string> Split(const std::string& s, const std::string& delims, bool fill);

} // namespace util
} // namespace himan


#endif /* UTIL_H_ */
