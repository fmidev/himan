/**
 * @file compiled_plugin_base.h
 *
 * @date Jan 15, 2013
 * @author partio
 */

#ifndef COMPILED_PLUGIN_BASE_H
#define COMPILED_PLUGIN_BASE_H

#include <NFmiGrid.h>
#include "info.h"
#include <mutex>

namespace himan
{
namespace plugin
{

class compiled_plugin_base
{
public:

	compiled_plugin_base() {}
    inline virtual ~compiled_plugin_base() {}

    compiled_plugin_base(const compiled_plugin_base& other) = delete;
    compiled_plugin_base& operator=(const compiled_plugin_base& other) = delete;

protected:

    virtual std::string ClassName() { return "himan::plugin::compiled_plugin_base"; }

	/**
	 * @brief Determine used thread count
	 *
	 * @param userThreadCount Number of threads specified by user, -1 if it's not specified
	 */

	unsigned short ThreadCount(short userThreadCount) const;

	/**
	 * @brief Interpolates value to point, or gets the value directly if grids are equal
	 *
	 * Function will try get the value without interpolation: if parameter gridsAreEqual is true,
	 * the value at given grid point is returned instantly. If not, function will check if the
	 * point projected from targetGrid to sourceGrid is actually the same point. If so, the
	 * value of that gridpoint is returned. As a last resort to do the interpolation.
	 *
	 * @param targetGrid Target grid where the point-to-interpolated is picked
	 * @param sourceGrid Source grid where the latlon point is projected, and the value fetched
	 * @param gridsAreEqual If the definitions of these two grids are equal, do no interpolation
	 * @param value The interpolated value
	 *
	 * @return Always true if no interpolation is made, otherwise returns the value given by newbase interpolation function
	 */

	bool InterpolateToPoint(std::shared_ptr<const NFmiGrid> targetGrid, std::shared_ptr<NFmiGrid> sourceGrid, bool gridsAreEqual, double& value);

	/**
	 * @brief Set leading dimension
	 */

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
  //  std::mutex itsAdjustDimensionMutex;
    HPDimensionType itsLeadingDimension;
    std::shared_ptr<info> itsFeederInfo;
};

} // namespace plugin
} // namespace himan

#endif /* COMPILED_PLUGIN_BASE_H */
