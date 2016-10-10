/**
 * @file aggregation.h
 *
 * @brief simple class to describe parameter aggregation metadata
 */

#ifndef AGGREGATION_H
#define AGGREGATION_H

#include "himan_common.h"

namespace himan
{

class aggregation
{
public:
	aggregation();
	aggregation(HPAggregationType theAggregationType, HPTimeResolution theTimeResolution, int theResolutionValue);
	~aggregation() {}
	aggregation(const aggregation& other);
	aggregation& operator=(const aggregation& other);

	bool operator==(const aggregation& other) const;
	bool operator!=(const aggregation& other) const;

	std::string ClassName() const { return "himan::aggregation"; }
	HPAggregationType Type() const;
	void Type(HPAggregationType theType);

	HPTimeResolution TimeResolution() const;
	void TimeResolution(HPTimeResolution theTimeResolution);

	int TimeResolutionValue() const;
	void TimeResolutionValue(int theTimeResolutionValue);

	std::ostream& Write(std::ostream& file) const;

private:
	HPAggregationType itsType;
	HPTimeResolution itsTimeResolution;
	int itsTimeResolutionValue;
};

inline std::ostream& operator<<(std::ostream& file, const aggregation& ob) { return ob.Write(file); }

}  // namespace himan

#endif /* AGGREGATION_H */
