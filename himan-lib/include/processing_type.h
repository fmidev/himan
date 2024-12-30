/**
 * @file processing_type.h
 *
 * @brief simple class to describe parameter processing type metadata,
 * like for example probability related metadata
 */

#pragma once

#include "himan_common.h"
#include "serialization.h"

namespace himan
{
enum HPProcessingType
{
	kUnknownProcessingType = 0,
	kProbabilityGreaterThanOrEqual,
	kProbabilityGreaterThan,
	kProbabilityLessThanOrEqual,
	kProbabilityLessThan,
	kProbabilityEquals,
	kProbabilityEqualsIn,
	kProbabilityBetween,
	kProbabilityNotEquals,
	kFractile,
	kMean,
	kSpread,
	kStandardDeviation,
	kEFI,
	kProbability,  // a general 'probability', when the processing is more complicated than just checking a treshold
	kAreaProbabilityGreaterThanOrEqual,
	kAreaProbabilityGreaterThan,
	kAreaProbabilityLessThanOrEqual,
	kAreaProbabilityLessThan,
	kAreaProbabilityBetween,
	kAreaProbabilityEquals,
	kAreaProbabilityNotEquals,
	kAreaProbabilityEqualsIn,
	kBiasCorrection,
	kFiltered,
	kDetrend,
	kAnomaly,
	kNormalized,
	kClimatology,
	kCategorized,
	kPercentChange
};

const std::unordered_map<HPProcessingType, std::string> HPProcessingTypeToString = {
    {kUnknownProcessingType, "unknown"},
    {kProbabilityGreaterThanOrEqual, "probability greater than or equal"},
    {kProbabilityGreaterThan, "probability greater than"},
    {kProbabilityLessThanOrEqual, "probability less than or equal"},
    {kProbabilityLessThan, "probability less than"},
    {kProbabilityEquals, "probability equals"},
    {kProbabilityBetween, "probability between"},
    {kProbabilityEqualsIn, "probability equals in"},
    {kProbabilityNotEquals, "probability not equals"},
    {kFractile, "fractile"},
    {kMean, "mean"},
    {kSpread, "spread"},
    {kStandardDeviation, "standard deviation"},
    {kEFI, "efi"},
    {kProbability, "probability"},
    {kAreaProbabilityGreaterThanOrEqual, "area probability greater than or equal"},
    {kAreaProbabilityGreaterThan, "area probability greater than"},
    {kAreaProbabilityLessThanOrEqual, "area probability less than or equal"},
    {kAreaProbabilityLessThan, "area probability less than"},
    {kAreaProbabilityEquals, "area probability equals"},
    {kAreaProbabilityBetween, "area probability between"},
    {kAreaProbabilityEqualsIn, "area probability equals in"},
    {kAreaProbabilityNotEquals, "area probability not equals"},
    {kBiasCorrection, "bias correction"},
    {kFiltered, "filtered"},
    {kDetrend, "detrend"},
    {kAnomaly, "anomaly"},
    {kNormalized, "normalized"},
    {kClimatology, "climatology"},
    {kCategorized, "categorized"},
    {kPercentChange, "percent change"}};

const std::unordered_map<std::string, HPProcessingType> HPStringToProcessingType = {
    {"unknown", kUnknownProcessingType},
    {"probability greater than", kProbabilityGreaterThan},
    {"probability greater than or equal", kProbabilityGreaterThanOrEqual},
    {"probability less than", kProbabilityLessThan},
    {"probability less than or equal", kProbabilityLessThanOrEqual},
    {"probability between", kProbabilityBetween},
    {"probability equals", kProbabilityEquals},
    {"probability not equals", kProbabilityNotEquals},
    {"probability equals in", kProbabilityEqualsIn},
    {"fractile", kFractile},
    {"mean", kMean},
    {"spread", kSpread},
    {"standard deviation", kStandardDeviation},
    {"efi", kEFI},
    {"probability", kProbability},
    {"area probability greater than", kAreaProbabilityGreaterThan},
    {"area probability greater than or equal", kAreaProbabilityGreaterThanOrEqual},
    {"area probability less than", kAreaProbabilityLessThan},
    {"area probability less than or equal", kAreaProbabilityLessThanOrEqual},
    {"area probability between", kAreaProbabilityBetween},
    {"area probability equals", kAreaProbabilityEquals},
    {"area probability not equals", kAreaProbabilityNotEquals},
    {"area probability equals in", kAreaProbabilityEqualsIn},
    {"bias correction", kBiasCorrection},
    {"filtered", kFiltered},
    {"detrend", kDetrend},
    {"anomaly", kAnomaly},
    {"normalized", kNormalized},
    {"climatology", kClimatology},
    {"categorized", kCategorized},
    {"percent change", kPercentChange}};

class processing_type
{
   public:
	processing_type() = default;
	processing_type(HPProcessingType theType);
	processing_type(HPProcessingType theType, double theValue);
	processing_type(HPProcessingType theType, double theValue, double theValue2);
	processing_type(HPProcessingType theType, double theValue, double theValue2, int theNumberOfEnsembleMembers);
	processing_type(const std::string& procstr);
	~processing_type() = default;

	processing_type(const processing_type& other) = default;
	processing_type& operator=(const processing_type& other) = default;

	bool operator==(const processing_type& other) const;
	bool operator!=(const processing_type& other) const;
	operator std::string() const;

	std::string ClassName() const
	{
		return "himan::processing_type";
	}

	HPProcessingType Type() const;
	void Type(HPProcessingType theType);

	std::optional<double> Value() const;
	void Value(double theValue);

	std::optional<double> Value2() const;
	void Value2(double theValue2);

	/**
	 * @brief Return how many ensemble members were used in the processing (if any)
	 */

	int NumberOfEnsembleMembers() const;
	void NumberOfEnsembleMembers(int theNumberOfEnsembleMembers);

	std::ostream& Write(std::ostream& file) const;

   private:
	HPProcessingType itsType = kUnknownProcessingType;
	std::optional<double> itsValue;
	std::optional<double> itsValue2;

	int itsNumberOfEnsembleMembers = kHPMissingInt;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsType), CEREAL_NVP(itsValue), CEREAL_NVP(itsValue2), CEREAL_NVP(itsNumberOfEnsembleMembers));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const processing_type& ob)
{
	return ob.Write(file);
}
}  // namespace himan
