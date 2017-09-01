/**
 * @file param.h
 *
 * This class holds all necessary parameter information, basically referring to
 * those parameters defined in Neons.
 * (In routine operations we always need to have parameter support in Neons since
 * otherwise we cannot store the calculated parameter)
 *
 * Ad-hoc interpreted calculations differ from this a bit: the source and result parameter
 * can be defined only in the source data (basically always querydata) which means
 * we cannot have a very rigorous validation of the parameters.
 *
 */

#ifndef PARAM_H
#define PARAM_H

#include "aggregation.h"
#include "himan_common.h"
#include "serialization.h"

namespace himan
{
class logger;

class param
{
   public:
	param();
	explicit param(const std::string& theName);
	param(const std::string& theName, unsigned long theUnivId);
	param(const std::string& theName, unsigned long theUnivId, HPParameterUnit theUnit);

	param(const std::string& theName, unsigned long theUnivId, long itsGribDiscipline, long itsGribCategory,
	      long itsGribParameter);
	param(const std::string& theName, unsigned long theUnivId, double theScale, double theBase,
	      HPInterpolationMethod theInterpolationMethod = kBiLinear);

	~param();

	param(const param& other);
	param& operator=(const param& other);

	std::string ClassName() const { return "himan::param"; }
	bool operator==(const param& other) const;
	bool operator!=(const param& other) const;

	/**
	 * @brief Set grib parameter number (grib2)
	 */

	void GribParameter(long theGribParameter);
	long GribParameter() const;

	/**
	 * @brief Set grib parameter discipline (grib2)
	 */

	void GribDiscipline(long theGribDiscipline);
	long GribDiscipline() const;

	/**
	 * @brief Set grib parameter category (grib2)
	 */

	void GribCategory(long theGribCategory);
	long GribCategory() const;

	/**
	 * @brief Set grib parameter table version number (grib1)
	 */

	void GribTableVersion(long theVersion);
	long GribTableVersion() const;

	/**
	 * @brief Set grib parameter number (grib1)
	 */

	void GribIndicatorOfParameter(long theGribIndicatorOfParameter);
	long GribIndicatorOfParameter() const;

	/**
	 * @brief Set universal id (newbase)
	 */

	unsigned long UnivId() const;
	void UnivId(unsigned long theUnivId);

	/**
	 * @brief Set parameter name
	 */

	void Name(const std::string& theName);
	std::string Name() const;

	/**
	 *
	 * @return Unit of parameter
	 */

	HPParameterUnit Unit() const;
	void Unit(HPParameterUnit theUnit);

	const aggregation& Aggregation() const;
	void Aggregation(const aggregation& theAggregation);

	double Base() const;
	void Base(double theBase);

	double Scale() const;
	void Scale(double theScale);

	HPInterpolationMethod InterpolationMethod() const;
	void InterpolationMethod(HPInterpolationMethod theInterpolationMethod);

	long Id() const;
	void Id(long theId);

	/**
	 * @brief Parameter output precision, number of decimals
	 */

	int Precision() const;
	void Precision(int thePrecision);

	std::ostream& Write(std::ostream& file) const;

   private:
	long itsId;           //<! neons id
	std::string itsName;  //!< neons name
	double itsScale;
	double itsBase;
	long itsUnivId;

	long itsGribParameter;             //!< Grib parameter number (only for grib2)
	long itsGribCategory;              //!< Grib parameter category (only for grib2)
	long itsGribDiscipline;            //!< Grib parameter discipline (only for grib2)
	long itsGribTableVersion;          //!< Grib table version (only in grib 1)
	long itsGribIndicatorOfParameter;  //!< Grib parameter number (only in grib 1)

	int itsVersion;
	HPInterpolationMethod itsInterpolationMethod;

	HPParameterUnit itsUnit;  //!< Unit of the parameter

	aggregation itsAggregation;
	int itsPrecision;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsId), CEREAL_NVP(itsName), CEREAL_NVP(itsScale), CEREAL_NVP(itsBase), CEREAL_NVP(itsUnivId),
		   CEREAL_NVP(itsGribParameter), CEREAL_NVP(itsGribCategory), CEREAL_NVP(itsGribDiscipline),
		   CEREAL_NVP(itsGribTableVersion), CEREAL_NVP(itsGribIndicatorOfParameter), CEREAL_NVP(itsUnit),
		   CEREAL_NVP(itsVersion), CEREAL_NVP(itsInterpolationMethod), CEREAL_NVP(itsAggregation),
		   CEREAL_NVP(itsPrecision));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const param& ob) { return ob.Write(file); }
typedef std::vector<himan::param> params;

}  // namespace himan

#endif /* PARAM_H */
