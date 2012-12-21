/*
 * param.h
 *
 *  Created on: Nov 27, 2012
 *      Author: partio
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

#ifndef HILPEE_PARAM_H
#define HILPEE_PARAM_H

#include "logger.h"
#include <NFmiParam.h>
// #include "NFmiGlobals.h" // FmiInterpolatioMethod

namespace hilpee
{

class param
{

	public:

		friend class hilpee_configuration;

		param();
		param(const std::string& theName);
		param(const std::string& theName, unsigned long theUnivId);

		param(const std::string& theName, unsigned long theUnivId,
		      float theScale,
		      float theBase,
		      const std::string& thePrecision = "%.1f",
		      FmiInterpolationMethod theInterpolationMethod = kNearestPoint);

		~param() {}

		std::string ClassName() const
		{
			return "hilpee::param";
		}

		HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		bool operator==(const param& other);
		bool operator!=(const param& other);

		void GribParameter(int theGribParameter);
		int GribParameter() const;

		void GribDiscipline(int theGribDiscipline);
		int GribDiscipline() const;

		void GribCategory(int theGribCategory);
		int GribCategory() const;

		unsigned long UnivId() const;
		void UnivId(unsigned long theUnivId);

		std::string Name() const;
		void Name(std::string theName);

		/**
		 *
		 * @return Unit of parameter
		 */

		HPParameterUnit Unit() const;

		/**
		 *
		 * @param theUnit
		 */

		void Unit(HPParameterUnit theUnit);

		std::ostream& Write(std::ostream& file) const;

	private:

		std::shared_ptr<NFmiParam> itsParam; //!< newbase param will hold name, univ_id and scale+base

		int itsGribParameter; //!< Grib parameter number whether in grib 1 or 2
		int itsGribTableVersion; //!< Grib table version (only in grib 1)
		int itsGribCategory; //!< Grib parameter category (only for grib2)
		int itsGribDiscipline; //!< Grib parameter discipline (only for grib2)

		HPParameterUnit itsUnit; //!< Unit of the parameter
		double itsMissingValue; //!< Missing value (default kFloatMissing)

		std::unique_ptr<logger> itsLogger;

};

inline
std::ostream& operator<<(std::ostream& file, param& ob)
{
	return ob.Write(file);
}

} // namespace hilpee

#endif /* PARAM_H */
