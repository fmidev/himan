/**
 * @file numerical_functions.h
 * @author tack
 *
 * @date July 2, 2015
 */

#ifndef NUMERICAL_FUNCTIONS_H
#define	NUMERICAL_FUNCTIONS_H

#include "himan_common.h"
#include "plugin_configuration.h"
#include <valarray>
namespace himan
{


namespace numerical_functions
{
/**
 * @class Integral
 * An object of type integral that can perform vertical integration in the atmosphere from a lower bound to an upper bound height. It can either be used to integrate a single variable (i.e. /int{x(h) dh} or a function /int{f(x(h)) dh}
 */
	
class integral
{
	public:

		integral();
		~integral() {}

		//virtual std::string ClassName() const { return "himan::numerical_functions::integral"; }

		//bool IsMissingValue(double theValue) const __attribute__((always_inline));

		//provide list of parameters that will be integrated over
		void Params(params theParams);

		//provide function of parameters that will be integrated over. TODO I'll document an example how that lambda function has to look like.
		void Function(std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> theFunction);

		//set bounds
		void LowerBound(const std::vector<double>& theLowerBound);
		void UpperBound(const std::vector<double>& theUpperBound);
		void LowerLevelLimit(int theLowestLevel);
		void UpperLevelLimit(int theHighestLevel);
		void ForecastType(forecast_type theType);
		void ForecastTime(forecast_time theTime);

		//return result
		const std::vector<double>& Result() const;

		//pass configuration to integration object (needed for fetching values)
                std::shared_ptr<const plugin_configuration> itsConfiguration;
		
		//evaluate the integral expression
		void Evaluate();
		//check if all necessary parameters are given
		bool Complete();

	private:

		//bool Evaluate();
		
		//bool itsMissingValuesAllowed;
		int itsLowestLevel;
		int itsHighestLevel;

                forecast_time itsTime;
                forecast_type itsType;
		level itsLevel;

		params itsParams;
                std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> itsFunction;

		std::vector<double> itsLowerBound;
		std::vector<double> itsUpperBound;
		std::vector<bool> itsOutOfBound;

		std::vector<double> itsResult; // variable is modified in some Result() const functions
		size_t itsIndex;
		
};

/*
inline
bool integral::IsMissingValue(double theValue) const
{
	if (theValue == kFloatMissing)
	{
		return true;
	}

	return false;
}
*/

} // namespace numerical_functions

} // namespace himan

#endif	/* NUMERICAL_FUNCTIONS_H */

