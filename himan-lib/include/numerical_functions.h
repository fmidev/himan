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

		//provide list of parameters that will be integrated over
		void Params(params theParams);

		//provide function of parameters that will be integrated over. TODO I'll document an example how that lambda function has to look like.
		void Function(std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> theFunction);

		//set bounds
		void LowerBound(const std::valarray<double>& theLowerBound);
		void UpperBound(const std::valarray<double>& theUpperBound);
		void LowerLevelLimit(int theLowestLevel);
		void UpperLevelLimit(int theHighestLevel);
		void SetLevelLimits();
		void ForecastType(forecast_type theType);
		void ForecastTime(forecast_time theTime);
		void LevelType(level theLevel);
		void HeightInMeters (bool theHeightInMeters);
		//return result
		const std::valarray<double>& Result() const;

		//pass configuration to integration object (needed for fetching values)
                std::shared_ptr<const plugin_configuration> itsConfiguration;
		
		//evaluate the integral expression
		void Evaluate();
		bool Complete();

	private:

		bool itsHeightInMeters;
		int itsLowestLevel;
		int itsHighestLevel;

                forecast_time itsTime;
                forecast_type itsType;
		level itsLevel;

		params itsParams;
                std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> itsFunction;

		std::valarray<double> itsLowerBound;
		std::valarray<double> itsUpperBound;

		std::valarray<double> itsResult; // variable is modified in some Result() const functions
		size_t itsIndex;

                std::valarray<double> Interpolate(std::valarray<double>, std::valarray<double>, std::valarray<double>, std::valarray<double>, std::valarray<double>) const __attribute__((always_inline));
		std::pair<level,level> LevelForHeight(const producer&, double) const;
};

inline
std::valarray<double> integral::Interpolate(std::valarray<double> currentLevelValue, std::valarray<double> previousLevelValue, std::valarray<double> currentLevelHeight, std::valarray<double> previousLevelHeight, std::valarray<double> itsBound) const
{
	return (previousLevelValue + (currentLevelValue - previousLevelValue) * (itsBound - previousLevelHeight) / (currentLevelHeight - previousLevelHeight));
}

} // namespace numerical_functions

} // namespace himan

#endif	/* NUMERICAL_FUNCTIONS_H */

