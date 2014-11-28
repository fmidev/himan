/**
 * @file modifier.h
 * @author partio
 *
 * @date September 27, 2013, 12:33 PM
 */

#ifndef MODIFIER_H
#define	MODIFIER_H

#include "himan_common.h"

namespace himan
{

/**
 * @class modifier
 *
 * Modifier is a tool to calculate different kinds of statistics from an input
 * data set. The layout is heavily influenced (or copied) from NFmiModifier,
 * but adjusted for himan.
 *
 * This is the parent class in a hierarchy, all operation-depended functionality
 * is implemented in child-classes.
 */
	
class modifier
{
	public:

		modifier();
		virtual ~modifier() {}

		// Compiler will create cctor and =, nothing but PODs here
		
		virtual std::string ClassName() const { return "himan::modifier"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing) = 0;
		virtual void Clear(double fillValue = kFloatMissing);

		virtual bool IsMissingValue(double theValue) const;

		void FindValue(const std::vector<double>& theFindValue);
		void LowerHeight(const std::vector<double>& theLowerHeight);
		void UpperHeight(const std::vector<double>& theUpperHeight);

		virtual const std::vector<double>& Result() const;

		virtual bool CalculationFinished() const;

		size_t FindNth() const;
		void FindNth(size_t theNth);
	
		virtual void Process(const std::vector<double>& theData, const std::vector<double>& theHeights);

		std::ostream& Write(std::ostream& file) const;

		size_t HeightsCrossed() const;
		HPModifierType Type() const;
		
	protected:
		modifier(HPModifierType theModifierType);
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);

		/**
		 * @brief Function checks that data is not missing and falls within the given height range.
		 */

		virtual bool Evaluate(double theValue, double theHeight);
		virtual double Value() const;
		virtual void Value(double theValue);
		bool itsMissingValuesAllowed;

		std::vector<double> itsLowerHeight;
		std::vector<double> itsUpperHeight;
		std::vector<double> itsFindValue;
		size_t itsFindNthValue;

		mutable std::vector<double> itsResult; // variable is modified in some Result() const functions
		size_t itsIndex;
		std::vector<bool> itsOutOfBoundHeights;

		HPModifierType itsModifierType;
		
};

inline
std::ostream& operator<<(std::ostream& file, const modifier& ob)
{
	return ob.Write(file);
}

/**
 * @class Calculates the maximum value in a given height range
 */

class modifier_max : public modifier
{
	public:
		modifier_max() : modifier(kMaximumModifier) {}
		virtual ~modifier_max() {}

		virtual std::string ClassName() const { return "himan::modifier_max"; }

		//virtual double MaximumValue() const;
		virtual void Calculate(double theValue, double theHeight = kFloatMissing);
};

/**
 * @class Calculates minimum value in a given height range
 */

class modifier_min : public modifier
{
	public:
		modifier_min() : modifier(kMinimumModifier) {}
		virtual ~modifier_min() {}

		virtual std::string ClassName() const { return "himan::modifier_min"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);
};

/**
 * @class Calculates the maximum and minimum values in a given height range
 */

class modifier_maxmin : public modifier
{
	public:
		modifier_maxmin() : modifier(kMaximumMinimumModifier) {}
		virtual ~modifier_maxmin() {}

		virtual std::string ClassName() const { return "himan::modifier_maxmin"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);

		virtual const std::vector<double>& Result() const;

		//virtual double Value() const;
		
		//virtual double MinimumValue() const;
		//virtual double MaximumValue() const;

	protected:
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);

		std::vector<double> itsMaximumResult;
};

/**
 * @class Calculate the sum of values in a given height range
 */

class modifier_sum : public modifier
{
	public:
		modifier_sum() : modifier(kAccumulationModifier) {}
		virtual ~modifier_sum() {}

		virtual std::string ClassName() const { return "himan::modifier_sum"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);
};

/**
 *  * @class Find the integral of a parameter between lower and upper height bounds. Integral is calculated by the trapezoidal rule.
 *   */

class modifier_integral : public modifier
{
	public:
		modifier_integral() : modifier(kIntegralModifier) {}
		virtual ~modifier_integral() {}

		virtual std::string ClassName() const { return "himan::::modifier_integral"; }
		
		virtual void Calculate(double theValue, double theHeight = kFloatMissing);

		virtual bool CalculationFinished() const;

	protected:
		modifier_integral(HPModifierType theModifierType) : modifier(theModifierType) {}

		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);
		virtual bool Evaluate(double theValue, double theHeight);

		std::vector<double> itsPreviousValue;
		std::vector<double> itsPreviousHeight;
};

/**
 * @class Calculate the mean of values in a given height range
 */

class modifier_mean : public modifier_integral
{
	public:
		modifier_mean() : modifier_integral(kAverageModifier) {}
		virtual ~modifier_mean() {}

		virtual std::string ClassName() const { return "himan::modifier_mean"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);

		virtual const std::vector<double>& Result() const;
		
		virtual bool CalculationFinished() const;

	protected:
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);
		virtual bool Evaluate(double theValue, double theHeight);

		std::vector<double> itsRange;
};

/**
 * @class Count the number of values in a given height range
 */

class modifier_count : public modifier
{
	public:
		modifier_count() : modifier(kCountModifier) {}
		virtual ~modifier_count() {}

		virtual std::string ClassName() const { return "himan::modifier_count"; }

		//virtual double Height() const;

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);

	protected:
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);

		std::vector<double> itsPreviousValue;

};

/**
 * @class Find height of a given value.
 *
 * By default class will calculate the height of the first value occurred. This can
 * be changed by setting itsFindNthValue. If value is zero, the last occurrence will
 * be returned.
 *
 * Note! Unlike NFmiModifier, if user has requested the height of 3rd value and that
 * is not found, value will be missing value (newbase gives the height of 2nd found value).
 */

class modifier_findheight : public modifier
{
	public:
		modifier_findheight() : modifier(kFindHeightModifier), itsValuesFound(0) {}
		virtual ~modifier_findheight() {}

		virtual std::string ClassName() const { return "himan::modifier_findheight"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);

		virtual bool CalculationFinished() const;

		virtual void Clear(double fillValue = kFloatMissing);

	protected:
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);

		std::vector<double> itsPreviousValue;
		std::vector<double> itsPreviousHeight;
		std::vector<size_t> itsFoundNValues;
		
		size_t itsValuesFound;

};

/**
 * @class Find value of parameter in a given height
 */

class modifier_findvalue : public modifier
{
	public:
		modifier_findvalue() : modifier(kFindValueModifier), itsValuesFound(0) {}
		virtual ~modifier_findvalue() {}

		virtual std::string ClassName() const { return "himan::modifier_findvalue"; }

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);

		virtual bool CalculationFinished() const;

		virtual void Clear(double fillValue = kFloatMissing);

	private:
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);

		std::vector<double> itsPreviousValue;
		std::vector<double> itsPreviousHeight;

		size_t itsValuesFound;
};

/**
 * @class Find positive and negative area under a function
 */

class modifier_plusminusarea : public modifier
{
	public:
		modifier_plusminusarea() : modifier(kPlusMinusAreaModifier), itsValuesFound(0) {}
		virtual ~modifier_plusminusarea() {}

		virtual std::string ClassName() const { return "himan::modifier_plusminusarea";}

		virtual void Calculate(double theValue, double theHeight = kFloatMissing);
 
		virtual const std::vector<double>& Result() const;

		virtual bool CalculationFinished() const;

	private:
		virtual void Init(const std::vector<double>& theData, const std::vector<double>& theHeights);
		virtual bool Evaluate(double theValue, double theHeight);

		std::vector<double> itsPreviousValue;
		std::vector<double> itsPreviousHeight;
		mutable std::vector<double> itsPlusArea;
		std::vector<double> itsMinusArea;

		size_t itsValuesFound;
};

} // namespace himan

#endif	/* MODIFIER_H */

