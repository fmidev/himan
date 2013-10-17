/**
 * @file modifier.h
 * @author partio
 *
 * @date September 27, 2013, 12:33 PM
 */

#ifndef MODIFIER_H
#define	MODIFIER_H

#include "himan_common.h"
#include "info.h"

namespace himan
{

/**
 * @class modifier
 *
 * Modifier is a tool to calculate different kinds of statistics from an input
 * data set. The layout is heavily influenced (or copied) from NFmiModifier,
 * but adjusted for himan.
 */
	
class modifier
{
	public:

		modifier();
		virtual ~modifier() {}

		// Compiler will create cctor and =, nothing but PODs here
		
		virtual std::string ClassName() const { return "himan::plugin::modifier"; }

		//virtual bool BoolOperation(float theValue);

		virtual double Value() const;
		virtual double MinimumValue() const;
		virtual double MaximumValue() const;

		virtual double Height() const;
		virtual double MinimumHeight() const;
		virtual double MaximumHeight() const;

		virtual void Calculate(double theValue, double theHeight) = 0;
		virtual void Clear(double fillValue = kHPMissingValue);

		bool ReturnHeight() const;
		void ReturnHeight(bool theReturnHeight);

		virtual bool IsMissingValue(double theValue) const;

		void FindValue(std::shared_ptr<const info> theFindValue);

		virtual void Init(std::shared_ptr<const himan::info> sourceInfo);

		bool NextLocation();
		void ResetLocation();
		size_t LocationIndex() const;

		std::shared_ptr<info> Results() const;

		virtual bool CalculationFinished() const
		{
			return false;
		}

	protected:

		bool itsReturnHeight;
		bool itsMissingValuesAllowed;

		std::shared_ptr<info> itsFindValue;
		size_t itsFindNthValue;

		std::shared_ptr<info> itsResult;

};

class modifier_max : public modifier
{
	public:
		modifier_max() : modifier() {}
		virtual ~modifier_max() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_max"; }

		virtual double MaximumValue() const;
		virtual double MaximumHeight() const;
		virtual void Calculate(double theValue, double theHeight);
};

class modifier_min : public modifier
{
	public:
		modifier_min() : modifier() {}
		virtual ~modifier_min() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_min"; }

		virtual double MinimumValue() const;
		virtual double MinimumHeight() const;
		virtual void Calculate(double theValue, double theHeight);
};

class modifier_maxmin : public modifier
{
	public:
		modifier_maxmin() : modifier() {}
		virtual ~modifier_maxmin() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_maxmin"; }

		virtual void Calculate(double theValue, double theHeight);

		virtual double Value() const;
		virtual double Height() const;
		
		virtual double MinimumValue() const;
		virtual double MaximumValue() const;
		virtual double MinimumHeight() const;
		virtual double MaximumHeight() const;

	private:
		
};

class modifier_sum : public modifier
{
	public:
		modifier_sum() : modifier() {}
		virtual ~modifier_sum() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_sum"; }

		virtual double Height() const;

		virtual void Calculate(double theValue, double theHeight);
};

class modifier_mean : public modifier_sum
{
	public:
		modifier_mean() : modifier_sum() {}
		virtual ~modifier_mean() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_mean"; }

		virtual void Calculate(double theValue, double theHeight);

		virtual std::shared_ptr<info> Results() const;

		virtual void Init(std::shared_ptr<const himan::info> sourceInfo);

	private:
		std::vector<size_t> itsValuesCount;
};

class modifier_count : public modifier
{
	public:
		modifier_count() : modifier() {}
		virtual ~modifier_count() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_count"; }

		virtual double Height() const;

		virtual void Calculate(double theValue, double theHeight);

		virtual void Init(std::shared_ptr<const himan::info> sourceInfo);

	private:
		std::vector<double> itsLowerValueThreshold;

};

class modifier_findheight : public modifier
{
	public:
		modifier_findheight() : modifier(), itsValuesFound(0) {}
		virtual ~modifier_findheight() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_findheight"; }

		virtual void Calculate(double theValue, double theHeight);

		virtual bool CalculationFinished() const;

		virtual void Init(std::shared_ptr<const himan::info> sourceInfo);

	private:
		std::vector<double> itsLowerValueThreshold;
		std::vector<double> itsLowerHeightThreshold;

		size_t itsValuesFound;
};

class modifier_findvalue : public modifier
{
	public:
		modifier_findvalue() : modifier(), itsValuesFound(0) {}
		virtual ~modifier_findvalue() {}

		virtual std::string ClassName() const { return "himan::plugin::modifier_findvalue"; }

		virtual void Calculate(double theValue, double theHeight);

		virtual bool CalculationFinished() const;

		virtual void Init(std::shared_ptr<const himan::info> sourceInfo);

	private:
		std::vector<double> itsLowerValueThreshold;
		std::vector<double> itsLowerHeightThreshold;

		size_t itsValuesFound;
};

} // namespace himan

#endif	/* MODIFIER_H */

