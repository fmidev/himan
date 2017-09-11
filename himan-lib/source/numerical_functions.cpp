/**
 * @file numerical_functions.cpp
 */

#include "numerical_functions.h"
#include "NFmiInterpolation.h"
#include "plugin_factory.h"
#include <algorithm>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "neons.h"
#include "radon.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;

using namespace numerical_functions;

#if 0
integral::integral() : itsComplete(8, true) {}
void integral::Params(std::vector<param> theParams) { itsParams = theParams; }
void integral::Function(std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> theFunction)
{
	itsFunction = theFunction;
}

const std::valarray<double>& integral::Result() const
{
	assert(itsResult.size());

	return itsResult;
}

void integral::Evaluate()
{
	assert(Complete());

	auto f = GET_PLUGIN(fetcher);

	std::vector<info_t> paramInfos;
	// Create a container that contains the parameter data. This container passed as an argument to the function that is
	// being integrated over.
	std::vector<std::valarray<double>> paramsData;
	std::valarray<double> previousLevelValue;
	std::valarray<double> currentLevelValue;
	std::valarray<double> previousLevelHeight;
	std::valarray<double> currentLevelHeight;
	std::valarray<bool> missingValueMask;

	for (int lvl = itsHighestLevel; lvl <= itsLowestLevel; ++lvl)
	{
		itsLevel.Value(lvl);
		// fetch parameters

		// for (param itsParam:itsParams) <-- only for g++ 4.8
		for (unsigned int i = 0; i < itsParams.size(); i++)
		{
			param itsParam = itsParams[i];
			paramInfos.push_back(f->Fetch(itsConfiguration, itsTime, itsLevel, itsParam, itsType,
			                              itsConfiguration->UseCudaForPacking()));
			paramsData.push_back(
			    std::valarray<double>(paramInfos.back()->Data().Values().data(), paramInfos.back()->Data().Size()));
			// allocate result container
			if (!itsResult.size()) itsResult.resize(paramInfos.back()->Data().Size());
			// initialize missingValueMask
			if (!missingValueMask.size())
				missingValueMask = std::valarray<bool>(false, paramInfos.back()->Data().Size());
		}

		// fetch height
		param heightParam;
		if (itsHeightInMeters)
		{
			heightParam = param("HL-M");
		}
		else
		{
			heightParam = param("P-HPa");
		}

		info_t heights =
		    f->Fetch(itsConfiguration, itsTime, itsLevel, heightParam, itsType, itsConfiguration->UseCudaForPacking());
		currentLevelHeight = std::valarray<double>(heights->Data().Values().data(), heights->Data().Size());

		// mask for missing values
		auto missingValueMaskFunction = std::valarray<bool>(false, heights->Data().Size());
		for (unsigned int i = 0; i < paramsData.size(); i++)
		{
			missingValueMaskFunction = (IsKFloatMissing(paramsData[i]) || missingValueMaskFunction);
		}

		// evaluate integration function
		if (itsFunction)
		{
			currentLevelValue = itsFunction(paramsData);
		}
		else
		{
			// use first param if no function is given
			currentLevelValue = paramsData[0];
		}

		// put missing values back in
		currentLevelValue[missingValueMaskFunction] = MissingDouble();

		// update mask of missing values in the result of the integral
		missingValueMask =
		    (IsKFloatMissing(currentLevelHeight) || isMissing(currentLevelValue) || missingValueMask);
		// move data from current level to previous level

		if (lvl == itsLowestLevel)
		{
			previousLevelHeight = std::move(currentLevelHeight);
			previousLevelValue = std::move(currentLevelValue);
			continue;
		}

		// perform trapezoideal integration
		//
		std::valarray<bool> lowerBoundMask;
		std::valarray<bool> upperBoundMask;
		std::valarray<bool> insideBoundsMask;
		// vectorized form of trapezoideal integration
		if (itsHeightInMeters)
		{
			lowerBoundMask = (previousLevelHeight > itsLowerBound && currentLevelHeight < itsLowerBound);
			upperBoundMask = (previousLevelHeight > itsUpperBound && currentLevelHeight < itsUpperBound);
			insideBoundsMask = (previousLevelHeight <= itsUpperBound && currentLevelHeight >= itsLowerBound);
		}
		else
		// height in Pascal
		{
			lowerBoundMask = (previousLevelHeight < itsLowerBound && currentLevelHeight > itsLowerBound);
			upperBoundMask = (previousLevelHeight < itsUpperBound && currentLevelHeight > itsUpperBound);
			insideBoundsMask = (previousLevelHeight >= itsUpperBound && currentLevelHeight <= itsLowerBound);
		}
		// TODO Perhaps it is better to cast valarrays from the mask_array before this step. According to Stroustrup all
		// operators and mathematical function can be applied to mask_array as well. Unfortunately not the case.
		itsResult[upperBoundMask] += (Interpolate(std::valarray<double>(currentLevelValue[upperBoundMask]),
		                                          std::valarray<double>(previousLevelValue[upperBoundMask]),
		                                          std::valarray<double>(currentLevelHeight[upperBoundMask]),
		                                          std::valarray<double>(previousLevelHeight[upperBoundMask]),
		                                          std::valarray<double>(itsUpperBound[upperBoundMask])) +
		                              std::valarray<double>(currentLevelValue[upperBoundMask])) /
		                             2 * (std::valarray<double>(itsUpperBound[upperBoundMask]) -
		                                  std::valarray<double>(currentLevelHeight[upperBoundMask]));
		itsResult[lowerBoundMask] += (Interpolate(std::valarray<double>(currentLevelValue[lowerBoundMask]),
		                                          std::valarray<double>(previousLevelValue[lowerBoundMask]),
		                                          std::valarray<double>(currentLevelHeight[lowerBoundMask]),
		                                          std::valarray<double>(previousLevelHeight[lowerBoundMask]),
		                                          std::valarray<double>(itsLowerBound[lowerBoundMask])) +
		                              std::valarray<double>(previousLevelValue[lowerBoundMask])) /
		                             2 * (std::valarray<double>(previousLevelHeight[lowerBoundMask]) -
		                                  std::valarray<double>(itsLowerBound[lowerBoundMask]));
		itsResult[insideBoundsMask] += (std::valarray<double>(previousLevelValue[insideBoundsMask]) +
		                                std::valarray<double>(currentLevelValue[insideBoundsMask])) /
		                               2 * (std::valarray<double>(previousLevelHeight[insideBoundsMask]) -
		                                    std::valarray<double>(currentLevelHeight[insideBoundsMask]));

		// serial version of trapezoideal integration
		//
		for (size_t i=0; i<paramInfos.back()->Data().Size(); ++i)
		{

		        // value is below the lowest limit
		        if (itsHeightInMeters && previousLevelHeight[i] > itsLowerBound[i] && currentLevelHeight[i] <
		itsLowerBound[i] || !itsHeightInMeters && previousLevelHeight[i] < itsLowerBound[i] && currentLevelHeight[i] >
		itsLowerBound[i])
		        {
		                double lowerValue =
		previousLevelValue[i]+(currentLevelValue[i]-previousLevelValue[i])*(itsLowerBound[i]-previousLevelHeight[i])/(currentLevelHeight[i]-previousLevelHeight[i]);
		                itsResult[i] += (lowerValue + previousLevelValue[i]) / 2 * (previousLevelHeight[i] -
		itsLowerBound[i]);
		    }
		        // value is above the highest limit
		        else if (itsHeightInMeters && previousLevelHeight[i] > itsUpperBound[i] && currentLevelHeight[i] <
		itsUpperBound[i] || !itsHeightInMeters && previousLevelHeight[i] < itsUpperBound[i] && currentLevelHeight[i] >
		itsUpperBound[i])
		        {
		                double upperValue =
		previousLevelValue[i]+(currentLevelValue[i]-previousLevelValue[i])*(itsUpperBound[i]-previousLevelHeight[i])/(currentLevelHeight[i]-previousLevelHeight[i]);
		                itsResult[i] += (upperValue + currentLevelValue[i]) / 2 * (itsUpperBound[i] -
		currentLevelHeight[i]);

		        }
		        else if (itsHeightInMeters && previousLevelHeight[i] <= itsUpperBound[i] && currentLevelHeight[i] >=
		itsLowerBound[i] || !itsHeightInMeters && previousLevelHeight[i] >= itsUpperBound[i] && currentLevelHeight[i] <=
		itsLowerBound[i])
		    {
		                itsResult[i] += (previousLevelValue[i] + currentLevelValue[i]) / 2 * (previousLevelHeight[i] -
		currentLevelHeight[i]);
		        }

		}

		// move data from current level to previous level at the end of the integration step
		previousLevelHeight = std::move(currentLevelHeight);
		previousLevelValue = std::move(currentLevelValue);
		paramInfos.clear();
		paramsData.clear();
	}
	// Insert missing values into result
	itsResult[missingValueMask] = MissingDouble();
}

void integral::LowerBound(const std::valarray<double>& theLowerBound)
{
	itsComplete[0] = true;
	itsLowerBound = theLowerBound;
}

void integral::UpperBound(const std::valarray<double>& theUpperBound)
{
	itsComplete[1] = true;
	itsUpperBound = theUpperBound;
}

void integral::LowerLevelLimit(int theLowestLevel)
{
	itsComplete[2] = true;
	itsLowestLevel = theLowestLevel;
}

void integral::UpperLevelLimit(int theHighestLevel)
{
	itsComplete[3] = true;
	itsHighestLevel = theHighestLevel;
}

void integral::SetLevelLimits()
{
	assert(itsComplete[0] && itsComplete[1]);

	producer prod = itsConfiguration->SourceProducer(0);

	double max_value = itsHeightInMeters ? itsUpperBound.max() : itsUpperBound.min();
	double min_value = itsHeightInMeters ? itsLowerBound.min() : itsLowerBound.max();

	if (IsKFloatMissing(max_value) || isMissing(min_value))
	{
		// itsLogger->Error("Min or max values of given heights are missing");
		throw kFileDataNotFound;
	}

	auto levelsForMaxHeight = LevelForHeight(prod, max_value);
	auto levelsForMinHeight = LevelForHeight(prod, min_value);

	itsHighestLevel = static_cast<int>(levelsForMaxHeight.second.Value());
	itsLowestLevel = static_cast<int>(levelsForMinHeight.first.Value());

	assert(itsLowestLevel >= itsHighestLevel);
	itsComplete[2] = true;
	itsComplete[3] = true;
}

void integral::ForecastType(forecast_type theType)
{
	itsComplete[4] = true;
	itsType = theType;
}

void integral::ForecastTime(forecast_time theTime)
{
	itsComplete[5] = true;
	itsTime = theTime;
}

void integral::LevelType(level theLevel)
{
	itsComplete[6] = true;
	itsLevel = theLevel;
}

void integral::HeightInMeters(bool theHeightInMeters)
{
	itsComplete[7] = true;
	itsHeightInMeters = theHeightInMeters;
}
// TODO add check that all information that is needed is given to the class object
bool integral::Complete()
{
	return std::all_of(itsComplete.begin(), itsComplete.end(), [](bool i) { return i == true; });
}

std::pair<level, level> integral::LevelForHeight(const producer& prod, double height) const
{
	using boost::lexical_cast;

	long producerId = 0;

	// Hybrid level heights are calculated by himan, so coalesce the related
	// forecast producer id with the himan producer id.

	switch (prod.Id())
	{
		case 1:
		case 230:
			producerId = 230;
			break;

		case 131:
		case 240:
			producerId = 240;
			break;

		case 199:
		case 210:
			producerId = 210;
			break;

		default:
			// itsLogger->Error("Unsupported producer for hitool::LevelForHeight(): " + lexical_cast<std::string>
			// (prod.Id()));
			break;
	}

	std::stringstream query;

	if (itsHeightInMeters)
	{
		query << "SELECT min(CASE WHEN maximum_height <= " << height
		      << " THEN level_value ELSE NULL END) AS lowest_level, "
		      << "max(CASE WHEN minimum_height >= " << height << " THEN level_value ELSE NULL END) AS highest_level "
		      << "FROM "
		      << "hybrid_level_height "
		      << "WHERE "
		      << "producer_id = " << producerId;
	}
	else
	{
		// Add/subtract 1 already in the query, since due to the composition of the query it will return
		// the first level that is higher than lower height and vice versa

		query << "SELECT max(CASE WHEN minimum_pressure <= " << height
		      << " THEN level_value+1 ELSE NULL END) AS lowest_level, "
		      << "min(CASE WHEN maximum_pressure >= " << height
		      << " THEN level_value-1 ELSE NULL END) AS highest_level "
		      << "FROM "
		      << "hybrid_level_height "
		      << "WHERE "
		      << "producer_id = " << producerId;
		;
	}

	HPDatabaseType dbtype = itsConfiguration->DatabaseType();

	std::vector<std::string> row;

	long absolutelowest = kHPMissingInt;
	long absolutehighest = kHPMissingInt;

	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);
		n->NeonsDB().Query(query.str());

		row = n->NeonsDB().FetchRow();

		absolutelowest = lexical_cast<long>(n->ProducerMetaData(prod.Id(), "last hybrid level number"));
		absolutehighest = lexical_cast<long>(n->ProducerMetaData(prod.Id(), "first hybrid level number"));
	}

	if (row.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
	{
		auto r = GET_PLUGIN(radon);
		r->RadonDB().Query(query.str());

		row = r->RadonDB().FetchRow();

		absolutelowest = lexical_cast<long>(r->RadonDB().GetProducerMetaData(prod.Id(), "last hybrid level number"));
		absolutehighest = lexical_cast<long>(r->RadonDB().GetProducerMetaData(prod.Id(), "first hybrid level number"));
	}

	long newlowest = absolutelowest, newhighest = absolutehighest;

	if (!row.empty())
	{
		// If requested height is below lowest level (f.ex. 0 meters) or above highest (f.ex. 80km)
		// database query will return null

		if (row[0] != "")
		{
			// SQL query returns the level value that precedes the requested value.
			// For first hybrid level (the highest ie max), get one level above the max level if possible
			// For last hybrid level (the lowest ie min), get one level below the min level if possible
			// This means that we have a buffer of three levels for both directions!

			newlowest = lexical_cast<long>(row[0]) + 1;

			if (newlowest > absolutelowest)
			{
				newlowest = absolutelowest;
			}
		}

		if (row[1] != "")
		{
			newhighest = lexical_cast<long>(row[1]) - 1;

			if (newhighest < absolutehighest)
			{
				newhighest = absolutehighest;
			}
		}

		if (newhighest > newlowest)
		{
			newhighest = newlowest;
		}
	}

	assert(newlowest >= newhighest);

	return std::make_pair<level, level>(level(kHybrid, newlowest), level(kHybrid, newhighest));
}
#endif

matrix<double> numerical_functions::Filter2D(const matrix<double>& A, const matrix<double>& B)
{
	// find center position of kernel (half of kernel size)
	matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double convolution_value;  // accumulated value of the convolution at a given grid point in A
	double kernel_weight_sum;  // accumulated value of the kernel weights in B that are used to compute the convolution
	                           // at given point A

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// check if data contains missing values
	if (A.MissingCount() == 0)  // if no missing values in the data we can use a faster algorithm
	{
		// calculate for inner field
		// the weights are used as given on input
		// assert (sum(B) == 1)
		for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
		{
			for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
			{
				convolution_value = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;          // column index of flipped kernel
					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);
						convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value;
			}
		}

		// treat boundaries separately
		// weights get adjusted so that the sum of weights for the active part of the kernel remains 1
		// calculate for upper boundary
		for (int j = 0; j < kCenterY; ++j)  // columns
		{
			for (int i = 0; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;          // column index of flipped kernel
					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary

						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
						{
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}

		// calculate for lower boundary
		for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
		{
			for (int i = 0; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;  // column index of flipped kernel

					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
						{
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}

		// calculate for left boundary
		for (int j = 0; j < ASizeY; ++j)  // columns
		{
			for (int i = 0; i < kCenterX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;  // column index of flipped kernel

					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
						{
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}

		// calculate for right boundary
		for (int j = 0; j < ASizeY; ++j)  // columns
		{
			for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;          // column index of flipped kernel
					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
						{
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}
	}
	else  // data contains missing values
	{
		std::cout << "util::Filter2D: Data contains missing values -> Choosing slow algorithm." << std::endl;
		double kernel_missing_count;
		for (int j = 0; j < ASizeY; ++j)  // columns
		{
			for (int i = 0; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
				kernel_missing_count = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;  // column index of flipped kernel

					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
						{
							if (A.IsMissing(ii, jj, 0))
							{
								kernel_missing_count++;
								continue;
							}

							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				if (kernel_missing_count < 3)
				{
					const size_t index = ret.Index(i, j, 0);
					ret[index] = convolution_value / kernel_weight_sum;
				}
				else
				{
					const size_t index = ret.Index(i, j, 0);
					ret[index] = himan::MissingDouble();
				}
			}
		}
	}
	return ret;
}

himan::matrix<double> numerical_functions::Max2D(const himan::matrix<double>& A, const himan::matrix<double>& B)
{
	using himan::MissingDouble;

	// find center position of kernel (half of kernel size)
	himan::matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double max_value;  // maximum value of the convolution

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// calculate for inner field
	// the weights are used as given on input
	// assert (sum(B) == 1)

	assert(B.MissingCount() == 0);

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			max_value = -1e38;
			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const double a = A.At(ii, jj, 0);
					const double b = B.At(mm, nn, 0);

					if (IsValid(a) && b != 0)
					{
						max_value = fmax(a * b, max_value);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// treat boundaries separately
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			max_value = -1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary

					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			max_value = -1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			max_value = -1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			max_value = -1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	return ret;
}

himan::matrix<double> numerical_functions::Min2D(const himan::matrix<double>& A, const himan::matrix<double>& B)
{
	using himan::MissingDouble;

	// find center position of kernel (half of kernel size)
	himan::matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double min_value;  // minimum value of the convolution

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// calculate for inner field
	// the weights are used as given on input
	// assert (sum(B) == 1)

	assert(B.MissingCount() == 0);

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			min_value = 1e38;
			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const double a = A.At(ii, jj, 0);
					const double b = B.At(mm, nn, 0);

					if (IsValid(a) && b != 0)
					{
						min_value = fmin(a * b, min_value);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// treat boundaries separately
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			min_value = 1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary

					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			min_value = 1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			min_value = 1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			min_value = 1e38;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	return ret;
}
