/**
 * @file matrix.h
 *
 * @brief 2-3d matrix to store data. Does not have any mathematical implications of matrices.
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "himan_common.h"
#include "serialization.h"
#include <mutex>
#include <algorithm>

namespace himan
{
class grid;

/**
 * @brief Compare float/double bitwise, i.e. nan comparison is possible
 *
 */

inline bool Compare(const double& lhs, const double& rhs)
{
	const uint64_t* lhs_ptr = reinterpret_cast<const uint64_t*>(&lhs);
	const uint64_t* rhs_ptr = reinterpret_cast<const uint64_t*>(&rhs);

	return *lhs_ptr == *rhs_ptr;
}

inline bool Compare(const float& lhs, const float& rhs)
{
	const uint32_t* lhs_ptr = reinterpret_cast<const uint32_t*>(&lhs);
	const uint32_t* rhs_ptr = reinterpret_cast<const uint32_t*>(&rhs);

	return *lhs_ptr == *rhs_ptr;
}

// For all types other than float/double use == operator
template <typename T>
inline bool Compare(const T& lhs, const T& rhs)
{
	return lhs == rhs;
}

template <class T>
class matrix
{
   public:
	matrix() : itsData(0), itsWidth(0), itsHeight(0), itsDepth(0) {}
	matrix(size_t theWidth, size_t theHeight, size_t theDepth, T theMissingValue)
	    : itsData(theWidth * theHeight * theDepth),
	      itsWidth(theWidth),
	      itsHeight(theHeight),
	      itsDepth(theDepth),
	      itsMissingValue(theMissingValue)
	{
	}

	matrix(size_t theWidth, size_t theHeight, size_t theDepth, T theMissingValue, T theFillValue)
	    : itsData(theWidth * theHeight * theDepth, theFillValue),
	      itsWidth(theWidth),
	      itsHeight(theHeight),
	      itsDepth(theDepth),
	      itsMissingValue(theMissingValue)
	{
	}

	matrix(const matrix& other)
	    : itsData(other.itsData)  // Copy contents!
	      ,
	      itsWidth(other.itsWidth),
	      itsHeight(other.itsHeight),
	      itsDepth(other.itsDepth),
	      itsMissingValue(other.itsMissingValue)
	{
	}

	matrix& operator=(const matrix& other)
	{
		itsData = other.itsData;  // Copy contents!
		itsWidth = other.itsWidth;
		itsHeight = other.itsHeight;
		itsDepth = other.itsDepth;
		itsMissingValue = other.itsMissingValue;

		return *this;
	}

	bool operator==(const matrix& other) const
	{
		assert(itsData.size() == other.itsData.size());

		if (itsWidth != other.itsWidth || itsHeight != other.itsHeight || itsDepth != other.itsDepth ||
		    !Compare(itsMissingValue, other.itsMissingValue))
		{
			return false;
		}

		for (size_t i = 0; i < itsData.size(); i++)
		{
			if (!Compare(itsData[i], other.itsData[i]))
			{
				return false;
			}
		}

		return true;
	}

	bool operator!=(const matrix& other) const { return !(*this == other); }
	T& operator[](const size_t& i) { return itsData[i]; }
	std::string ClassName() const { return "himan::matrix"; }
	T At(size_t combinedIndex) const
	{
		assert(itsData.size() > combinedIndex);
		return itsData[combinedIndex];
	}

	T At(size_t x, size_t y, size_t z = 0) const { return itsData[Index(x, y, z)]; }
	std::ostream& Write(std::ostream& file) const
	{
		file << "<" << ClassName() << ">" << std::endl;
		file << "__itsWidth__ " << itsWidth << std::endl;
		file << "__itsHeight__ " << itsHeight << std::endl;
		file << "__itsDepth__ " << itsDepth << std::endl;
		file << "__itsSize__ " << itsData.size() << std::endl;

		PrintData(file, itsData);

		return file;
	}

/**
 * @brief Print information on contents if T == double
 *
 */
#ifndef __NVCC__
	void PrintData(std::ostream& file, const std::vector<double>& theValues) const
	{
		if (!theValues.size())
		{
			file << "__no-data__" << std::endl;
			return;
		}

		assert(theValues.size() > 0);

		double min = 1e38;
		double max = -1e38;
		double sum = 0;
		size_t count = 0;
		size_t missing = 0;
		size_t nan = 0;

		for (size_t i = 0; i < theValues.size(); i++)
		{
			double d = theValues[i];

			// Choosing the lesser evil between two options to compare
			// 1. itsMissingValue that can be of any type
			// 2. kFloatMissing which is of type double but can be different from itsMissingValue even for a double
			// matrix
			// ->this function should not be a member function of Matrix in this form
			if (himan::IsMissing(d))
			{
				missing++;
				continue;
			}
			else if (d != d)
			{
				nan++;
				continue;
			}

			min = (min < d) ? min : d;
			max = (max > d) ? max : d;
			sum += d;
			count++;
		}

		file << "__min__ " << (min == 1e38 ? std::numeric_limits<double>::quiet_NaN() : min) << std::endl;
		file << "__max__ " << (max == -1e38 ? std::numeric_limits<double>::quiet_NaN() : max) << std::endl;
		file << "__avg__ " << (count == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / static_cast<double>(count))
		     << std::endl;
		file << "__missing__ " << missing << std::endl;
		file << "__nan__ " << nan << std::endl;

		if (max == -1e38 || min == 1e38)
		{
			return;
		}

		int binn = 10;
		double binw = (max - min) / 10;

		double binmin = min;
		double binmax = binmin + binw;

		file << "distribution:" << std::endl;

		for (int i = 1; i <= binn; i++)
		{
			if (i == binn) binmax += 0.001;

			size_t count = 0;

			for (size_t j = 0; j < theValues.size(); j++)
			{
				double val = theValues[j];
				// same problem as above with other missing value case
				if (himan::IsMissing(val)) continue;

				if (val >= binmin && val < binmax)
				{
					count++;
				}
			}

			if (i == binn) binmax -= 0.001;

			file << binmin << ":" << binmax << " " << count << std::endl;

			binmin += binw;
			binmax += binw;
		}
	}
#endif

	size_t Size() const { return itsData.size(); }
	size_t SizeX() const { return itsWidth; }
	size_t SizeY() const { return itsHeight; }
	size_t SizeZ() const { return itsDepth; }
	void SizeX(size_t theWidth) { Resize(theWidth, itsHeight, itsDepth); }
	void SizeY(size_t theHeight) { Resize(itsWidth, theHeight, itsDepth); }
	void SizeZ(size_t theDepth) { Resize(itsWidth, itsHeight, theDepth); }
	/**
	 * @brief Resize matrix to given size
	 *
	 * @param theWidth X-size of matrix
	 * @param theHeight Y-size of matrix
	 * @param theDepth Z-size of matrix, if 2D matrix then depth = 1
	 */

	void Resize(size_t theWidth, size_t theHeight, size_t theDepth = 1)
	{
		itsData.resize(theWidth * theHeight * theDepth, itsMissingValue);
		itsWidth = theWidth;
		itsHeight = theHeight;
		itsDepth = theDepth;
	}

	T* ValuesAsPOD()
	{
		assert(itsData.size());
		return itsData.data();
	}

	std::vector<T>& Values() { return itsData; }
	friend std::ostream& operator<<(std::ostream& file, const matrix<T>& ob) { return ob.Write(file); }
	/**
	 * @brief Set value of whole matrix or a slice of it.
	 *
	 * Function will set whole matrix value (or a slice, depending
	 * on the size of the matrix and the size of the argument len)
	 *
	 * This function is thread-safe.
	 *
	 * @param arr Pointer to array of values
	 * @param len Length of the array
	 */

	void Set(T* arr, size_t len)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);
		itsData.assign(arr, arr + len);
	}

	/**
	 * @brief Set value of whole matrix
	 *
	 * The size of the new data must be equal to size of old data
	 *
	 * @param theData
	 */
	void Set(const std::vector<T>& theData)
	{
		assert(itsData.size() == theData.size());
		itsData = theData;
	}

	/**
	 * @brief Set value of a matrix element
	 *
	 * Function will set matrix value in a serialized way -- this
	 * function is thread-safe. Bounds-checking is made for the
	 * index.
	 *
	 * @param x X index
	 * @param y Y index
	 * @param z Z index
	 * @param theValue The value
	 */

	void Set(size_t x, size_t y, size_t z, T theValue)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);
		size_t index = Index(x, y, z);
		assert(index < itsData.size());
		itsData[index] = theValue;
	}

	/**
	 * @brief Set value of a matrix element
	 *
	 * Function will set matrix value in a serialized way -- this
	 * function is thread-safe. No bounds-checking is made for the
	 * index.
	 *
	 * @param theIndex Combined index of the element
	 * @param theValue The value
	 */

	void Set(size_t theIndex, T theValue)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);
		assert(theIndex < itsData.size());
		itsData[theIndex] = theValue;
	}

	/**
	 * @brief Fill matrix with a given value
	 */

	void Fill(T fillValue) { std::fill(itsData.begin(), itsData.end(), fillValue); }
	// Only used for calculating statistics in PrintFloatData()

	void MissingValue(T theMissingValue)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);

		// Replace old missing values in data by new ones
		for (size_t i = 0; i < itsData.size(); i++)
		{
			if(IsMissing(i)) itsData[i] = theMissingValue;
		}
		/*struct Compare_val
		{
  			Compare_val(T x) : x(x) {}
  			bool operator()(T y) const { return Compare(x,y); }

			private:
  			T x;
		};
		Compare_val Missing(itsMissingValue);
		std::replace_if(itsData.begin(),itsData.end(),Missing,theMissingValue);*/
		itsMissingValue = theMissingValue;
	}

	T MissingValue() const { return itsMissingValue; }
	/**
	 * @brief Clear contents of matrix (set size = 0)
	 */

	void Clear()
	{
		itsData.clear();
		itsData.shrink_to_fit();
		itsWidth = 0;
		itsHeight = 0;
		itsDepth = 0;
	}

	bool IsMissing(size_t theIndex) const
	{
		assert(itsData.size() > theIndex);
		return Compare(itsData[theIndex], itsMissingValue);
	}

	bool IsMissing(size_t theX, size_t theY, size_t theZ = 1) const { return IsMissing(Index(theX, theY, theZ)); }
	/**
	 * @brief Calculate missing values in data
	 *
	 * As for the performance of this function, a brief benchmark shows that with
	 * optimization level -O2 it takes a bit more than 1ms to loop through 1M
	 * elements. So Hirlam would be around 1ms with 840480 grid points and global
	 * EC ~5ms with 4.5M grid points.
	 *
	 * The proportion of missing values in the data does not make any difference
	 * in performance.
	 *
	 * @return Number of missing values in data.
	 */

	size_t MissingCount() const
	{
		size_t missing = 0;

		for (size_t i = 0; i < itsData.size(); i++)
		{
			if (IsMissing(i))
			{
				missing++;
			}
		}

		return missing;
	}

	size_t Index(size_t x, size_t y, size_t z) const { return z * itsWidth * itsHeight + y * itsWidth + x; }
   private:
	std::vector<T> itsData;

	size_t itsWidth, itsHeight, itsDepth;

	T itsMissingValue;

	std::mutex itsValueMutex;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsData), CEREAL_NVP(itsWidth), CEREAL_NVP(itsHeight), CEREAL_NVP(itsDepth),
		   CEREAL_NVP(itsMissingValue));
	}
#endif
};

typedef matrix<double> d_matrix_t;

}  // namespace himan

#endif /* MATRIX_H */
