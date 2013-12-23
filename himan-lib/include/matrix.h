/**
 * @file matrix.h
 *
 * @date Dec 14, 2012
 * @author partio
 *
 * @brief 2-3d matrix to store data. Does not have any mathematical implications of matrices.
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <mutex>
#include "himan_common.h"

namespace himan
{

template<class T>
class matrix
{
public:
	matrix() : itsData(0), itsWidth(0), itsHeight(0), itsDepth(0) {}

	matrix(size_t theWidth, size_t theHeight, size_t theDepth = 1)
		: itsData(theWidth* theHeight* theDepth)
		, itsWidth(theWidth)
		, itsHeight(theHeight)
		, itsDepth(theDepth)
	{
	}

	matrix(const matrix& other)
		: itsData(other.itsData) // Copy contents!
		, itsWidth(other.itsWidth)
		, itsHeight(other.itsHeight)
		, itsDepth(other.itsDepth)
		, itsMissingValue(other.itsMissingValue)
	{
	}

	matrix& operator=(const matrix& other)
	{
		itsData = other.itsData; // Copy contents!
		itsWidth = other.itsWidth;
		itsHeight = other.itsHeight;
		itsDepth = other.itsDepth;
		itsMissingValue = other.itsMissingValue;

		return *this;
	}
	
	std::string ClassName() const
	{
		return "himan::matrix";
	}

	T& At(size_t combinedIndex)
	{
		return itsData[combinedIndex];
	}

	T& At(size_t x, size_t y, size_t z = 0)
	{
		return itsData[Index(x, y, z)];
	}

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

	void PrintData(std::ostream& file, std::vector<double> theValues) const
	{

		if (!itsData.size())
		{
			file << "__no-data__" << std::endl;
			return;
		}

		double min = 1e38;
		double max = -1e38;
		double sum = 0;
		long missing = 0;

		for (size_t i = 0; i < theValues.size(); i++)
		{
			double d = theValues[i];

			if (d == kFloatMissing)
			{
				missing++;
				continue;
			}

			min = (min < d) ? min : d;
			max = (max > d) ? max : d;
			sum += d;
		}

		file << "__min__ " << min << std::endl;
		file << "__max__ " << max << std::endl;
		file << "__avg__ " << sum / static_cast<double> (theValues.size()) << std::endl;
		file << "__missing__ " << missing << std::endl;
	}

	/**
	 * @brief Print information on contents if T == grid
	 *
	 */
/*
	void PrintData(std::ostream& file, std::vector<std::shared_ptr<grid>> theValues) const
	{

		if (!itsData.size())
		{
			file << "__no-data__" << std::endl;
			return;
		}

		for (size_t i = 0; i < theValues.size(); i++)
		{
			file << *theValues[i];
		}
	}
*/
	size_t Size() const
	{
		return itsData.size();
	}

	size_t SizeX() const
	{
		return itsWidth;
	}

	size_t SizeY() const
	{
		return itsHeight;
	}

	size_t SizeZ() const
	{
		return itsDepth;
	}

	void SizeX(size_t theWidth)
	{
		itsWidth = theWidth;
		Resize(itsWidth, itsHeight, itsDepth);
	}

	void SizeY(size_t theHeight)
	{
		itsHeight = theHeight;
		Resize(itsWidth, itsHeight, itsDepth);
	}

	void SizeZ(size_t theDepth)
	{
		itsDepth = theDepth;
		Resize(itsWidth, itsHeight, itsDepth);
	}

	/**
	 * @brief Resize matrix to given size
	 *
	 * @param theWidth X-size of matrix
	 * @param theHeight Y-size of matrix
	 * @param theDepth Z-size of matrix, if 2D matrix then depth = 1
	 */

	void Resize(size_t theWidth, size_t theHeight, size_t theDepth = 1)
	{
		itsData.resize(theWidth * theHeight * theDepth);
		itsWidth = theWidth;
		itsHeight = theHeight;
		itsDepth = theDepth;
	}

	/**
	 * @brief Return const pointer to data as POD
     * @return const pointer to data
     */
	const T* ValuesAsPOD() const
	{
		return &itsData[0];
	}

	/**
	 * @brief Return reference to data (in c++ style)
     * @return const reference to data
     */
	
	const std::vector<T>& Values() const
	{
		return itsData;
	}

	friend std::ostream& operator<<(std::ostream& file, const matrix<T> & ob)
	{
		return ob.Write(file);
	}

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
	 *
	 * @return Always true
	 * @todo Return void ?
	 */

	bool Set(T* arr, size_t len)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);

		assert(itsData.size() == len);
		
		if (itsData.size() != len)
		{
			return false;
		}

		itsData.assign(arr, arr + len);
		return true;
	}

	/**
	 * @brief Set value of whole matrix
	 * 
	 * The size of the new data must be equal to size of old data
	 * 
     * @param theData
     * @return 
     */
	bool Set(const std::vector<T>& theData)
	{
		assert(itsData.size() == theData.size());

		if (itsData.size() != theData.size())
		{
			return false;
		}
		itsData = theData;

		return true;
	}

	/**
	 * @brief Set value of a matrix element
	 *
	 * Function will set matrix value in a serialized way -- this
	 * function is thread-safe. No bounds-checking is made for the
	 * index.
	 *
	 * @param x X index
	 * @param y Y index
	 * @param z Z index
	 * @param theValue The value
	 *
	 * @return Always true
	 * @todo Return void ?
	 */

	bool Set(size_t x, size_t y, size_t z, T theValue)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);

		assert(Index(x,y,z) < itsData.size());

		if (Index(x,y,z) >= itsData.size())
		{
			return false;
		}
		
		itsData[Index(x,y,z)] = theValue;

		return true;
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
	 *
	 * @return Always true
	 * @todo Return void ?
	 */

	bool Set(size_t theIndex, T theValue)
	{
		std::lock_guard<std::mutex> lock(itsValueMutex);

		assert(theIndex < itsData.size());

		if (theIndex >= itsData.size())
		{
			return false;
		}
		
		itsData[theIndex] = theValue;

		return true;
	}

	/**
	 * @brief Fill matrix with a given value
	 */

	void Fill(T fillValue)
	{
		std::fill(itsData.begin(),itsData.end(),fillValue);
	}

	// Only used for calculating statistics in PrintFloatData()

	void MissingValue(T theMissingValue)
	{
		itsMissingValue = theMissingValue;
	}

	T MissingValue()
	{
		return itsMissingValue;
	}

	/**
	 * @brief Clear contents of matrix (set size = 0)
     */

	void Clear()
	{
		itsData.clear();
		itsWidth=0;
		itsHeight=0;
		itsDepth=0;
	}

private:

	inline size_t Index(size_t x, size_t y, size_t z) const
	{
		return z * itsWidth * itsHeight + y * itsWidth + x;
	}

	std::vector<T> itsData;

	size_t itsWidth, itsHeight, itsDepth;

	T itsMissingValue;
	
	std::mutex itsValueMutex;
};


typedef matrix <double> d_matrix_t;
typedef matrix <unsigned char> uc_matrix_t;

} // namespace himan

#endif /* MATRIX_H */
