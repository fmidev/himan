/*
 * matrix.h
 *
 *  Created on: Dec 14, 2012
 *      Author: partio
 *
 * 2d or 3d matrix to store data. Does not have any mathematical
 * implication of matrices.
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <boost/thread.hpp>
#include "hilpee_common.h"
#include <vector>

namespace hilpee
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

		std::string ClassName() const
		{
			return "hilpee::matrix";
		}

		HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		// Perhaps dangerous functions: return reference with no const

		T& operator()(size_t x, size_t y, size_t z = 1)
		{
			assert(itsData.size() > 0);
			return itsData[Index(x, y, z)];
		}
		T& operator()(size_t combinedIndex)
		{
			assert(itsData.size() > 0);
			return itsData[combinedIndex];
		}


		// Copied from newbase, is this good ?

		T& At(size_t combinedIndex)
		{
			try
			{
				return itsData[combinedIndex];
			}
			catch (std::exception& e)
			{
				throw std::runtime_error(e.what());
			}

			throw std::runtime_error("Stupid compiler");
		}

		T& At(size_t x, size_t y, size_t z = 1)
		{
			try
			{
				return itsData[Index(x, y, z)];
			}
			catch (std::exception& e)
			{
				throw std::runtime_error(e.what());
			}

			throw std::runtime_error("Stupid compiler");
		}

		void Data(T* arr, size_t len)
		{
			assert(itsData.size() == len);
			itsData.assign(arr, arr + len);
		}

		std::ostream& Write(std::ostream& file) const
		{
			file << "<" << ClassName() << " " << Version() << ">" << std::endl;
			file << "__itsWidth__ " << itsWidth << std::endl;
			file << "__itsHeight__ " << itsHeight << std::endl;
			file << "__itsDepth__ " << itsDepth << std::endl;
			file << "__itsSize__ " << itsData.size() << std::endl;

			DoStuff(file, itsData);

			return file;
		}

		void DoStuff(std::ostream& file, std::vector<double> theValues) const
		{
			double min = 1e38;
			double max = -1e38;
			double sum = 0;
			long missing = 0;

			// kFloatMissing should be substituted with itsMissingValue

			for (double d : theValues)
			{
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
			file << "__avg__ " << sum / theValues.size() << std::endl;
			file << "__missing__ " << missing << std::endl;
		}

		size_t Size() const
		{
			return itsData.size() ;
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

		void Resize(size_t x, size_t y, size_t z = 1)
		{
			itsData.resize(x * y * z);
			itsWidth = x;
			itsHeight = y;
			itsDepth = z;
		}

		const T* Values() const
		{
			return &itsData[0];
		}

		friend std::ostream& operator<<(std::ostream& file, matrix<T> & ob)
		{
			return ob.Write(file);
		}

		bool Set(size_t theIndex, T theValue)
		{
			boost::mutex::scoped_lock lock(itsValueMutex);

			itsData[theIndex] = theValue;

			return true;
		}

		// Only used for calculating statistics in DoStuff()

		void MissingValue(T theMissingValue)
		{
			itsMissingValue = theMissingValue;
		}
		T MissingValue()
		{
			return itsMissingValue;
		}

	private:

		size_t Index(size_t x, size_t y, size_t z) const
		{
			return z * itsWidth * itsHeight + y * itsWidth + x;
		}

		std::vector<T> itsData;
		T itsMissingValue;
		size_t itsWidth, itsHeight, itsDepth;

		boost::mutex itsValueMutex;
};


typedef matrix <double> d_matrix_t;

} // namespace hilpee

#endif /* MATRIX_H */
