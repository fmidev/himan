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

    matrix(const matrix& other) = delete;
    matrix& operator=(const matrix& other) = delete;

    std::string ClassName() const
    {
        return "himan::matrix";
    }

    HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

    // Perhaps dangerous: return reference with no const

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

    std::ostream& Write(std::ostream& file) const
    {
        file << "<" << ClassName() << " " << Version() << ">" << std::endl;
        file << "__itsWidth__ " << itsWidth << std::endl;
        file << "__itsHeight__ " << itsHeight << std::endl;
        file << "__itsDepth__ " << itsDepth << std::endl;
        file << "__itsSize__ " << itsData.size() << std::endl;

        PrintFloatData(file, itsData);

        return file;
    }

    /**
     * @brief Print information on contents if T == double
     *
     */

    void PrintFloatData(std::ostream& file, std::vector<double> theValues) const
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

        // kFloatMissing should be substituted with itsMissingValue

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
        file << "__avg__ " << sum / theValues.size() << std::endl;
        file << "__missing__ " << missing << std::endl;
    }

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

    const T* Values() const
    {
        return &itsData[0];
    }

    friend std::ostream& operator<<(std::ostream& file, matrix<T> & ob)
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
        itsData.assign(arr, arr + len);
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

private:

    inline size_t Index(size_t x, size_t y, size_t z) const
    {
        return z * itsWidth * itsHeight + y * itsWidth + x;
    }

    std::vector<T> itsData;

    T itsMissingValue;

    size_t itsWidth, itsHeight, itsDepth;

    std::mutex itsValueMutex;
};


typedef matrix <double> d_matrix_t;

} // namespace himan

#endif /* MATRIX_H */
