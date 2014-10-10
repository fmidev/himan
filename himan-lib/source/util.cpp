/**
 * @file util.cpp
 *
 * @brief Different utility functions and classes in a namespace
 *
 * @date Dec 1, 2012
 * @author partio
 */

#include "util.h"
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <iomanip>
#include <NFmiStereographicArea.h>
#include "level.h"
#include "param.h"
#include "forecast_time.h"

using namespace himan;
using namespace std;

string util::MakeFileName(HPFileWriteOption fileWriteOption, shared_ptr<const info> info)
{

	ostringstream fileName;
	ostringstream base;

	base.str(".");

	// For neons get base directory

	if (fileWriteOption == kNeons)
	{
		char* path;

		path = std::getenv("MASALA_PROCESSED_DATA_BASE");

		if (path != NULL)
		{
			base.str("");
			base << path;
		}
		else
		{
			cout << "Warning::util MASALA_PROCESSED_DATA_BASE not set" << endl;
		}

		base <<	"/"
			 << info->Producer().Centre()
			 << "_"
			 << info->Producer().Process()
			 << "/"
			 << info->Time().OriginDateTime()->String("%Y%m%d%H%M")
			 << "/"
			 << info->Time().Step();
	}

	// Create a unique file name when creating multiple files from one info

	if (fileWriteOption == kNeons || fileWriteOption == kMultipleFiles)
	{
		fileName	<< base.str()
					<< "/"
					<< info->Param().Name()
					<< "_"
					<< HPLevelTypeToString.at(info->Level().Type())
					<< "_"
					<< info->Level().Value()
					<< "_"
					<< HPProjectionTypeToString.at(info->Grid()->Projection())
					<< "_"
					<< info->Ni()
					<< "_"
					<< info->Nj()
					<< "_0_"
					<< setw(3)
					<< setfill('0')
					<< info->Time().Step()
					;

	}
	else
	{
		// TODO!
		
		fileName	<< base
					<< "/"
					<< "TODO.file";
	}

	return fileName.str();

}

himan::HPFileType util::FileType(const string& theFile)
{

	using namespace std;

	ifstream f(theFile.c_str(), ios::in | ios::binary);

	long keywordLength = 4;

	char content[keywordLength];
	
	f.read(content, keywordLength);

	HPFileType ret = kUnknownFile;

	if (strncmp(content, "GRIB", 4) == 0)
	{
		ret = kGRIB;
	}
	else if (strncmp(content, "CDF", 3) == 0)
	{
		ret = kNetCDF;
	}
	else
	{
		// Not GRIB or NetCDF, keep on searching

		keywordLength = 5;

		char content[keywordLength];

		f.read(content, keywordLength);

		if (strncmp(content, "QINFO", 5) == 0)
		{
			ret = kQueryData;
		}

	}

	return ret;
}


// copied from http://stackoverflow.com/questions/236129/splitting-a-string-in-c and modified a bit

vector<string> util::Split(const string& s, const std::string& delims, bool fill)
{

	vector<string> orig_elems;

	boost::split(orig_elems, s, boost::is_any_of(delims));

	if (!fill || orig_elems.size() == 0)
	{
		return orig_elems;
	}

	vector<string> filled_elems;
	vector<string> splitted_elems;

	vector<string>::iterator it;

	for (it = orig_elems.begin(); it != orig_elems.end(); )
	{

		boost::split(splitted_elems, *it, boost::is_any_of("-"));

		if (splitted_elems.size() == 2)
		{
			it = orig_elems.erase(it);

			for (int i = boost::lexical_cast<int> (splitted_elems[0]); i <= boost::lexical_cast<int> (splitted_elems[1]); i++)
			{
				filled_elems.push_back(boost::lexical_cast<string> (i));
			}
		}
		else
		{
			++it;
		}
	}

	vector<string> all_elems;

	all_elems.reserve(orig_elems.size() + filled_elems.size());

	all_elems.insert(all_elems.end(), orig_elems.begin(), orig_elems.end());
	all_elems.insert(all_elems.end(), filled_elems.begin(), filled_elems.end());

	return all_elems;
}

string util::Join(const vector<string> &elements, const string& delim)
{
	ostringstream s;

	for (size_t i = 0; i < elements.size(); i++)
	{
		if (i != 0)
		{
			s << delim;
		}
		s << elements[i];
	}
	
	return s.str();
}

pair<point,point> util::CoordinatesFromFirstGridPoint(const point& firstPoint, size_t ni, size_t nj, double di, double dj, HPScanningMode scanningMode)
{
	double dni = static_cast<double> (ni) - 1;
	double dnj = static_cast<double> (nj) - 1;

	point bottomLeft, topRight, topLeft, bottomRight;

	switch (scanningMode)
	{
	case kBottomLeft:
		bottomLeft = firstPoint;
		topRight = point(bottomLeft.X() + dni*di, bottomLeft.Y() + dnj*dj);
		break;

	case kTopLeft: // +x-y
		bottomRight = point(firstPoint.X() + dni*di, firstPoint.Y() - dnj*dj);
		bottomLeft = point(bottomRight.X() - dni*di, firstPoint.Y() - dnj*dj);
		topRight = point(bottomLeft.X() + dni*di, bottomLeft.Y() + dnj*dj);
		break;

	case kTopRight: // -x-y
		topRight = firstPoint;
		bottomLeft = point(topRight.X() - dni*di, topRight.Y() - dnj*dj);
		break;

	case kBottomRight: // -x+y
		topLeft = point(firstPoint.X() - dni*di, firstPoint.Y() + dnj*dj);
		bottomLeft = point(firstPoint.X() - dni*di, topLeft.Y() - dnj*dj);
		topRight = point(bottomLeft.X() + dni*di, bottomLeft.Y() + dnj*dj);
		break;

	default:
		throw runtime_error("util::CoordinatesFromFirstGridPoint(): Calculating first grid point when scanning mode is unknown");
		break;

	}

	return pair<point,point> (bottomLeft, topRight);
}

pair<point,point> util::CoordinatesFromFirstGridPoint(const point& firstPoint, double orientation, size_t ni, size_t nj, double xSizeInMeters, double ySizeInMeters)
{

	double xWidthInMeters = (static_cast<double> (ni)-1.) * xSizeInMeters;
	double yWidthInMeters = (static_cast<double> (nj)-1.) * ySizeInMeters;

	NFmiStereographicArea a(static_cast<NFmiPoint> (firstPoint),
								xWidthInMeters,
								yWidthInMeters,
								orientation,
								NFmiPoint(0,0),
								NFmiPoint(1,1),
								90.);

	point bottomLeft(a.BottomLeftLatLon());
	point topRight(a.TopRightLatLon());

	return pair<point,point> (bottomLeft, topRight);

}

tuple<double,double,double,double> util::EarthRelativeUVCoefficients(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole)
{
	point newSouthPole;

	if (southPole.Y() > 0)
	{
		newSouthPole.Y(-southPole.Y());
		newSouthPole.X(0);
	}
	else
	{
		newSouthPole = southPole;
	}

	double southPoleY = constants::kDeg * (newSouthPole.Y()+90);
	double sinPoleY, cosPoleY;

	sincos(southPoleY, &sinPoleY, &cosPoleY);

	double cosRegY = cos(constants::kDeg * regPoint.Y()); // zcyreg

	double zxmxc = constants::kDeg * (regPoint.X() - newSouthPole.X());

	double sinxmxc, cosxmxc;

	sincos(zxmxc, &sinxmxc, &cosxmxc);

	double rotXRad = constants::kDeg * rotPoint.X();
	double rotYRad = constants::kDeg * rotPoint.Y();

	double sinRotX, cosRotX;
	sincos(rotXRad, &sinRotX, &cosRotX);

	double sinRotY, cosRotY;
	sincos(rotYRad, &sinRotY, &cosRotY);

	double PA = cosxmxc * cosRotX + cosPoleY * sinxmxc * sinRotX;
	double PB = cosPoleY * sinxmxc * cosRotX * sinRotY + sinPoleY * sinxmxc * cosRotY - cosxmxc * sinRotX * sinRotY;
	double PC = (-sinPoleY) * sinRotX / cosRegY;
	double PD = (cosPoleY * cosRotY - sinPoleY * cosRotX * sinRotY) / cosRegY;

	return make_tuple(PA,PB,PC,PD);


}

tuple<double,double,double,double> util::GridRelativeUVCoefficients(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole)
{

	point newSouthPole;

	if (southPole.Y() > 0)
	{
		newSouthPole.Y(-southPole.Y());
		newSouthPole.X(0);
	}
	else
	{
		newSouthPole = southPole;
	}

	double sinPoleY = sin(constants::kDeg * (newSouthPole.Y()+90)); // zsyc
	double cosPoleY = cos(constants::kDeg * (newSouthPole.Y()+90)); // zcyc

	//double sinRegX = sin(constants::kDeg * regPoint.X()); // zsxreg
	//double cosRegX = cos(constants::kDeg * regPoint.X()); // zcxreg
	double sinRegY = sin(constants::kDeg * regPoint.Y()); // zsyreg
	double cosRegY = cos(constants::kDeg * regPoint.Y()); // zcyreg

	double zxmxc = constants::kDeg * (regPoint.X() - newSouthPole.X());
	double sinxmxc = sin(zxmxc); // zsxmxc
	double cosxmxc = cos(zxmxc); // zcxmxc

	double sinRotX = sin(constants::kDeg * rotPoint.X()); // zsxrot
	double cosRotX = cos(constants::kDeg * rotPoint.X()); // zcxrot
	//double sinRotY = sin(constants::kDeg * rotPoint.Y()); // zsyrot
	double cosRotY = cos(constants::kDeg * rotPoint.Y()); // zcyrot

	double PA = cosPoleY * sinxmxc * sinRotX + cosxmxc * cosRotX;
	double PB = cosPoleY * cosxmxc * sinRegY * sinRotX - sinPoleY * cosRegY * sinRotX - sinxmxc * sinRegY * cosRotX;
	double PC = sinPoleY * sinxmxc / cosRotY;
	double PD = (sinPoleY * cosxmxc * sinRegY + cosPoleY * cosRegY) / cosRotY;

	return make_tuple(PA,PB,PC,PD);

}

point util::UVToGeographical(double longitude, const point& stereoUV)
{

	double U,V;

	if (stereoUV.X() == 0 && stereoUV.Y() == 0)
	{
		return point(0,0);
	}

	double sinLon, cosLon;

	sincos(longitude * constants::kDeg, &sinLon, &cosLon);
	
	U = stereoUV.X() * cosLon + stereoUV.Y() * sinLon;
	V = -stereoUV.X() * sinLon + stereoUV.Y() * cosLon;

	return point(U,V);
}

double util::ToPower(double value, double power)
{
  double divisor = 1.0;

  while(value < 0)
  {
	divisor /= power;
	value++;
  }

  while(value > 0)
  {
	divisor *= power;
	value--;
  }

  return divisor;
}

matrix<double> util::Filter2D(matrix<double>& A, matrix<double>& B)
{
	// find center position of kernel (half of kernel size)
	matrix<double> ret(A.SizeX(),A.SizeY(),1);
	double convolution_value; // accumulated value of the convolution at a given grid point in A
	double kernel_weight_sum; // accumulated value of the kernel weights in B that are used to compute the convolution at given point A

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	
	// check if data contains missing values
	if (A.MissingCount()==0) // if no missing values in the data we can use a faster algorithm
	{
		// calculate for inner field
		// the weights are used as given on input
		// assert (sum(B) == 1)
		for(int i=kCenterX; i < ASizeX-kCenterX; ++i)	// rows
		{
			for(int j=kCenterY; j < ASizeY-kCenterY; ++j)	// columns
			{
		  		convolution_value = 0;
		  		for(int m=0; m < BSizeX; ++m)	// kernel rows
		  		{
	            			int mm = BSizeX - 1 - m;	// row index of flipped kernel
	           			for(int n=0; n < BSizeY; ++n)	// kernel columns
	            			{
						int nn = BSizeY - 1 - n;  // column index of flipped kernel

			                	// index of input signal, used for checking boundary
                	                	int ii = i + (m - kCenterX);
                	                	int jj = j + (n - kCenterY);
	               				convolution_value += A.At(ii,jj,0) * B.At(mm,nn,0);
					}
	                	}
	                	ret.Set(i,j,0,convolution_value);
			}
		}
	
		// treat boundaries separately
		// weights get adjusted so that the sum of weights for the active part of the kernel remains 1
 		// calculate for upper boundary
		for(int i=0; i < ASizeX; ++i)              // rows
	  	{
	    		for(int j=0; j < kCenterY; ++j)          // columns
	    		{
				convolution_value = 0;
				kernel_weight_sum = 0;
	        		for(int m=0; m < BSizeX; ++m)     // kernel rows
	       			{
	            			int mm = BSizeX - 1 - m;      // row index of flipped kernel
	
					for(int n=0; n < BSizeY; ++n) // kernel columns
	            			{
	                			int nn = BSizeY - 1 - n;  // column index of flipped kernel	    	
						// index of input signal, used for checking boundary

						int ii = i + (m - kCenterX);
	                			int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if( ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY )
						{
			  				convolution_value += A.At(ii,jj,0) * B.At(mm,nn,0);
			  				kernel_weight_sum += B.At(mm,nn,0);
						}
	            			}
	 			}
	        		ret.Set(i,j,0,convolution_value/kernel_weight_sum);
	    		}
	  	}

		// calculate for lower boundary
		for(int i=0; i < ASizeX; ++i)              // rows
	  	{
	    		for(int j=ASizeY-kCenterY; j < ASizeY; ++j)          // columns
	    		{
				convolution_value = 0;
				kernel_weight_sum = 0;
	        		for(int m=0; m < BSizeX; ++m)     // kernel rows
	        		{
	            			int mm = BSizeX - 1 - m;      // row index of flipped kernel
	
	            			for(int n=0; n < BSizeY; ++n) // kernel columns
	            			{
	                			int nn = BSizeY - 1 - n;  // column index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
	               		 		int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if( ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY )
						{
			  				convolution_value += A.At(ii,jj,0) * B.At(mm,nn,0);
			  				kernel_weight_sum += B.At(mm,nn,0);
						}
	            			}
	        		}
	        		ret.Set(i,j,0,convolution_value/kernel_weight_sum);
	    		}
	  	}

		// calculate for left boundary
		for(int i=0; i < kCenterX; ++i)              // rows
	  	{
	    		for(int j=0; j < ASizeY; ++j)          // columns
	    		{
				convolution_value = 0;
				kernel_weight_sum = 0;
	        		for(int m=0; m < BSizeX; ++m)     // kernel rows
	        		{
	            			int mm = BSizeX - 1 - m;      // row index of flipped kernel
	
	            			for(int n=0; n < BSizeY; ++n) // kernel columns
	            			{
	                			int nn = BSizeY - 1 - n;  // column index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
	                			int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if( ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY )
						{
			  				convolution_value += A.At(ii,jj,0) * B.At(mm,nn,0);
			  				kernel_weight_sum += B.At(mm,nn,0);
						}
	            			}
	        		}
	        		ret.Set(i,j,0,convolution_value/kernel_weight_sum);
	    		}
	  	}

		// calculate for right boundary
		for(int i=ASizeX-kCenterX; i < ASizeX; ++i)              // rows
	  	{
	    		for(int j=0; j < ASizeY; ++j)          // columns
	    		{
				convolution_value = 0;
				kernel_weight_sum = 0;
	        		for(int m=0; m < BSizeX; ++m)     // kernel rows
	        		{
	            			int mm = BSizeX - 1 - m;      // row index of flipped kernel
	
	            			for(int n=0; n < BSizeY; ++n) // kernel columns
	            			{
	                			int nn = BSizeY - 1 - n;  // column index of flipped kernel

	                			// index of input signal, used for checking boundary
	                 	                int ii = i + (m - kCenterX);
	                 	                int jj = j + (n - kCenterY);

				                // ignore input samples which are out of bound
				                if( ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY )
						{
			  				convolution_value += A.At(ii,jj,0) * B.At(mm,nn,0);
			  				kernel_weight_sum += B.At(mm,nn,0);
						}
	            			}
	        		}
	        		ret.Set(i,j,0,convolution_value/kernel_weight_sum);
	    		}
	  	}
	}
	else	// data contains missing values
	{
		std::cout << "Data contains missing values -> Choosing slow algorithm." << std::endl;
		double kernel_missing_count;
		for(int i=0; i < ASizeX; ++i)              // rows
	  	{
	    		for(int j=0; j < ASizeY; ++j)          // columns
	    		{
				convolution_value = 0;
				kernel_weight_sum = 0;
				kernel_missing_count = 0;
	        		for(int m=0; m < BSizeX; ++m)     // kernel rows
	        		{
	            			int mm = BSizeX- 1 - m;      // row index of flipped kernel
	
	            			for(int n=0; n < BSizeY; ++n) // kernel columns
	            			{
	                			int nn = BSizeY - 1 - n;  // column index of flipped kernel
	
	                			// index of input signal, used for checking boundary
	                 	                int ii = i + (m - kCenterX);
	                       	                int jj = j + (n - kCenterY);

						// ignore input samples which are out of bound
						if( ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY )
						{
			  				if( A.IsMissing(ii,jj,0) )
			  				{
			    					kernel_missing_count++;
			    					continue;
			  				}
	    
			  				convolution_value += A.At(ii,jj,0) * B.At(mm,nn,0);
			  				kernel_weight_sum += B.At(mm,nn,0);
						}
	            			}
	        		}
	        		if (kernel_missing_count < 3)
				{
		  			ret.Set(i,j,0,convolution_value/kernel_weight_sum); // if less than three values are missing in kernel fill gap by average of surrounding values
				}
				else
				{
		  			ret.Set(i,j,0,himan::kFloatMissing); // if three or more values are missing in kernel put kFloatMissing
				}
	    		}
	  	}
	}
	return ret;
}
	
#ifdef HAVE_CUDA
void util::Unpack(initializer_list<shared_ptr<grid>> grids)
{
	vector<cudaStream_t*> streams;

	for (auto it = grids.begin(); it != grids.end(); ++it)
	{
		shared_ptr<grid> tempGrid = *it;

		if (!tempGrid->IsPackedData())
		{
			// Safeguard: This particular info does not have packed data
			continue;
		}

		assert(tempGrid->PackedData()->ClassName() == "simple_packed");

		double* arr = 0;
		size_t N = tempGrid->PackedData()->unpackedLength;

		assert(N > 0);

		cudaStream_t* stream = new cudaStream_t;
		CUDA_CHECK(cudaStreamCreate(stream));
		streams.push_back(stream);

#ifdef MALLOC_SIMPLE
		assert(tempGrid->Data()->Size() == N);
		arr = const_cast<double*> (tempGrid->Data()->ValuesAsPOD());
#ifdef MALLOC_REGISTER
		CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*> (arr), sizeof(double) * N, 0));
#endif

#else
		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&arr), sizeof(double) * N));
#endif
		assert(arr);

		dynamic_pointer_cast<simple_packed> (tempGrid->PackedData())->Unpack(arr, N, stream);

#if not defined MALLOC_SIMPLE
		tempGrid->Data()->Set(arr, N);

		CUDA_CHECK(cudaFreeHost(arr));
#endif

#if defined MALLOC_REGISTER
		CUDA_CHECK(cudaHostUnregister(arr));
#endif

		tempGrid->PackedData()->Clear();
	}

	for (size_t i = 0; i < streams.size(); i++)
	{
		CUDA_CHECK(cudaStreamDestroy(*streams[i]));

	}
}
#endif
