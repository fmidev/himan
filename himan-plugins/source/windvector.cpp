/**
 * @file windvector.cpp
 *
 *  Created on: Jan 21, 2013
 *  @author aaltom
 */

#include "windvector.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"
#include <math.h>
#include "NFmiRotatedLatLonArea.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "pcuda.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#undef HAVE_CUDA

#ifdef HAVE_CUDA
namespace himan
{
namespace plugin
{
namespace windvector_cuda
{
void doCuda(const float* Tin, float TBase, const float* Pin, float TScale, float* TPout, size_t N, float PConst, unsigned short index);
}
}
}
#endif

const double kPi = 3.14159265359;
const double kRadToDeg = 57.295779513082; // 180 / PI
const double kDegToRad = 0.017453292; // PI / 180

windvector::windvector() : itsUseCuda(false)
{
	itsClearTextFormula = "WV = ";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvector"));

}

void windvector::Process(shared_ptr<configuration> theConfiguration)
{

	shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	if (c && c->HaveCuda())
	{
		string msg = "I possess the powers of CUDA ";

		if (!theConfiguration->UseCuda())
		{
			msg += ", but I won't use them";
		}
		else
		{
			msg += ", and I'm not afraid to use them";
			itsUseCuda = true;
		}

		itsLogger->Info(msg);

	}

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(theConfiguration->ThreadCount());

	boost::thread_group g;

	/*
	 * The target information is parsed from the configuration file.
	 */

	shared_ptr<info> theTargetInfo = theConfiguration->Info();

	/*
	 * Get producer information from neons if whole_file_write is false.
	 */

	if (!theConfiguration->WholeFileWrite())
	{
		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string,string> prodInfo = n->ProducerInfo(theTargetInfo->Producer().Id());

		if (!prodInfo.empty())
		{
			producer prod(theTargetInfo->Producer().Id());

			prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
			prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
			prod.Name(prodInfo["name"]);

			theTargetInfo->Producer(prod);
		}

	}

	/*
	 * Set target parameter to windvector
	 * - name ICEIND-N
	 * - univ_id 480
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from theConfiguration but why bother?)
	 *
	 */

	vector<param> theParams;

	param requestedDFParam("DF-MS", 22);
	param requestedFFParam("FF-MS", 21);
	requestedFFParam.GribDiscipline(0);
	requestedFFParam.GribCategory(2);
	requestedFFParam.GribParameter(1);

	param requestedDDParam("DD-D", 20);
	requestedDDParam.GribDiscipline(0);
	requestedDDParam.GribCategory(2);
	requestedDDParam.GribParameter(0);

	//theRequestedParam.GribParameter(103);
    //    theRequestedParam.GribTableVersion(204);

	theParams.push_back(requestedDFParam);
	theParams.push_back(requestedFFParam);
	theParams.push_back(requestedDDParam);

	theTargetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	theTargetInfo->Create();

	/*
	 * Initialize parent class functions for dimension handling
	 */

	Dimension(theConfiguration->LeadingDimension());
	FeederInfo(theTargetInfo->Clone());
	FeederInfo()->Param(requestedDFParam);

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > theTargetInfos;

	theTargetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		theTargetInfos[i] = theTargetInfo->Clone();

		boost::thread* t = new boost::thread(&windvector::Run,
								this,
								theTargetInfos[i],
								theConfiguration,
								i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (theConfiguration->WholeFileWrite())
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		theTargetInfo->FirstTime();

		string theOutputFile = "himan_" + theTargetInfo->Param().Name() + "_" + theTargetInfo->Time().OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theConfiguration->OutputFileType(), false, theOutputFile);

	}
}

void windvector::Run(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void windvector::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param UParam("U-MS");
	param VParam("V-MS");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvectorThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->Param(param("DF-MS"));

	shared_ptr<info> DDInfo = myTargetInfo->Clone();
	DDInfo->Param(param("DD-D"));

	shared_ptr<info> FFInfo = myTargetInfo->Clone();
	FFInfo->Param(param("FF-MS"));

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		DDInfo->Level(myTargetInfo->Level());
		FFInfo->Level(myTargetInfo->Level());

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		myTargetInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());
		DDInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());
		FFInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());

		shared_ptr<info> UInfo;
		shared_ptr<info> VInfo;

		try
		{
			// Source info for U
			UInfo = theFetcher->Fetch(theConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 UParam);
				
			// Source info for V
			VInfo = theFetcher->Fetch(theConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 VParam);
				
		}
		catch (HPExceptionType e)
		{
		
			switch (e)
			{
			case kFileDataNotFound:
				itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
				myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value
				DDInfo->Data()->Fill(kFloatMissing);
				FFInfo->Data()->Fill(kFloatMissing);
				continue;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
			}
		}

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> UGrid = UInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> VGrid = VInfo->ToNewbaseGrid();

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (myTargetInfo->GridAndAreaEquals(UInfo) &&
							myTargetInfo->GridAndAreaEquals(VInfo));

		myTargetInfo->ResetLocation();
		DDInfo->ResetLocation();
		FFInfo->ResetLocation();

		targetGrid->Reset();

		bool needRotLatLonGridRotation = (myTargetInfo->Projection() == kRotatedLatLonProjection && UInfo->UVRelativeToGrid());
		bool needStereographicGridRotation = (myTargetInfo->Projection() == kStereographicProjection && UInfo->UVRelativeToGrid());

		while (myTargetInfo->NextLocation() && DDInfo->NextLocation() && FFInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double U = kFloatMissing;
			double V = kFloatMissing;

			InterpolateToPoint(targetGrid, UGrid, equalGrids, U);
			InterpolateToPoint(targetGrid, VGrid, equalGrids, V);

			if (U == kFloatMissing || V == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			if (needRotLatLonGridRotation)
			{
				/*
				 * 1. Get coordinates of current grid point in earth-relative form
				 * 2. Get coordinates of current grid point in grid-relative form
				 * 3. Call function UVToEarthRelative() that transforms U and V from grid-relative
				 *    to earth-relative
				 */

				const point regPoint(targetGrid->LatLon());
				const point rotPoint(reinterpret_cast<NFmiRotatedLatLonArea*> (targetGrid->Area())->ToRotLatLon(regPoint.ToNFmiPoint()));

				point regUV = UVToEarthRelative(regPoint, rotPoint, UInfo->SouthPole(), point(U,V));

				// Wind speed should the same with both forms of U and V

				assert(fabs((U*U+V*V) - (regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);

				U = regUV.X();
				V = regUV.Y();
			}
			else if (needStereographicGridRotation)
			{
				throw runtime_error(ClassName() + ": I don't know how to rotate stereographic projections (yet)!");
			}

			double FF = sqrt(U*U + V*V);

			double DD = 0;

			if (FF > 0)
			{
				DD = round(kRadToDeg * atan2(U,V) + 180.0); // Rounding DD
			}

			DDInfo->Value(DD);
			FFInfo->Value(FF);

			if (U > 360)
			{
				U = U - 360;
			}

			if (U < 0)
			{
				U = U + 360;
			}
                        
			double windVector = round(U/10) + 100 * round(V);

			if (!myTargetInfo->Value(windVector))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (!theConfiguration->WholeFileWrite())
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			theWriter->ToFile(myTargetInfo->Clone(), theConfiguration->OutputFileType(), true);
		}
	}
}

himan::point windvector::UVToEarthRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV)
{

	himan::point newSouthPole;

	if (southPole.Y() > 0)
	{
		newSouthPole.Y(-southPole.Y());
		newSouthPole.X(0);
	}
	else
	{
		newSouthPole = southPole;
	}

	double sinPoleY = sin(kDegToRad * (newSouthPole.Y()+90)); // zsyc
	double cosPoleY = cos(kDegToRad * (newSouthPole.Y()+90)); // zcyc

	//double zsxreg = sin(kDegToRad * regPoint.X());
	//double zcxreg = cos(kDegToRad * regPoint.X());
	//double zsyreg = sin(kDegToRad * regPoint.Y());
	double cosRegY = cos(kDegToRad * regPoint.Y()); // zcyreg

	double zxmxc = kDegToRad * (regPoint.X() - newSouthPole.X());
	double sinxmxc = sin(zxmxc); // zsxmxc
	double cosxmxc = cos(zxmxc); // zcxmxc

	double sinRotX = sin(kDegToRad * rotPoint.X()); // zsxrot
	double cosRotX = cos(kDegToRad * rotPoint.X()); // zcxrot
	double sinRotY = sin(kDegToRad * rotPoint.Y()); // zsyrot
	double cosRotY = cos(kDegToRad * rotPoint.Y()); // zcyrot

	double PA = cosxmxc * cosRotX + cosPoleY * sinxmxc * sinRotX;
	double PB = cosPoleY * sinxmxc * cosRotX * sinRotY + sinPoleY * sinxmxc * cosRotY - cosxmxc * sinRotX * sinRotY;
	double PC = (-sinPoleY) * sinRotX / cosRegY;
	double PD = (cosPoleY * cosRotY - sinPoleY * cosRotX * sinRotY) / cosRegY;

	double U = PA * UV.X() + PB * UV.Y();
	double V = PC * UV.X() + PD * UV.Y();

	return point(U,V);
}

himan::point windvector::UVToGridRelative(const himan::point& regPoint, const himan::point& rotPoint, const himan::point& southPole, const himan::point& UV)
{

	himan::point newSouthPole;

	if (southPole.Y() > 0)
	{
		newSouthPole.Y(-southPole.Y());
		newSouthPole.X(0);
	}
	else
	{
		newSouthPole = southPole;
	}

	double sinPoleY = sin(kDegToRad * (newSouthPole.Y()+90)); // zsyc
	double cosPoleY = cos(kDegToRad * (newSouthPole.Y()+90)); // zcyc

	//double sinRegX = sin(kDegToRad * regPoint.X()); // zsxreg
	//double cosRegX = cos(kDegToRad * regPoint.X()); // zcxreg
	double sinRegY = sin(kDegToRad * regPoint.Y()); // zsyreg
	double cosRegY = cos(kDegToRad * regPoint.Y()); // zcyreg

	double zxmxc = kDegToRad * (regPoint.X() - newSouthPole.X());
	double sinxmxc = sin(zxmxc); // zsxmxc
	double cosxmxc = cos(zxmxc); // zcxmxc

	double sinRotX = sin(kDegToRad * rotPoint.X()); // zsxrot
	double cosRotX = cos(kDegToRad * rotPoint.X()); // zcxrot
	//double sinRotY = sin(kDegToRad * rotPoint.Y()); // zsyrot
	double cosRotY = cos(kDegToRad * rotPoint.Y()); // zcyrot

	double PA = cosPoleY * sinxmxc * sinRotX + cosxmxc * cosRotX;
	double PB = cosPoleY * cosxmxc * sinRegY * sinRotX - sinPoleY * cosRegY * sinRotX - sinxmxc * sinRegY * cosRotX;
	double PC = sinPoleY * sinxmxc / cosRotY;
	double PD = (sinPoleY * cosxmxc * sinRegY + cosPoleY * cosRegY) / cosRotY;

	double U = PA * UV.X() + PB * UV.Y();
	double V = PC * UV.X() + PD * UV.Y();

	return point(U,V);
}
