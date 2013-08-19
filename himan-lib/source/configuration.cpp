/**
 * @file configuration.cpp
 *
 * @date Nov 26, 2012
 * @author partio
 */

#include "configuration.h"
#include "logger_factory.h"

using namespace himan;

configuration::configuration() : itsSourceProducerIterator(new producer_iter())
{

	Init();
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("configuration"));

}

configuration::configuration(const configuration& other)
{

	itsOutputFileType = other.itsOutputFileType;
	itsConfigurationFile = other.itsConfigurationFile;
	itsAuxiliaryFiles = other.itsAuxiliaryFiles;
	itsOriginTime = other.itsOriginTime;

	itsFileWriteOption = other.itsFileWriteOption;
	itsReadDataFromDatabase = other.itsReadDataFromDatabase;

	itsFileWaitTimeout = other.itsFileWaitTimeout;
	itsUseCuda = other.itsUseCuda;
	itsUseCudaForPacking = other.itsUseCudaForPacking;
	itsUseCache = other.itsUseCache;

	itsLeadingDimension = other.itsLeadingDimension;
	itsThreadCount = other.itsThreadCount;

	itsTargetGeomName = other.itsTargetGeomName;
	itsSourceGeomName = other.itsSourceGeomName;
	
	itsTargetProducer = other.itsTargetProducer;
	//itsSourceProducers = other.itsSourceProducers;

	itsStatisticsLabel = other.itsStatisticsLabel;

	itsSourceProducerIterator = std::unique_ptr<producer_iter> (new producer_iter(*other.itsSourceProducerIterator));
	itsSourceProducerIterator->Set(other.itsSourceProducerIterator->Index());

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("configuration"));
}

std::ostream& configuration::Write(std::ostream& file) const
{

	file << "<" << ClassName() << " " << Version() << ">" << std::endl;

   // file << "__itsSourceProducer__ " << itsSourceProducer << std::endl;
	file << itsTargetProducer;

	file << "__itsOutputFileType__ " << itsOutputFileType << std::endl;
	file << "__itsFileWriteOption__ " << itsFileWriteOption << std::endl;
	file << "__itsUseCuda__ " << itsUseCuda << std::endl;
	file << "__itsFileWaitTimeout__ " << itsFileWaitTimeout << std::endl;
	file << "__itsReadDataFromDatabase__ " << itsReadDataFromDatabase << std::endl;
	file << "__itsLeadingDimension__ " << itsLeadingDimension << std::endl;

	file << "__itsThreadCount__ " << itsThreadCount << std::endl;

	file << "__itsTargetGeomName__ " << itsTargetGeomName << std::endl;
	file << "__itsSourceGeomName__ " << itsSourceGeomName << std::endl;
	
	file << "__itsStatisticsLabel__ " << itsStatisticsLabel << std::endl;

	file << "__itsConfigurationFile__ " << itsConfigurationFile << std::endl;

	file << "__itsUseCudaForPacking__ " << itsUseCudaForPacking << std::endl;

	file << "__itsUseCache__ " << itsUseCache << std::endl;
	
	for (size_t i = 0; i < itsAuxiliaryFiles.size(); i++)
	{
		file << "__itsAuxiliaryFiles__ " << i << " " << itsAuxiliaryFiles[i] << std::endl;
	}
	return file;
}

std::vector<std::string> configuration::AuxiliaryFiles() const
{
	return itsAuxiliaryFiles;
}

void configuration::AuxiliaryFiles(const std::vector<std::string>& theAuxiliaryFiles)
{
	itsAuxiliaryFiles = theAuxiliaryFiles;
}

void configuration::Init()
{
	itsOutputFileType = kQueryData;
	itsFileWriteOption = kSingleFile;
	itsReadDataFromDatabase = true;
	itsUseCuda = true;
	itsFileWaitTimeout = 0;
	itsLeadingDimension = kTimeDimension;
	itsThreadCount = -1;
	itsTargetGeomName = "";
	itsSourceGeomName = "";
	itsConfigurationFile = "";
	itsUseCudaForPacking = true;
	itsUseCache = true;
}

HPFileType configuration::OutputFileType() const
{
	return itsOutputFileType;
}

void configuration::OutputFileType(HPFileType theOutputFileType)
{
	itsOutputFileType = theOutputFileType;
}

HPFileWriteOption configuration::FileWriteOption() const
{
	return itsFileWriteOption;
}

void configuration::FileWriteOption(HPFileWriteOption theFileWriteOption)
{
	itsFileWriteOption = theFileWriteOption;
}

bool configuration::ReadDataFromDatabase() const
{
	return itsReadDataFromDatabase;
}

void configuration::ReadDataFromDatabase(bool theReadDataFromDatabase)
{
	itsReadDataFromDatabase = theReadDataFromDatabase;
}

unsigned short configuration::FileWaitTimeout() const
{
	return itsFileWaitTimeout;
}

void configuration::FileWaitTimeout(unsigned short theFileWaitTimeout)
{
	itsFileWaitTimeout = theFileWaitTimeout;
}

bool configuration::UseCuda() const
{
	return itsUseCuda && HaveCuda();
}

void configuration::UseCuda(bool theUseCuda)
{
	itsUseCuda = theUseCuda;
}

HPDimensionType configuration::LeadingDimension() const
{
	return itsLeadingDimension;
}

short configuration::ThreadCount() const
{
	return itsThreadCount;
}

void configuration::ThreadCount(short theThreadCount)
{
	itsThreadCount = theThreadCount;
}

std::string configuration::ConfigurationFile() const
{
	return itsConfigurationFile;
};

void configuration::ConfigurationFile(const std::string& theConfigurationFile)
{
	itsConfigurationFile = theConfigurationFile;
}

void configuration::SourceProducers(const std::vector<producer> theSourceProducers)
{
	itsSourceProducerIterator = std::unique_ptr<producer_iter> (new producer_iter(theSourceProducers));
}

bool configuration::SourceProducer(const producer& theSourceProducer)
{
	return itsSourceProducerIterator->Set(theSourceProducer);
}

bool configuration::NextSourceProducer() const
{
	return itsSourceProducerIterator->Next();
}

bool configuration::FirstSourceProducer() const
{
	return itsSourceProducerIterator->First();
}

producer configuration::SourceProducer() const
{
	return itsSourceProducerIterator->At();
}

producer configuration::TargetProducer() const
{
	return itsTargetProducer;
}

void configuration::StatisticsLabel(const std::string& theStatisticsLabel)
{
	itsStatisticsLabel = theStatisticsLabel;
}

std::string configuration::StatisticsLabel() const
{
	return itsStatisticsLabel;
}

bool configuration::UseCudaForPacking() const
{
	return itsUseCudaForPacking;
}

void configuration::UseCudaForPacking(bool theUseCudaForPacking)
{
	itsUseCudaForPacking = theUseCudaForPacking;
}

bool configuration::UseCache() const
{
	return itsUseCache;
}

void configuration::UseCache(bool theUseCache)
{
	itsUseCache = theUseCache;
}

std::string configuration::SourceGeomName() const
{
	return itsSourceGeomName;
}

void configuration::StoreCudaDeviceCount()
{
	std::shared_ptr<plugin::pcuda> p = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	itsCudaDeviceCount = p->DeviceCount();
}

bool configuration::HaveCuda() const
{
	return (itsCudaDeviceCount > 0);
}

short configuration::CudaDeviceCount() const
{
	return itsCudaDeviceCount;
}
