/*
 * configuration.cpp
 *
 *  Created on: Nov 26, 2012
 *      Author: partio
 */

#include "configuration.h"
#include "logger_factory.h"

using namespace himan;

configuration::configuration()
{

    Init();
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("configuration"));
    itsInfo = std::shared_ptr<info> (new info());

}

std::ostream& configuration::Write(std::ostream& file) const
{

    file << "<" << ClassName() << " " << Version() << ">" << std::endl;

    for (size_t i = 0; i < itsPlugins.size(); i++)
    {
        file << "__itsPlugins__ " << itsPlugins[i] << std::endl;
    }

    file << "__itsSourceProducer__ " << itsSourceProducer << std::endl;
    file << "__itsTargetProducer__ " << itsTargetProducer << std::endl;
    file << "__itsOutputFileType__ " << itsOutputFileType << std::endl;
    file << "__itsWholeFileWrite__ " << itsWholeFileWrite << std::endl;
    file << "__itsUseCuda__ " << itsUseCuda << std::endl;
    file << "__itsFileWaitTimeout__ " << itsFileWaitTimeout << std::endl;
    file << "__itsReadDataFromDatabase__ " << itsReadDataFromDatabase << std::endl;

    file << "__itsThreadCount__ " << itsThreadCount << std::endl;

    file << "__itsNi__ " << itsNi << std::endl;
    file << "__itsNj__ " << itsNj << std::endl;

    file << "__itsGeomName__ " << itsGeomName << std::endl;
    file << "__itsScanningMode__ " << itsScanningMode << std::endl;

    file << *itsInfo;

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

std::vector<std::string> configuration::Plugins() const
{
    return itsPlugins;
}

void configuration::Plugins(const std::vector<std::string>& thePlugins)
{
    itsPlugins = thePlugins;
}

void configuration::Init()
{
    itsOutputFileType = kQueryData;
    itsSourceProducer = kHPMissingInt;
    itsTargetProducer = kHPMissingInt;
    itsWholeFileWrite = false;
    itsReadDataFromDatabase = true;
    itsUseCuda = true;
    itsFileWaitTimeout = 0;
    itsLeadingDimension = kTimeDimension;
    itsThreadCount = -1;
    itsGeomName = "";
    itsScanningMode = kUnknownScanningMode;
}

HPFileType configuration::OutputFileType() const
{
    return itsOutputFileType;
}

void configuration::OutputFileType(HPFileType theOutputFileType)
{
	itsOutputFileType = theOutputFileType;
}

unsigned int configuration::SourceProducer() const
{
    return itsSourceProducer;
}

unsigned int configuration::TargetProducer() const
{
    return itsTargetProducer;
}

void configuration::SourceProducer(unsigned int theSourceProducer)
{
    itsSourceProducer = theSourceProducer;
}

void configuration::TargetProducer(unsigned int theTargetProducer)
{
    itsTargetProducer = theTargetProducer;
}

bool configuration::WholeFileWrite() const
{
    return itsWholeFileWrite;
}

void configuration::WholeFileWrite(bool theWholeFileWrite)
{
    itsWholeFileWrite = theWholeFileWrite;
}

size_t configuration::Ni() const
{
    return itsNi;
}

void configuration::Ni(size_t theNi)
{
    itsNi = theNi;
}

size_t configuration::Nj() const
{
    return itsNj;
}

void configuration::Nj(size_t theNj)
{
    itsNj = theNj;
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
    return itsUseCuda;
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

HPScanningMode configuration::ScanningMode() const
{
	return itsScanningMode;
}

void configuration::ScanningMode(HPScanningMode theScanningMode)
{
	itsScanningMode = theScanningMode;
}

std::string configuration::ConfigurationFile() const
{
	return itsConfigurationFile;
};

void configuration::ConfigurationFile(const std::string& theConfigurationFile)
{
	itsConfigurationFile = theConfigurationFile;
}
