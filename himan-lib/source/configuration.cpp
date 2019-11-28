/**
 * @file configuration.cpp
 *
 */

#include "configuration.h"
#include "plugin_factory.h"

using namespace himan;

configuration::configuration()
    : itsSourceProducers(),
      itsOutputFileType(kGRIB1),
      itsWriteMode(kAllGridsToAFile),
      itsFileCompression(kNoCompression),
      itsDatabaseType(kRadon),
      itsConfigurationFile(),
      itsAuxiliaryFiles(),
      itsOriginTime(),
      itsReadFromDatabase(true),
      itsThreadCount(-1),
      itsTargetGeomName(),
      itsSourceGeomNames(),
      itsStatisticsLabel(),
      itsTargetProducer(),
      itsUseCuda(true),
      itsUseCudaForPacking(true),
      itsUseCudaForUnpacking(true),
      itsUseCacheForReads(true),
      itsUseCacheForWrites(true),
      itsUseDynamicMemoryAllocation(false),
      itsReadAllAuxiliaryFilesToCache(true),
      itsCudaDeviceCount(-1),
      itsCudaDeviceId(0),
      itsForecastStep(),
      itsCacheLimit(-1),
      itsParamFile(),
      itsAsyncExecution(false),
      itsUpdateSSStateTable(true),
      itsUploadStatistics(true),
      itsWriteToDatabase(false),
      itsLegacyWriteMode(false)
{
}

std::ostream& configuration::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << itsTargetProducer;

	file << "__itsOutputFileType__ " << HPFileTypeToString.at(itsOutputFileType) << std::endl;
	file << "__itsWriteMode__ " << HPWriteModeToString.at(itsWriteMode) << std::endl;
	file << "__itsFileCompression__ " << HPFileCompressionToString.at(itsFileCompression) << std::endl;
	file << "__itsUseCuda__ " << itsUseCuda << std::endl;
	file << "__itsReadFromDatabase__ " << itsReadFromDatabase << std::endl;

	file << "__itsThreadCount__ " << itsThreadCount << std::endl;

	file << "__itsTargetGeomName__ " << itsTargetGeomName << std::endl;

	for (size_t i = 0; i < itsSourceGeomNames.size(); i++)
	{
		file << "__itsSourceGeomName__ " << itsSourceGeomNames[i] << std::endl;
	}

	file << "__itsStatisticsLabel__ " << itsStatisticsLabel << std::endl;

	file << "__itsConfigurationFile__ " << itsConfigurationFile << std::endl;

	file << "__itsCudaDeviceId__ " << itsCudaDeviceId << std::endl;
	file << "__itsUseCudaForPacking__ " << itsUseCudaForPacking << std::endl;
	file << "__itsUseCudaForUnpacking__ " << itsUseCudaForUnpacking << std::endl;

	file << "__itsUseCacheForReads__ " << itsUseCacheForReads << std::endl;
	file << "__itsUseCacheForWrites__ " << itsUseCacheForWrites << std::endl;

	file << "__itsForecastStep__ " << itsForecastStep << std::endl;
	file << "__itsCacheLimit__ " << itsCacheLimit << std::endl;
	file << "__itsUseDynamicMemoryAllocation__ " << itsUseDynamicMemoryAllocation << std::endl;
	file << "__itsReadAllAuxiliaryFilesToCache__" << itsReadAllAuxiliaryFilesToCache << std::endl;

	for (size_t i = 0; i < itsAuxiliaryFiles.size(); i++)
	{
		file << "__itsAuxiliaryFiles__ " << i << " " << itsAuxiliaryFiles[i] << std::endl;
	}

	file << "__itsParamFile__ " << itsParamFile << std::endl;
	file << "__itsAsyncExecution__ " << itsAsyncExecution << std::endl;
	file << "__itsUpdateSSStateTable__" << itsUpdateSSStateTable << std::endl;
	file << "__itsUploadStatistics__" << itsUploadStatistics << std::endl;
	file << "__itsWriteToDatabase__" << itsWriteToDatabase << std::endl;
	file << "__itsLegacyWriteMode__" << itsLegacyWriteMode << std::endl;

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

HPFileType configuration::OutputFileType() const
{
	return itsOutputFileType;
}
void configuration::OutputFileType(HPFileType theOutputFileType)
{
	itsOutputFileType = theOutputFileType;
}
HPWriteMode configuration::WriteMode() const
{
	return itsWriteMode;
}
void configuration::WriteMode(HPWriteMode theWriteMode)
{
	itsWriteMode = theWriteMode;
}
HPFileCompression configuration::FileCompression() const
{
	return itsFileCompression;
}
void configuration::FileCompression(HPFileCompression theFileCompression)
{
	itsFileCompression = theFileCompression;
}
bool configuration::ReadFromDatabase() const
{
	return itsReadFromDatabase;
}
void configuration::ReadFromDatabase(bool theReadFromDatabase)
{
	itsReadFromDatabase = theReadFromDatabase;
}

bool configuration::UseCuda() const
{
	return itsUseCuda && HaveCuda();
}
void configuration::UseCuda(bool theUseCuda)
{
	itsUseCuda = theUseCuda;
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
}
void configuration::ConfigurationFile(const std::string& theConfigurationFile)
{
	itsConfigurationFile = theConfigurationFile;
}
const std::vector<producer>& configuration::SourceProducers() const
{
	return itsSourceProducers;
}
void configuration::SourceProducers(const std::vector<producer>& theSourceProducers)
{
	itsSourceProducers = theSourceProducers;
}
const producer& configuration::SourceProducer(size_t theIndexNumber) const
{
	return itsSourceProducers[theIndexNumber];
}
const producer& configuration::TargetProducer() const
{
	return itsTargetProducer;
}
void configuration::TargetProducer(const producer& theTargetProducer)
{
	itsTargetProducer = theTargetProducer;
}
void configuration::StatisticsLabel(const std::string& theStatisticsLabel)
{
	itsStatisticsLabel = theStatisticsLabel;
}
std::string configuration::StatisticsLabel() const
{
	return itsStatisticsLabel;
}
bool configuration::UseCudaForUnpacking() const
{
	return itsUseCudaForUnpacking;
}
void configuration::UseCudaForUnpacking(bool theUseCudaForUnpacking)
{
	itsUseCudaForUnpacking = theUseCudaForUnpacking;
}

bool configuration::UseCudaForPacking() const
{
	return itsUseCudaForPacking;
}
void configuration::UseCudaForPacking(bool theUseCudaForPacking)
{
	itsUseCudaForPacking = theUseCudaForPacking;
}
bool configuration::UseCacheForReads() const
{
	return itsUseCacheForReads;
}
void configuration::UseCacheForReads(bool theUseCacheForReads)
{
	itsUseCacheForReads = theUseCacheForReads;
}
bool configuration::UseCacheForWrites() const
{
	return itsUseCacheForWrites;
}
void configuration::UseCacheForWrites(bool theUseCacheForWrites)
{
	itsUseCacheForWrites = theUseCacheForWrites;
}

void configuration::SourceGeomNames(const std::vector<std::string>& theNames)
{
	itsSourceGeomNames = theNames;
}
std::vector<std::string> configuration::SourceGeomNames() const
{
	return itsSourceGeomNames;
}
void configuration::CudaDeviceCount(int theCudaDeviceCount)
{
	itsCudaDeviceCount = theCudaDeviceCount;
}
bool configuration::HaveCuda() const
{
	return (itsCudaDeviceCount > 0);
}
int configuration::CudaDeviceCount() const
{
	return itsCudaDeviceCount;
}
int configuration::CudaDeviceId() const
{
	return itsCudaDeviceId;
}
void configuration::CudaDeviceId(int theCudaDeviceId)
{
	itsCudaDeviceId = theCudaDeviceId;
}
time_duration configuration::ForecastStep() const
{
	return itsForecastStep;
}
void configuration::ForecastStep(const time_duration& theForecastStep)
{
	itsForecastStep = theForecastStep;
}
HPDatabaseType configuration::DatabaseType() const
{
	return itsDatabaseType;
}
void configuration::DatabaseType(HPDatabaseType theDatabaseType)
{
	itsDatabaseType = theDatabaseType;
}
std::string configuration::TargetGeomName() const
{
	return itsTargetGeomName;
}
void configuration::TargetGeomName(const std::string& theTargetGeomName)
{
	itsTargetGeomName = theTargetGeomName;
}
int configuration::CacheLimit() const
{
	return itsCacheLimit;
}
void configuration::CacheLimit(int theCacheLimit)
{
	itsCacheLimit = theCacheLimit;
}
bool configuration::UseDynamicMemoryAllocation() const
{
	return itsUseDynamicMemoryAllocation;
}
void configuration::UseDynamicMemoryAllocation(bool theUseDynamicMemoryAllocation)
{
	itsUseDynamicMemoryAllocation = theUseDynamicMemoryAllocation;
}

bool configuration::ReadAllAuxiliaryFilesToCache() const
{
	return itsReadAllAuxiliaryFilesToCache;
}
void configuration::ReadAllAuxiliaryFilesToCache(bool theReadAllAuxiliaryFilesToCache)
{
	itsReadAllAuxiliaryFilesToCache = theReadAllAuxiliaryFilesToCache;
}
std::string configuration::ParamFile() const
{
	return itsParamFile;
}
void configuration::ParamFile(const std::string& theParamFile)
{
	itsParamFile = theParamFile;
}
bool configuration::AsyncExecution() const
{
	return itsAsyncExecution;
}
void configuration::AsyncExecution(bool theAsyncExecution)
{
	itsAsyncExecution = theAsyncExecution;
}

bool configuration::UpdateSSStateTable() const
{
	return itsUpdateSSStateTable;
}

void configuration::UpdateSSStateTable(bool theUpdateSSStateTable)
{
	itsUpdateSSStateTable = theUpdateSSStateTable;
}

bool configuration::UploadStatistics() const
{
	return itsUploadStatistics;
}

void configuration::UploadStatistics(bool theUploadStatistics)
{
	itsUploadStatistics = theUploadStatistics;
}

bool configuration::WriteToDatabase() const
{
	return itsWriteToDatabase;
}

void configuration::WriteToDatabase(bool theWriteToDatabase)
{
	itsWriteToDatabase = theWriteToDatabase;
}

bool configuration::LegacyWriteMode() const
{
	return itsLegacyWriteMode;
}

void configuration::LegacyWriteMode(bool theLegacyWriteMode)
{
	itsLegacyWriteMode = theLegacyWriteMode;
}
