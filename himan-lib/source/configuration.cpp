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
      itsConfigurationFileName(),
      itsConfigurationFileContent(),
      itsAuxiliaryFiles(),
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
      itsCacheLimit(0),
      itsParamFile(),
      itsAsyncExecution(false),
      itsUpdateSSStateTable(true),
      itsUploadStatistics(true),
      itsWriteToDatabase(false),
      itsLegacyWriteMode(false),
      itsWriteToObjectStorageBetweenPluginCalls(false),
      itsWriteStorageType(kLocalFileSystem),
      itsFilenameTemplate(),
      itsPackingType(kSimplePacking),
      itsAllowedMissingValues(std::numeric_limits<size_t>::max()),
      itsForecastTypes({forecast_type(kDeterministic)}),
      itsBaseGrid(nullptr),
      itsSSStateTableName(),
      itsProgramName(kHiman),
      itsValidateMetadata(true)
{
}

configuration::configuration(const configuration& o)
    : itsSourceProducers(o.itsSourceProducers),
      itsOutputFileType(o.itsOutputFileType),
      itsWriteMode(o.itsWriteMode),
      itsFileCompression(o.itsFileCompression),
      itsDatabaseType(o.itsDatabaseType),
      itsConfigurationFileName(o.itsConfigurationFileName),
      itsConfigurationFileContent(o.itsConfigurationFileContent),
      itsAuxiliaryFiles(o.itsAuxiliaryFiles),
      itsReadFromDatabase(o.itsReadFromDatabase),
      itsThreadCount(o.itsThreadCount),
      itsTargetGeomName(o.itsTargetGeomName),
      itsSourceGeomNames(o.itsSourceGeomNames),
      itsStatisticsLabel(o.itsStatisticsLabel),
      itsTargetProducer(o.itsTargetProducer),
      itsUseCuda(o.itsUseCuda),
      itsUseCudaForPacking(o.itsUseCudaForPacking),
      itsUseCudaForUnpacking(o.itsUseCudaForUnpacking),
      itsUseCacheForReads(o.itsUseCacheForReads),
      itsUseCacheForWrites(o.itsUseCacheForWrites),
      itsUseDynamicMemoryAllocation(o.itsUseDynamicMemoryAllocation),
      itsReadAllAuxiliaryFilesToCache(o.itsReadAllAuxiliaryFilesToCache),
      itsCudaDeviceCount(o.itsCudaDeviceCount),
      itsCudaDeviceId(o.itsCudaDeviceId),
      itsForecastStep(o.itsForecastStep),
      itsCacheLimit(o.itsCacheLimit),
      itsParamFile(o.itsParamFile),
      itsAsyncExecution(o.itsAsyncExecution),
      itsUpdateSSStateTable(o.itsUpdateSSStateTable),
      itsUploadStatistics(o.itsUploadStatistics),
      itsWriteToDatabase(o.itsWriteToDatabase),
      itsLegacyWriteMode(o.itsLegacyWriteMode),
      itsWriteToObjectStorageBetweenPluginCalls(o.itsWriteToObjectStorageBetweenPluginCalls),
      itsWriteStorageType(o.itsWriteStorageType),
      itsFilenameTemplate(o.itsFilenameTemplate),
      itsPackingType(o.itsPackingType),
      itsAllowedMissingValues(o.itsAllowedMissingValues),
      itsForecastTypes(o.itsForecastTypes),
      itsLevels(o.itsLevels),
      itsTimes(o.itsTimes),
      itsBaseGrid(o.itsBaseGrid ? std::unique_ptr<grid>(o.itsBaseGrid->Clone()) : nullptr),
      itsSSStateTableName(o.itsSSStateTableName),
      itsProgramName(o.itsProgramName),
      itsValidateMetadata(o.itsValidateMetadata)
{
}

std::ostream& configuration::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsTargetProducer__" << std::endl;
	file << itsTargetProducer;
	file << "__itsSourceProducers__ " << itsSourceProducers.size() << std::endl;
	for (size_t i = 0; i < itsSourceProducers.size(); i++)
	{
		file << itsSourceProducers[i];
	}
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

	file << "__itsConfigurationFileName__ " << itsConfigurationFileName << std::endl;

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
	file << "__itsWriteToObjectStorageBetweenPluginCalls__ " << itsWriteToObjectStorageBetweenPluginCalls << std::endl;
	file << "__itsLegacyWriteMode__" << itsLegacyWriteMode << std::endl;
	file << "__itsWriteStorageType__" << HPFileStorageTypeToString.at(itsWriteStorageType) << std::endl;
	file << "__itsFilenameTemplate__" << itsFilenameTemplate << std::endl;
	file << "__itsPackingType__" << HPPackingTypeToString.at(itsPackingType) << std::endl;
	file << "__itsAllowedMissingValues__" << itsAllowedMissingValues << std::endl;
	file << "__itsSSStateTableName__ " << itsSSStateTableName << std::endl;
	file << "__itsProgramName__ " << static_cast<int>(itsProgramName) << std::endl;
	file << "__itsValidateMetadata__ " << itsValidateMetadata << std::endl;

	for (size_t i = 0; i < itsForecastTypes.size(); i++)
	{
		file << "__itsForecastTypes__ " << i << " " << itsForecastTypes[i] << std::endl;
	}
	for (size_t i = 0; i < itsTimes.size(); i++)
	{
		file << "__itsTimes__ " << i << " " << itsTimes[i] << std::endl;
	}
	for (size_t i = 0; i < itsLevels.size(); i++)
	{
		file << "__itsLevels__ " << i << " " << itsLevels[i] << std::endl;
	}

	if (itsBaseGrid)
	{
		file << "__itsBaseGrid__" << std::endl;
		itsBaseGrid->Write(file);
	}
	else
	{
		file << "__itsBaseGrid__ nullptr" << std::endl;
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
std::string configuration::ConfigurationFileName() const
{
	return itsConfigurationFileName;
}
void configuration::ConfigurationFileName(const std::string& theConfigurationFileName)
{
	itsConfigurationFileName = theConfigurationFileName;
}
std::string configuration::ConfigurationFileContent() const
{
	return itsConfigurationFileContent;
}
void configuration::ConfigurationFileContent(const std::string& theConfigurationFileContent)
{
	itsConfigurationFileContent = theConfigurationFileContent;
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
size_t configuration::CacheLimit() const
{
	return itsCacheLimit;
}
void configuration::CacheLimit(size_t theCacheLimit)
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

HPFileStorageType configuration::WriteStorageType() const
{
	return itsWriteStorageType;
}

void configuration::WriteStorageType(HPFileStorageType theStorageType)
{
	itsWriteStorageType = theStorageType;
}

std::string configuration::FilenameTemplate() const
{
	return itsFilenameTemplate;
}

void configuration::FilenameTemplate(const std::string& theFilenameTemplate)
{
	itsFilenameTemplate = theFilenameTemplate;
}

HPPackingType configuration::PackingType() const
{
	return itsPackingType;
}

void configuration::PackingType(HPPackingType thePackingType)
{
	itsPackingType = thePackingType;
}

size_t configuration::AllowedMissingValues() const
{
	return itsAllowedMissingValues;
}

void configuration::AllowedMissingValues(size_t theAllowedMissingValues)
{
	itsAllowedMissingValues = theAllowedMissingValues;
}

void configuration::BaseGrid(std::unique_ptr<grid> theBaseGrid)
{
	itsBaseGrid = std::move(theBaseGrid);
}

const grid* configuration::BaseGrid() const
{
	if (itsBaseGrid)
	{
		return itsBaseGrid.get();
	}

	return nullptr;
}

std::vector<forecast_type> configuration::ForecastTypes() const
{
	return itsForecastTypes;
}
void configuration::ForecastTypes(const std::vector<forecast_type>& theForecastTypes)
{
	itsForecastTypes = theForecastTypes;
}

std::vector<forecast_time> configuration::Times() const
{
	return itsTimes;
}
void configuration::Times(const std::vector<forecast_time>& theTimes)
{
	itsTimes = theTimes;
}

std::vector<level> configuration::Levels() const
{
	return itsLevels;
}
void configuration::Levels(const std::vector<level>& theLevels)
{
	itsLevels = theLevels;
}

std::string configuration::SSStateTableName() const
{
	return itsSSStateTableName;
}

void configuration::SSStateTableName(const std::string& theSSStateTableName)
{
	itsSSStateTableName = theSSStateTableName;
}

HPProgramName configuration::ProgramName() const
{
	return itsProgramName;
}

void configuration::ProgramName(HPProgramName theName)
{
	itsProgramName = theName;
}

bool configuration::WriteToObjectStorageBetweenPluginCalls() const
{
	return itsWriteToObjectStorageBetweenPluginCalls;
}

void configuration::WriteToObjectStorageBetweenPluginCalls(bool flag)
{
	itsWriteToObjectStorageBetweenPluginCalls = flag;
}

bool configuration::ValidateMetadata() const
{
	return itsValidateMetadata;
}

void configuration::ValidateMetadata(bool val)
{
	itsValidateMetadata = val;
}
