/**
 * @file configuration.h
 *
 * @brief Class to hold configuration information read from configuration file.
 *
 * Class will read metadata from configuration file and create an info instance
 * from it.
 *
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "producer.h"
#include "time_duration.h"

namespace himan
{
class logger;

class configuration
{
   public:
	friend class json_parser;

	configuration();
	virtual ~configuration() = default;

	configuration(const configuration&) = default;
	configuration& operator=(const configuration&) = default;

	/**
	 * @return Class name
	 */

	std::string ClassName() const
	{
		return "himan::configuration";
	}
	std::ostream& Write(std::ostream& file) const;

	/**
	 * @return List of auxiliary file names found in the configuration file
	 */

	std::vector<std::string> AuxiliaryFiles() const;
	void AuxiliaryFiles(const std::vector<std::string>& theAuxiliaryFiles);

	/**
	 * @return Filetype of created file. One of: grib1, grib2, querydata, netcdf
	 */

	HPFileType OutputFileType() const;
	void OutputFileType(HPFileType theOutputFileType);

	/**
	 * @brief Set source producers (create iterator)
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 * @param theSourceProducers Vector of producers
	 */

	void SourceProducers(const std::vector<producer>& theSourceProducers);
	const std::vector<producer>& SourceProducers() const;

	/**
	 * @brief Get current source producer
	 *
	 * @return const reference to the current source producer
	 */

	const producer& SourceProducer(size_t theIndexNumber) const;

	/**
	 * @brief Get target producer
	 *
	 * @return const reference to to the target producer
	 */

	const producer& TargetProducer() const;

	/**
	 * @brief Set target producer
	 *
	 * @param theTargetProducer New target producer
	 */

	void TargetProducer(const producer& theTargetProducer);

	/**
	 * @brief Decide how many fields to write in a single file
	 *
	 */

	void WriteMode(HPWriteMode theWriteMode);
	HPWriteMode WriteMode() const;

	/**
	 * @brief Enable or disable output file compression
	 *
	 * @param theFileCompression set output file compression to gzip, bzip2 or none
	 */

	void FileCompression(HPFileCompression theFileCompression);
	HPFileCompression FileCompression() const;

	/**
	 * @brief Enable or disable reading of source data from Neons
	 * @param theReadFromDatabase true = read data from neons, false = use only data specified in command
	 * line (auxiliary files)
	 */

	void ReadFromDatabase(bool theReadFromDatabase);
	bool ReadFromDatabase() const;

	/**
	 * @brief Top level function for CUDA calculation
	 * @return True if CUDA can be used (does not tell IF it's used)
	 */
	bool UseCuda() const;
	void UseCuda(bool theUseCuda);

	void ThreadCount(short theThreadCount);
	short ThreadCount() const;

	std::string ConfigurationFile() const;
	void ConfigurationFile(const std::string& theConfigurationFile);

	void StatisticsLabel(const std::string& theStatisticsLabel);
	std::string StatisticsLabel() const;

	/**
	 * @brief Top level function for CUDA grib packing and unpacking.
	 * @return True if CUDA can be used (does not tell IF it's used)
	 */

	bool UseCudaForPacking() const;
	void UseCudaForPacking(bool theUseCudaForPacking);

	bool UseCudaForUnpacking() const;
	void UseCudaForUnpacking(bool theUseCudaForUnpacking);

	bool UseCacheForReads() const;
	void UseCacheForReads(bool theUseCacheForReads);

	bool UseCacheForWrites() const;
	void UseCacheForWrites(bool theUseCacheForWrites);

	void SourceGeomNames(const std::vector<std::string>& theNames);
	std::vector<std::string> SourceGeomNames() const;

	/**
	 * @brief Store number of CUDA devices found
	 */

	void CudaDeviceCount(int theCudaDeviceCount);

	/**
	 * @brief Check if we have any CUDA-enabled devices available
	 * @return True if there is at least one CUDA enable device present
	 */

	bool HaveCuda() const;

	/**
	 * @return Number of CUDA enabled devices found
	 */

	int CudaDeviceCount() const;

	/**
	 * @return Id of the selected CUDA device
	 */

	int CudaDeviceId() const;
	void CudaDeviceId(int theCudaDeviceId);

	/**
	 * @brief Return the value if key 'step'.
	 *
	 * @return Value of 'step' if present
	 */

	time_duration ForecastStep() const;
	void ForecastStep(const time_duration& theForecastStep);

	HPDatabaseType DatabaseType() const;
	void DatabaseType(HPDatabaseType theDatabaseType);

	std::string TargetGeomName() const;
	void TargetGeomName(const std::string& theTargetGeomName);

	int CacheLimit() const;
	void CacheLimit(int theCacheLimit);

	bool UseDynamicMemoryAllocation() const;
	void UseDynamicMemoryAllocation(bool theUseDynamicMemoryAllocation);

	bool ReadAllAuxiliaryFilesToCache() const;
	void ReadAllAuxiliaryFilesToCache(bool theReadAllAuxiliaryFilesToCache);

	std::string ParamFile() const;
	void ParamFile(const std::string& theParamFile);

	bool AsyncExecution() const;
	void AsyncExecution(bool theAsyncExecution);

	bool UpdateSSStateTable() const;
	void UpdateSSStateTable(bool theUpdateSSStateTable);

	bool UploadStatistics() const;
	void UploadStatistics(bool theUploadStatistics);

	bool WriteToDatabase() const;
	void WriteToDatabase(bool theWriteToDatabase);

	bool LegacyWriteMode() const;
	void LegacyWriteMode(bool theLegacyWriteMode);

   protected:
	std::vector<producer> itsSourceProducers;

	HPFileType itsOutputFileType;
	HPWriteMode itsWriteMode;
	HPFileCompression itsFileCompression;
	HPDatabaseType itsDatabaseType;

	std::string itsConfigurationFile;
	std::vector<std::string> itsAuxiliaryFiles;
	std::string itsOriginTime;

	bool itsReadFromDatabase;
	short itsThreadCount;
	std::string itsTargetGeomName;
	std::vector<std::string> itsSourceGeomNames;
	std::string itsStatisticsLabel;

	producer itsTargetProducer;

	bool itsUseCuda;
	bool itsUseCudaForPacking;
	bool itsUseCudaForUnpacking;
	bool itsUseCacheForReads;
	bool itsUseCacheForWrites;
	bool itsUseDynamicMemoryAllocation;
	bool itsReadAllAuxiliaryFilesToCache;

	int itsCudaDeviceCount;
	int itsCudaDeviceId;

	time_duration itsForecastStep;

	int itsCacheLimit;
	std::string itsParamFile;
	bool itsAsyncExecution;
	bool itsUpdateSSStateTable;
	bool itsUploadStatistics;
	bool itsWriteToDatabase;
	bool itsLegacyWriteMode;
};

inline std::ostream& operator<<(std::ostream& file, const configuration& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#endif /* CONFIGURATION_H */
