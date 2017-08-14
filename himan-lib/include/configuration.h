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

#include "info.h"

namespace himan
{
class logger;

class configuration
{
   public:
	friend class json_parser;

	configuration();
	virtual ~configuration() {}
	configuration(const configuration& other);
	configuration& operator=(const configuration& other) = delete;

	/**
	 * @return Class name
	 */

	std::string ClassName() const { return "himan::configuration"; }
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

	void SourceProducers(std::vector<producer> theSourceProducers);

	/**
	 * @brief Get current source producer
	 *
	 * @return const reference to the current source producer
	 */

	const producer& SourceProducer(size_t theIndexNumber = kHPMissingInt) const;

	/**
	 * @brief Search for source producer given as argument, if found set iterator
	 * position to that and return true
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 * @param theSourceProducer
	 * @return True if found, false if not
	 */

	bool SourceProducer(const producer& theSourceProducer);

	/**
	 * @brief Advance iterator by one
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 * @return True if not at last source producer
	 */

	bool NextSourceProducer();

	/**
	 * @brief Goto first source producer
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 * @return True if iterator has at least one source producer
	 */

	bool FirstSourceProducer();

	/**
	 * @brief Reset source producer iterator
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 */

	void ResetSourceProducer();

	/**
	 * @brief Return number of source producers
	 *
	 * @return Number of source producers
	 */

	size_t SizeSourceProducers() const;

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
	 * @brief Enable or disable writing to one or many file
	 *
	 * @param theWholeFileWrite true = all data is written to one file, false = each data descriptor
	 * combination is written to separate file
	 */

	void FileWriteOption(HPFileWriteOption theFileWriteOption);
	HPFileWriteOption FileWriteOption() const;

	/**
	 * @brief Enable or disable output file compression
	 *
	 * @param theFileCompression set output file compression to gzip, bzip2 or none
	 */

	void FileCompression(HPFileCompression theFileCompression);
	HPFileCompression FileCompression() const;

	/**
	 * @brief Enable or disable reading of source data from Neons
	 * @param theReadDataFromDatabase true = read data from neons, false = use only data specified in command
	 * line (auxiliary files)
	 */

	void ReadDataFromDatabase(bool theReadDataFromDatabase);
	bool ReadDataFromDatabase() const;

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

	bool UseCudaForInterpolation() const;
	void UseCudaForInterpolation(bool theUseCudaForInterpolation);

	bool UseCache() const;
	void UseCache(bool theUseCache);

	void SourceGeomNames(std::vector<std::string> theNames);
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

	int ForecastStep() const;

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

   protected:
	std::unique_ptr<producer_iter> itsSourceProducerIterator;

	HPFileType itsOutputFileType;
	HPFileWriteOption itsFileWriteOption;
	HPFileCompression itsFileCompression;
	HPDatabaseType itsDatabaseType;

	std::string itsConfigurationFile;
	std::vector<std::string> itsAuxiliaryFiles;
	std::string itsOriginTime;

	bool itsReadDataFromDatabase;
	short itsThreadCount;
	std::string itsTargetGeomName;
	std::vector<std::string> itsSourceGeomNames;
	std::string itsStatisticsLabel;

	producer itsTargetProducer;

	bool itsUseCuda;
	bool itsUseCudaForPacking;
	bool itsUseCudaForUnpacking;
	bool itsUseCudaForInterpolation;
	bool itsUseCache;
	bool itsUseDynamicMemoryAllocation;
	bool itsReadAllAuxiliaryFilesToCache;

	int itsCudaDeviceCount;
	int itsCudaDeviceId;

	int itsForecastStep;

	int itsCacheLimit;
	std::string itsParamFile;
	bool itsAsyncExecution;
};

inline std::ostream& operator<<(std::ostream& file, const configuration& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* CONFIGURATION_H */
