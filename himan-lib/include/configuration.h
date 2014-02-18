/**
 * @file configuration.h
 *
 * @brief Class to hold configuration information read from configuration file.
 *
 * Class will read metadata from configuration file and create an info instance
 * from it.
 *
 * @author Mikko Partio, FMI
 * @date Nov 26, 2012
 *
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "logger.h"
#include "info.h"

namespace himan
{

class configuration
{

public:

	friend class json_parser;

	configuration();
	~configuration() {}

	configuration(const configuration& other);
	configuration& operator=(const configuration& other) = delete;

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

	bool NextSourceProducer() const; // THIS SHOULD NOT BE CONST ??

	/**
	 * @brief Goto first source producer
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 * @return True if iterator has at least one source producer
	 */

	bool FirstSourceProducer() const; // THIS SHOULD NOT BE CONST ??

	/**
	 * @brief Reset source producer iterator
	 *
	 * This function is not thread safe as configuration class is in effect
	 * a global variable.
	 *
	 */

	void ResetSourceProducer() const;

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
	 * @brief Enable or disable reading of source data from Neons
	 * @param theReadDataFromDatabase true = read data from neons, false = use only data specified in command
	 * line (auxiliary files)
	 */

	void ReadDataFromDatabase(bool theReadDataFromDatabase);
	bool ReadDataFromDatabase() const;

	/**
	 * @brief Enable or disable waiting for files
	 * @param theFileWaitTimeout Value in minutes
	 */

	void FileWaitTimeout(unsigned short theFileWaitTimeout);
	unsigned short FileWaitTimeout() const;

	/**
	 * @brief Top level function for CUDA calculation
	 * @return True if CUDA can be used (does not tell IF it's used)
	 */
	bool UseCuda() const;
	void UseCuda(bool theUseCuda);

	HPDimensionType LeadingDimension() const;

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

	bool UseCache() const;
	void UseCache(bool theUseCache);

	std::string SourceGeomName() const;

	/**
	 * @brief Store number of CUDA devices found
     */

	void CudaDeviceCount(short theCudaDeviceCount);

	/**
	 * @brief Check if we have any CUDA-enabled devices available
     * @return True if there is at least one CUDA enable device present
     */

	bool HaveCuda() const;

	/**
	 * @return Number of CUDA enabled devices found
	 */

	short CudaDeviceCount() const;

	/**
	 * @return Id of the selected CUDA device
	 */

	short CudaDeviceId() const;
	void CudaDeviceId(short theCudaDeviceId);

protected:
	
	void Init();

	HPFileType itsOutputFileType;
	std::string itsConfigurationFile;
	std::vector<std::string> itsAuxiliaryFiles;

	std::string itsOriginTime;

	std::shared_ptr<logger> itsLogger;

	HPFileWriteOption itsFileWriteOption;
	bool itsReadDataFromDatabase;

	unsigned short itsFileWaitTimeout; //<! Minutes
	bool itsUseCuda;

	HPDimensionType itsLeadingDimension;

	short itsThreadCount;

	std::string itsTargetGeomName;
	std::string itsSourceGeomName;
	
	producer itsTargetProducer;

	std::string itsStatisticsLabel;

	bool itsUseCudaForPacking;
	bool itsUseCache;

	short itsCudaDeviceCount;

	std::unique_ptr<producer_iter> itsSourceProducerIterator;

	short itsCudaDeviceId;
};


inline
std::ostream& operator<<(std::ostream& file, const configuration& ob)
{
	return ob.Write(file);
}

} // namespace himan

#endif /* CONFIGURATION_H */
