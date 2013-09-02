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

	/**
	 * @return Class name
	 */

	std::string ClassName() const
	{
		return "himan::configuration";
	}

	HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
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

	void SourceProducers(std::vector<producer> theSourceProducers);
	producer SourceProducer() const;

	producer TargetProducer() const;

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

	bool UseCuda() const;
	void UseCuda(bool theUseCuda);

	std::string TargetProjection() const;

	HPDimensionType LeadingDimension() const;

	void ThreadCount(short theThreadCount);
	short ThreadCount() const;

	std::string ConfigurationFile() const;
	void ConfigurationFile(const std::string& theConfigurationFile);

	void StatisticsLabel(const std::string& theStatisticsLabel);
	std::string StatisticsLabel() const;

	bool UseCudaForPacking() const;
	void UseCudaForPacking(bool theUseCudaForPacking);

	bool UseCache() const;
	void UseCache(bool theUseCache);

	std::string SourceGeomName() const;

	bool SourceProducer(const producer& theSourceProducer);
	bool NextSourceProducer() const; // THIS SHOULD NOT BE CONST ??
	bool FirstSourceProducer() const; // THIS SHOULD NOT BE CONST ??
	void ResetSourceProducer() const;

	void StoreCudaDeviceCount();
	bool HaveCuda() const;
	short CudaDeviceCount() const;

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
};


inline
std::ostream& operator<<(std::ostream& file, const configuration& ob)
{
	return ob.Write(file);
}

} // namespace himan

#endif /* CONFIGURATION_H */
