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

    friend class ini_parser;

    configuration();
    ~configuration() {}

    configuration(const configuration& other) = delete;
    configuration& operator=(const configuration& other) = delete;

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
     * @return Info instance created from configuration file metadata
     */

    inline std::shared_ptr<info> Info() const
    {
        return itsInfo;
    }

    /**
     * @return List of plugin names found in the configuration file
     */

    std::vector<std::string> Plugins() const;

    /**
     * @param List of plugin names
     */

    void Plugins(const std::vector<std::string>& thePlugins) ;

    /**
     * @return List of auxiliary file names found in the configuration file
     */

    std::vector<std::string> AuxiliaryFiles() const;
    void AuxiliaryFiles(const std::vector<std::string> theAuxiliaryFiles);

    /**
     * @return Filetype of created file. One of: grib1, grib2, querydata, netcdf
     */

    HPFileType OutputFileType() const;

    unsigned int SourceProducer() const;

    unsigned int TargetProducer() const;

    void SourceProducer(unsigned int theSourceProducer);
    void TargetProducer(unsigned int theTargetProducer);

    /**
     * @brief Enable or disable writing to one or many file
     * @param Value, true = all data is written to one file, false = each data descriptor
     * combination is written to separate file
     */

    void WholeFileWrite(bool theWholeFileWrite);
    bool WholeFileWrite() const;

    size_t Ni() const;
    size_t Nj() const;

    void Ni(size_t theNi);
    void Nj(size_t theNj);

    /**
     * @brief Enable or disable reading of source data from Neons
     * @param Value, true = read data from neons, false = use only data specified in command
     * line (auxiliary files)
     */

    void ReadDataFromDatabase(bool theReadDataFromDatabase);
    bool ReadDataFromDatabase() const;

    /**
     * @brief Enable or disable waiting for files
     * @param Value in minutes
     */

    void FileWaitTimeout(unsigned short theFileWaitTimeout);
    unsigned short FileWaitTimeout() const;

    bool UseCuda() const;

    HPDimensionType LeadingDimension() const;

private:

    void Init();

    std::shared_ptr<info> itsInfo; //!< @see class 'info'

    HPFileType itsOutputFileType;
    std::string itsConfigurationFile;
    std::vector<std::string> itsAuxiliaryFiles;

    std::vector<std::string> itsPlugins;
    std::string itsOriginTime;

    std::shared_ptr<logger> itsLogger;

    unsigned int itsSourceProducer;
    unsigned int itsTargetProducer;

    bool itsWholeFileWrite;
    bool itsReadDataFromDatabase;

    size_t itsNi;
    size_t itsNj;

    unsigned short itsFileWaitTimeout; //<! Minutes
    bool itsUseCuda;

    HPDimensionType itsLeadingDimension;
};

inline
std::ostream& operator<<(std::ostream& file, configuration& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* CONFIGURATION_H */
