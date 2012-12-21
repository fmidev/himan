/*
 * configuration.h
 *
 *  Created on: Nov 26, 2012
 *      Author: partio
 *
 *
 *
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "logger.h"
#include "info.h"

namespace hilpee
{

class configuration
{

	public:

		friend class ini_parser;

		configuration();

		~configuration() {}

		std::string ClassName() const
		{
			return "hilpee::configuration";
		}

		HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		std::ostream& Write(std::ostream& file) const;

		std::shared_ptr<info> Info() const
		{
			return itsInfo;
		}

		std::vector<std::string> Plugins() const;
		void Plugins(const std::vector<std::string>& thePlugins) ;

		std::vector<std::string> InputFiles() const;
		std::vector<std::string> AuxiliaryFiles() const;

		HPFileType OutputFileType() const;

		unsigned int SourceProducer() const;
		unsigned int TargetProducer() const;

		void SourceProducer(unsigned int theSourceProducer);
		void TargetProducer(unsigned int theTargetProducer);

		void WholeFileWrite(bool theWholeFileWrite);
		bool WholeFileWrite() const;

		size_t Ni() const;
		size_t Nj() const;

		void Ni(size_t theNi);
		void Nj(size_t theNj);

		bool ReadDataFromDatabase() const;
		void ReadDataFromDatabase(bool theReadDataFromDatabase);

	private:

		std::shared_ptr<info> itsInfo;

		void Init();

		HPFileType itsOutputFileType;
		std::string itsConfigurationFile;
		std::vector<std::string> itsInputFiles;
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
};

inline
std::ostream& operator<<(std::ostream& file, configuration& ob)
{
	return ob.Write(file);
}

} // namespace hilpee

#endif /* HILPEE_CONFIGURATION_H */
