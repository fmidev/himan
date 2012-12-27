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
	file << "__itsNi__ " << itsNi << std::endl;
	file << "__itsNj__ " << itsNj << std::endl;

	file << *itsInfo;

	return file;
}

std::vector<std::string> configuration::AuxiliaryFiles() const
{
	return itsAuxiliaryFiles;
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

	itsOutputFileType = kUnknownFile;
	itsSourceProducer = kHPMissingInt;
	itsTargetProducer = kHPMissingInt;
	itsWholeFileWrite = false;
	itsReadDataFromDatabase = true;
}

HPFileType configuration::OutputFileType() const
{
	return itsOutputFileType;

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

int configuration::FileWaitTimeout() const
{
	return itsFileWaitTimeout;
}

void configuration::FileWaitTimeout(int theFileWaitTimeout)
{
	itsFileWaitTimeout = theFileWaitTimeout;
}
