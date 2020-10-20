/**
 * @file producer.cpp
 *
 * @date Dec 28, 2012
 * @author partio
 */

#include "producer.h"

using namespace himan;

producer::producer()
    : itsFmiProducerId(kHPMissingInt),
      itsProcess(kHPMissingInt),
      itsCentre(kHPMissingInt),
      itsTableVersion(kHPMissingInt),
      itsClass(kGridClass),
      itsNeonsName("himanDefaultProducer")
{
}

producer::producer(long theFmiProducerId)
    : itsFmiProducerId(theFmiProducerId),
      itsProcess(kHPMissingInt),
      itsCentre(kHPMissingInt),
      itsTableVersion(kHPMissingInt),
      itsClass(kGridClass),
      itsNeonsName("himanDefaultProducer")
{
}

producer::producer(long theCentre, long theProcess)
    : itsFmiProducerId(kHPMissingInt),
      itsProcess(theProcess),
      itsCentre(theCentre),
      itsTableVersion(kHPMissingInt),
      itsClass(kGridClass),
      itsNeonsName("himanDefaultProducer")
{
}

producer::producer(long theFmiProducerId, long theCentre, long theProcess, const std::string& theRadonName)
    : itsFmiProducerId(theFmiProducerId),
      itsProcess(theProcess),
      itsCentre(theCentre),
      itsTableVersion(kHPMissingInt),
      itsClass(kGridClass),
      itsNeonsName(theRadonName)
{
}

producer::producer(long theFmiProducerId, const std::string& theRadonName)
    : itsFmiProducerId(theFmiProducerId),
      itsProcess(kHPMissingInt),
      itsCentre(kHPMissingInt),
      itsTableVersion(kHPMissingInt),
      itsClass(kGridClass),
      itsNeonsName(theRadonName)
{
}

void producer::Centre(long theCentre)
{
	itsCentre = theCentre;
}
long producer::Centre() const
{
	return itsCentre;
}
void producer::Process(long theProcess)
{
	itsProcess = theProcess;
}
long producer::Process() const
{
	return itsProcess;
}
void producer::Id(long theId)
{
	itsFmiProducerId = theId;
}
long producer::Id() const
{
	return itsFmiProducerId;
}
void producer::Name(const std::string& theName)
{
	itsNeonsName = theName;
}
std::string producer::Name() const
{
	return itsNeonsName;
}
long producer::TableVersion() const
{
	return itsTableVersion;
}
void producer::TableVersion(long theTableVersion)
{
	itsTableVersion = theTableVersion;
}
HPProducerClass producer::Class() const
{
	return itsClass;
}
void producer::Class(HPProducerClass theClass)
{
	itsClass = theClass;
}
std::ostream& producer::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsFmiProducerId__ " << itsFmiProducerId << std::endl;
	file << "__itsProcess__ " << itsProcess << std::endl;
	file << "__itsCentre__ " << itsCentre << std::endl;
	file << "__itsNeonsName__ " << itsNeonsName << std::endl;
	file << "__itsTableVersion__ " << itsTableVersion << std::endl;
	file << "__itsClass__" << HPProducerClassToString.at(itsClass) << std::endl;

	return file;
}

bool producer::operator==(const producer& other) const
{
	if (itsFmiProducerId != other.itsFmiProducerId)
	{
		return false;
	}

	if (itsProcess != other.itsProcess)
	{
		return false;
	}

	if (itsCentre != other.itsCentre)
	{
		return false;
	}

	if (itsNeonsName != other.itsNeonsName)
	{
		return false;
	}

	if (itsTableVersion != other.itsTableVersion)
	{
		return false;
	}

	if (itsClass != other.itsClass)
	{
		return false;
	}

	return true;
}

bool producer::operator!=(const producer& other) const
{
	return !(*this == other);
}
