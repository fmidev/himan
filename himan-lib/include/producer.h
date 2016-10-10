/**
 * @file producer.h
 *
 *
 * @brief Class to hold the necessary producer information: neons id, grib process
 * and grib centre (and name)
 */

#ifndef PRODUCER_H
#define PRODUCER_H

#include "himan_common.h"

namespace himan
{
class producer
{
public:
	producer();
	explicit producer(long theFmiProducerId);
	producer(long theCentre, long theProcess);
	producer(long theFmiProducerId, long theCentre, long theProcess, const std::string& theNeonsName);

	~producer() {}
	
	std::string ClassName() const { return "himan::producer"; }
	std::ostream& Write(std::ostream& file) const;

	void Centre(long theCentre);
	long Centre() const;

	void Process(long theProcess);
	long Process() const;

	void Id(long theId);
	long Id() const;

	void Name(const std::string& theName);
	std::string Name() const;

	long TableVersion() const;
	void TableVersion(long theTableVersion);

	bool operator==(const producer& other);
	bool operator!=(const producer& other);

private:
	long itsFmiProducerId;
	long itsProcess;
	long itsCentre;
	long itsTableVersion;

	std::string itsNeonsName;
};

inline std::ostream& operator<<(std::ostream& file, const producer& ob) { return ob.Write(file); }

}  // namespace himan

#endif /* PRODUCER_H */
