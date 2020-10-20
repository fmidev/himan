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
#include "serialization.h"

namespace himan
{
enum HPProducerClass
{
	kUnknownProducerClass = 0,
	kGridClass = 1,
	kPreviClass = 3
};

const boost::unordered_map<HPProducerClass, std::string> HPProducerClassToString =
    ba::map_list_of(kUnknownProducerClass, "unknown")(kGridClass, "grid")(kPreviClass, "previ");

const boost::unordered_map<std::string, HPProducerClass> HPStringToProducerClass =
    ba::map_list_of("unknown", kUnknownProducerClass)("grid", kGridClass)("previ", kPreviClass);

class producer
{
   public:
	producer();
	explicit producer(long theFmiProducerId);
	producer(long theCentre, long theProcess);
	producer(long theFmiProducerId, long theCentre, long theProcess, const std::string& theRadonName);
	producer(long theFmiProducerId, const std::string& theRadonName);

	~producer() = default;

	std::string ClassName() const
	{
		return "himan::producer";
	}
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

	HPProducerClass Class() const;
	void Class(HPProducerClass theClass);

	bool operator==(const producer& other) const;
	bool operator!=(const producer& other) const;

   private:
	long itsFmiProducerId;
	long itsProcess;
	long itsCentre;
	long itsTableVersion;
	HPProducerClass itsClass;

	std::string itsNeonsName;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsFmiProducerId), CEREAL_NVP(itsProcess), CEREAL_NVP(itsCentre), CEREAL_NVP(itsTableVersion),
		   CEREAL_NVP(itsNeonsName), CEREAL_NVP(itsClass));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const producer& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#endif /* PRODUCER_H */
