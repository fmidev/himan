/*
 * producer.h
 *
 *  Created on: Dec 28, 2012
 *      Author: partio
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

    ~producer() {}

    std::string ClassName() const
    {
        return "himan::producer";
    }

    HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
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

private:

    long itsFmiProducerId;
    long itsProcess;
    long itsCentre;
    std::string itsNeonsName;

};

inline
std::ostream& operator<<(std::ostream& file, const producer& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* PRODUCER_H */
