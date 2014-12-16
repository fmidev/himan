/**
 * @file grid.cpp
 *
 * @date Jan 23, 2013
 * @author partio
 */

#include "grid.h"


using namespace himan;
using namespace std;

grid::grid() : itsGridType(kUnknownGridType) {}

grid::~grid() {}

bool grid::EqualsTo(const grid& other) const
{
	if (other.Type() == itsGridType)
	{
		return true;
	}
	
	return false;
}