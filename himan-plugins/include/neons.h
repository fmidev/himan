/**
 * @file neons.h
 *
 * @class neons
 *
 * @brief Access to neons database.
 *
 * A note on compiling and linking:
 *
 * This class is only an overcoat to fmidb. fmidb-library is just a bunch of objects
 * and it contains no linking, so we have to link here. Now, the problem is that
 * oracle instant client provides only shared library version of libclntsh. This
 * means that we have to link every library/executable that used this class against
 * libclntsh. And libclntsh want libaio etc.
 *
 *  One option to solve this would be to link this class statically with libclntsh,
 *  but to do so we'd need the full version of oracle client. Also, the .a version
 *  of oracle client library is HUGE, nearly 100M.
 */

#ifndef NEONS_H
#define NEONS_H

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6)) && defined override
#undef override
#endif

#include "NFmiNeonsDB.h"
#include "auxiliary_plugin.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{
class neons : public auxiliary_plugin
{
   public:
	neons();

	inline virtual ~neons()
	{
		if (itsNeonsDB)
		{
			NFmiNeonsDBPool::Instance()->Release(&(*itsNeonsDB));  // Return connection back to pool
			itsNeonsDB.release();
		}
	}

	neons(const neons& other) = delete;
	neons& operator=(const neons& other) = delete;

	virtual std::string ClassName() const { return "himan::plugin::neons"; }
	virtual HPPluginClass PluginClass() const { return kAuxiliary; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
	std::vector<std::string> Files(search_options& options);
	bool Save(const info& resultInfo, const std::string& theFileName);

	/// Gets grib parameter name based on number and code table
	/**
	 *  \par fmiParameterId - parameter number
	 *  \par codeTableVersion  - code table number
	 *  \par timeRangeIndicator - time range indicator (grib 1)
	 */

	std::string GribParameterName(long fmiParameterId, long codeTableVersion, long timeRangeIndicator, long levelType);
	std::string GribParameterName(long fmiParameterId, long category, long discipline, long producer, long levelType);

	/**
	 * @brief Function to expose the NFmiNeonsDB interface
	 *
	 * @return Reference to the NFmiNeonsDB instance
	 */

	NFmiNeonsDB& NeonsDB();

	/**
	 * @brief Fetch producer metadata from neons (eventually)
	 *
	 * @param producerId Producer id
	 * @param attribute
	 * @return String containing the result, empty string if attribute is not found
	 */

	std::string ProducerMetaData(long producerId, const std::string& attribute) const;

	void PoolMaxWorkers(int maxWorkers);

   private:
	/**
	 * @brief Connect to database
	 *
	 * We cannot connect to database directly in the constructor, but we need
	 * to use another function for that.
	 */

	void Init();

	bool itsInit;                             //!< Holds the initialization status of the database connection
	std::unique_ptr<NFmiNeonsDB> itsNeonsDB;  //<! The actual database class instance
};

inline NFmiNeonsDB& neons::NeonsDB()
{
	Init();
	return *itsNeonsDB.get();
}

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<neons>(new neons()); }
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* NEONS_H */
