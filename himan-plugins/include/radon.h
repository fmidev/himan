/**
 * @file radon.h
 *
 * @date Oct 28, 2014
 *
 * @class radon
 *
 * @brief Access to radon database.
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

#ifndef RADON_H
#define RADON_H

#include "NFmiRadonDB.h"
#include "auxiliary_plugin.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{
class radon : public auxiliary_plugin
{
   public:
	radon();

	inline virtual ~radon()
	{
		if (itsRadonDB)
		{
			auto raw = itsRadonDB.release();            // Give up ownership
			NFmiRadonDBPool::Instance()->Release(raw);  // Return connection back to pool
		}
	}

	radon(const radon& other) = delete;
	radon& operator=(const radon& other) = delete;

	virtual std::string ClassName() const { return "himan::plugin::radon"; }
	virtual HPPluginClass PluginClass() const { return kAuxiliary; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
	/**
	 * @brief Return filename of a field
	 */

	std::vector<std::string> Files(search_options& options);

	/**
	 * @brief Return previ data in CSV format
	 */

	std::vector<std::string> CSV(search_options& options);

	/**
	 * @brief Save either file metadata or previ information to database.
	 */

	bool Save(const info& resultInfo, const std::string& theFileName);

	/**
	 * @brief Function to expose the NFmiRadonDB interface
	 *
	 * @return Reference to the NFmiRadonDB instance
	 */

	NFmiRadonDB& RadonDB();

	void PoolMaxWorkers(int maxWorkers);

   private:
	/**
	 * @brief Connect to database
	 *
	 * We cannot connect to database directly in the constructor, but we need
	 * to use another function for that.
	 */

	void Init();
	bool SaveGrid(const info& resultInfo, const std::string& theFileName);
	bool SavePrevi(const info& resultInfo);

	bool itsInit;                             //!< Holds the initialization status of the database connection
	std::unique_ptr<NFmiRadonDB> itsRadonDB;  //<! The actual database class instance
};

inline NFmiRadonDB& radon::RadonDB()
{
	Init();
	return *itsRadonDB.get();
}

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<radon>(new radon()); }
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* RADON_H */
