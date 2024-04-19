/**
 * @file radon.h
 *
 * @date Oct 28, 2014
 *
 * @class radon
 *
 * @brief Access to radon database.
 *
 */

#ifndef RADON_H
#define RADON_H

#include "NFmiRadonDB.h"
#include "auxiliary_plugin.h"
#include "file_information.h"
#include "info.h"
#include "radon_record.h"
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

	virtual std::string ClassName() const override
	{
		return "himan::plugin::radon";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	}

	/**
	 * @brief Return filename of a field
	 */

	std::vector<file_information> Files(search_options& options);

	/**
	 * @brief Return previ data in CSV format
	 */

	std::vector<std::string> CSV(search_options& options);

	/**
	 * @brief Save either file metadata or previ information to database.
	 */

	template <typename T>
	std::pair<bool, radon_record> Save(const info<T>& resultInfo, const file_information& finfo,
	                                   const std::string& targetGeomName, bool dryRun = false);

	std::pair<bool, radon_record> Save(const info<double>& resultInfo, const file_information& finfo,
	                                   const std::string& targetGeomName, bool dryRun = false);

	/**
	 * @brief Function to expose the NFmiRadonDB interface
	 *
	 * @return Reference to the NFmiRadonDB instance
	 */

	NFmiRadonDB& RadonDB();

	void PoolMaxWorkers(int maxWorkers);

	std::string GetVersion() const;

   private:
	/**
	 * @brief Connect to database
	 *
	 * We cannot connect to database directly in the constructor, but we need
	 * to use another function for that.
	 */

	void Init();
	template <typename T>
	std::pair<bool, radon_record> SaveGrid(const info<T>& resultInfo, const file_information& theFileName,
	                                       const std::string& targetGeomName, bool dryRun = false);

	template <typename T>
	std::pair<bool, radon_record> SavePrevi(const info<T>& resultInfo, bool dryRun = false);

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

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<radon>();
}
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* RADON_H */
