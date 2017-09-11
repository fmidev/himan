/**
 * @file plugin_configuration.h
 *
 */

#ifndef PLUGIN_CONFIGURATION_H
#define PLUGIN_CONFIGURATION_H

#include "configuration.h"
#include "himan_common.h"
#include "statistics.h"
#include <map>
#include <utility>

namespace himan
{
class plugin_configuration : public configuration
{
   public:
	plugin_configuration();
	plugin_configuration(const plugin_configuration& other);
	plugin_configuration& operator=(const plugin_configuration& other) = delete;

	explicit plugin_configuration(const configuration& theConfiguration);
	plugin_configuration(const std::string& theName, const std::map<std::string, std::vector<std::string>>& theOptions);

	~plugin_configuration() = default;

	/**
	 * @return Class name
	 */

	std::string ClassName() const { return "himan::plugin_configuration"; }
	std::ostream& Write(std::ostream& file) const;

	/**
	 * @brief Set plugin name
	 * @param theName New name
	 */

	void Name(const std::string& theName);

	/**
	 *
	 * @return Plugin name
	 */
	std::string Name() const;

	/**
	 * @brief Add new element to options map
	 * @param key Key name
	 * @param value Value
	 */

	void AddOption(const std::string& key, const std::string& value);

	/**
	 * @brief Check if a key exists in options map
	 */

	bool Exists(const std::string& key) const;

	/**
	 * @brief Get value for a given key
	 * @param key Key name
	 * @return Value as string, empty string if key doesn't exist
	 */

	std::string GetValue(const std::string& key) const;
	const std::vector<std::string>& GetValueList(const std::string& key) const;

	/**
	  * @brief Add a new parameter to the preconfigured parameters map
	  * @param paramName Parameter name
	  * @param opts      Parameter specific options
	  */

	void AddParameter(const std::string& paramName, const std::vector<std::pair<std::string, std::string>>& opts);

	/**
	* @brief Check if the preconfigured parameter exists
	* @param paramName Parameter name
	*/

	bool ParameterExists(const std::string& paramName) const;

	/**
	 * @brief Get parameter names
	 * @return Parameter names as a vector<string>
	 */
	std::vector<std::string> GetParameterNames() const;

	/**
	* @brief Get parameter options
	* @param paramName Parameter name
	* @return Parameter options as a vector of strings, empty vector if the parameter doesn't exist
	* @throws runtime_error if the parameter was not found, use ParamExists(paramName) to check if the parameter exists
	*/

	const std::vector<std::pair<std::string, std::string>>& GetParameterOptions(const std::string& paramName) const;

	void Info(std::shared_ptr<info> theInfo);
	std::shared_ptr<info> Info() const;

	std::shared_ptr<statistics> Statistics() const;

	bool StatisticsEnabled() const;
	void StartStatistics();
	void WriteStatistics();

   private:
	std::string itsName;
	std::map<std::string, std::vector<std::string>> itsOptions;
	std::map<std::string, std::vector<std::pair<std::string, std::string>>> itsPreconfiguredParams;
	std::shared_ptr<info> itsInfo;
	std::shared_ptr<statistics> itsStatistics;
};

inline std::ostream& operator<<(std::ostream& file, const plugin_configuration& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* PLUGIN_CONFIGURATION_H */
