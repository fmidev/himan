#pragma once

#include <string>

namespace himan
{
namespace plugin
{

struct radon_record
{
        std::string schema_name;
        std::string table_name;
        std::string partition_name;
        std::string geometry_name;
        int geometry_id;

        radon_record() = default;
        radon_record(const std::string& schema_name_, const std::string& table_name_, const std::string& partition_name_,
                     const std::string& geometry_name_, int geometry_id_)
            : schema_name(schema_name_),
              table_name(table_name_),
              partition_name(partition_name_),
              geometry_name(geometry_name_),
              geometry_id(geometry_id_)
        {
        }
};

} // namespace plugin
} // namespace himan
