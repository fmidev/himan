#pragma once
#include "info.h"

namespace himan
{

#ifdef HAVE_CEREAL
class spiller
{
   public:
	template <typename T>
	static std::shared_ptr<himan::info<T>> ReadFromUniqueName(const std::string& uniqueName);
	template <typename T>
	static std::shared_ptr<himan::info<T>> ReadFromFileName(const std::string& fileName);
	template <typename T>
	static std::string Write(std::shared_ptr<himan::info<T>>& info);

	static void RemoveFromFileName(const std::string& fileName);
	static void RemoveFromUniqueName(const std::string& uniqueName);
	static void RemoveAll();

	static bool Enabled();
};
#endif

};  // namespace himan
