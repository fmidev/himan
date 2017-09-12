#include "debug.h"
#include "himan_common.h"

#include <execinfo.h>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>

namespace himan
{
static void PrintBacktrace();

bool AssertionFailed(const char* expr, long line, const char* fn, const char* file)
{
	printf("Assertion (%s) failed at: %s::%s:%ld\n", expr, file, fn, line);
	PrintBacktrace();
	// XXX Check if we're running inside debugger, return the result here.
	return true;
}

//
// https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
// https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.
//
static const int kMaxReturnAddresses = 128;

void PrintBacktrace()
{
	void* retAddreses[kMaxReturnAddresses];

	const int size = backtrace(retAddreses, kMaxReturnAddresses);
	if (size < 3)
	{
		// No useful information for us, return.
		return;
	}

	char** symbols = backtrace_symbols(retAddreses, size);

	if (symbols)
	{
		for (int i = 2; i < size; i++)
		{
			// XXX Of course this doesn't work with pathnames including '('!
			const std::string symbol = std::string(symbols[i]);
			auto start = symbol.find_last_of('(');
			// Stop at the start of the offset, like in:
			// `_Z13ExecutePluginSt10shared_ptrIN5himan20plugin_configurationEE+0x2a0`
			auto end = symbol.find_last_of('+');
			// End of the offset.
			auto closingParen = symbol.find_last_of(')');
			// Address of the function can be useful (e.g. if no symbol is found).
			auto addrStart = symbol.find_last_of('[');

			if (start == std::string::npos || end == std::string::npos || closingParen == std::string::npos ||
				addrStart == std::string::npos)
			{
				continue;
			}

			const std::string cppSymbol = symbol.substr(start + 1, end - start - 1);
			const std::string offset = symbol.substr(end, closingParen - end);
			const std::string addr = symbol.substr(addrStart);

			int status;
			char* demangledName = abi::__cxa_demangle(cppSymbol.c_str(), 0, 0, &status);
			// failed to demangle
			if (status < 0)
			{
				printf("%d: %s (%s) %s\n", i - 2, cppSymbol.c_str(), offset.c_str(), addr.c_str());
			}
			else
			{
				printf("%d: %s (%s) %s\n", i - 2, demangledName, offset.c_str(), addr.c_str());
				free(demangledName);
			}
		}
		free(symbols);
	}
}

void Abort()
{
	PrintBacktrace();
	abort();
}
} // namespace himan
