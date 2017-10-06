#include "debug.h"
#include "himan_common.h"

#include <execinfo.h>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <signal.h>
#include <sys/ptrace.h>
#include <dlfcn.h>

namespace himan
{

static bool DebuggerAttached()
{
	// Other possibilities:
	// - look at /proc/{PID or self}/status to get TracerPid.
	// - fork, try to PTRACE_ATTACH
	if (ptrace(PTRACE_TRACEME, 0, NULL, 0) < 0)
	{
		return true;
	}
	return false;
}

static void SignalHandler(int signum)
{
	switch (signum)
	{
	case SIGINT:
		// No stack trace since this was requested by the user.
		_Exit(1);
		break;
	case SIGQUIT:
		// 'dump core signal'
		printf("Received SIGQUIT, aborting\n");
		himan::Abort();
		break;
	case SIGSEGV:
		printf("Received SIGSEGV, aborting\n");
		himan::Abort();
		break;
	default:
		// We haven't registered any other handlers.
		break;
	}
}

void SignalHandlerInit()
{
	struct sigaction act;
	act.sa_handler = SignalHandler;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;

	int ret;
	if ((ret = sigaction(SIGINT, &act, nullptr)) == -1)
	{
		printf("Failed to initialize signal handler for SIGINT\n");
		himan::Abort();
	}
	if ((ret = sigaction(SIGQUIT, &act, nullptr)) == -1)
	{
		printf("Failed to initialize signal handler for SIGQUIT\n");
		himan::Abort();
	}
	if ((ret = sigaction(SIGSEGV, &act, nullptr)) == -1)
	{
		printf("Failed to initialize signal handler for SIGSEGV\n");
		himan::Abort();
	}
}

static void PrintBacktrace();

bool AssertionFailed(const char* expr, long line, const char* fn, const char* file)
{
	printf("Assertion (%s) failed at: %s::%s:%ld\n\n", expr, file, fn, line);
	PrintBacktrace();
	return DebuggerAttached();
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
	char** symbols = backtrace_symbols(retAddreses, size);

	if (symbols)
	{
		for (int i = 0; i < size; i++)
		{
			// XXX Of course this doesn't work with pathnames including '('!
			const std::string symbol = std::string(symbols[i]);
			// Symbol start '('
			size_t start = std::string::npos; // '('
			// Stop at the start of the offset, like in:
			// `_Z13ExecutePluginSt10shared_ptrIN5himan20plugin_configurationEE+0x2a0`
			size_t end = std::string::npos;   // '+'
			size_t closingParen = std::string::npos;
			size_t addrStart = std::string::npos; // '['

			// Make only one pass through the string.
			for (size_t i = 0; i < symbol.size(); i++)
			{
				switch(symbol[i])
				{
					case '(':
						start = i;
						break;
					case '+':
						end = i;
						break;
					case ')':
						closingParen = i;
						break;
					case '[':
						addrStart = i;
						break;
					default:
						break;
				}
			}

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
				if (cppSymbol.empty())
				{
					// Since we failed to demangle the symbol and there's no actual symbol associated with
					// this address, get the address' originating shared object.
					Dl_info info;

					if (dladdr(retAddreses[i], &info) == 0)
					{
						printf("%d: <no symbol> (%s) %s\n", i, offset.c_str(), addr.c_str());
					}
					else
					{
						printf("%d: %s: (%s) %s\n", i, info.dli_fname, offset.c_str(), addr.c_str());
					}
				}
				else
				{
					printf("%d: %s (%s) %s\n", i, cppSymbol.c_str(), offset.c_str(), addr.c_str());
				}
			}
			else
			{
				printf("%d: %s (%s) %s\n", i, demangledName, offset.c_str(), addr.c_str());
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
