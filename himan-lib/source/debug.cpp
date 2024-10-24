#include "debug.h"
#include "himan_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/ptrace.h>
#include <unistd.h>
namespace himan
{
static bool DebuggerAttached()
{
	// (https://stackoverflow.com/a/24969863)
	// - look at /proc/{PID or self}/status to get TracerPid. (current)
	// - fork, try to PTRACE_ATTACH (previous, hanged in some occasions)

	char buf[1024];
	int debugger_present = 0;

	int status_fd = open("/proc/self/status", O_RDONLY);
	if (status_fd == -1)
		return 0;

	ssize_t num_read = read(status_fd, buf, sizeof(buf) - 1);

	if (num_read > 0)
	{
		static const char TracerPid[] = "TracerPid:";
		char* tracer_pid;

		buf[num_read] = '\0';
		tracer_pid = strstr(buf, TracerPid);
		if (tracer_pid)
			debugger_present = !!atoi(tracer_pid + sizeof(TracerPid) - 1);
	}

	return debugger_present;
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
			size_t start = std::string::npos;  // '('
			// Stop at the start of the offset, like in:
			// `_Z13ExecutePluginSt10shared_ptrIN5himan20plugin_configurationEE+0x2a0`
			size_t end = std::string::npos;  // '+'
			size_t closingParen = std::string::npos;
			size_t addrStart = std::string::npos;  // '['

			// Make only one pass through the string.
			for (size_t j = 0; j < symbol.size(); j++)
			{
				switch (symbol[j])
				{
					case '(':
						start = j;
						break;
					case '+':
						end = j;
						break;
					case ')':
						closingParen = j;
						break;
					case '[':
						addrStart = j;
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
	// flush stdout before aborting to ensure that the backtrace is printed
	fflush(stdout);
	abort();
}
}  // namespace himan
