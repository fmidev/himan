#
# SConscript for himan-lib

import os
import platform

OS_NAME = platform.linux_distribution()[0]
OS_VERSION = float('.'.join(platform.linux_distribution()[1].split('.')[:2]))

IS_RHEL = False
IS_SLES = False

if OS_NAME == "Red Hat Enterprise Linux Server" or OS_NAME == "CentOS Linux":
	IS_RHEL=True
elif OS_NAME == "SUSE Linux Enterprise Server ":
	IS_SLES=True

# Should also get compiler version here but it seems to be rather
# complicated with python subprocess -module

env = Environment(ENV = {'PATH' : os.environ['PATH']})

# Get color output from gcc / clang
try:
    env['ENV']['TERM'] = os.environ['TERM']
except KeyError:
    pass

env['CC'] = 'gcc'
env['CXX'] = 'g++'

if os.environ.get('CC') != None:
        env['CC'] = os.environ.get('CC')

if os.environ.get('CXX') != None:
        env['CXX'] = os.environ.get('CXX')

IS_GCC = False
IS_CLANG = False

if env['CXX'] == 'g++':
	IS_GCC=True
elif env['CXX'] == 'clang++':
	IS_CLANG=True

AddOption(
    '--debug-build',
    dest='debug-build',
    action='store_true',
    help='debug build',
    default=False)

AddOption(
    '--no-cuda-build',
    dest='no-cuda-build',
    action='store_true',
    help='no cuda build',
    default=False)

# Check build

NOCUDA = GetOption('no-cuda-build')
DEBUG = GetOption('debug-build')
RELEASE = (not DEBUG)

env['DEBUG'] = DEBUG
env['RELEASE'] = RELEASE

# Workspace

env['WORKSPACE'] = "%s/../" % os.getcwd()
# rpm builds:
#env['WORKSPACE'] = "/home/dev/partio/himan"

# cuda toolkit path

if not NOCUDA:
	cuda_toolkit_path = '/usr/local/cuda'

	if os.environ.get('CUDA_TOOLKIT_PATH') is not None:
        	cuda_toolkit_path = os.environ['CUDA_TOOLKIT_PATH']

env['HAVE_CUDA'] = False

if not NOCUDA and os.path.isfile(cuda_toolkit_path + '/lib64/libcudart.so'):
        env['HAVE_CUDA'] = True

env['HAVE_S3'] = False

if os.path.isfile('/usr/include/libs3.h'):
	env['HAVE_S3'] = True

# Required for scan-build
env["ENV"].update(x for x in os.environ.items() if x[0].startswith("CCC_"))

# Includes

includes = []

try:
	includes.extend(os.environ['INCLUDEPATHS'].split(';'))
except:
	pass

includes.append(env['WORKSPACE'] + '/himan-lib/include')
includes.append(env['WORKSPACE'] + '/himan-plugins/include')

if env['HAVE_CUDA']:
        includes.append(cuda_toolkit_path + '/include')

env.Append(CPPPATH = includes)

# Library paths

librarypaths = []

try:
	librarypaths.extend(os.environ['LIBRARYPATHS'].split(';'))
except:
	pass

librarypaths.append('/usr/lib64')
env.Append(LIBPATH = librarypaths)

# Libraries

env.Append(LIBS = ['z', 'bz2'])

if env['HAVE_CUDA']:
        env.Append(LIBS=env.File(cuda_toolkit_path + '/lib64/libcudart_static.a'))

# CFLAGS

# "Normal" flags

cflags_normal = []
cflags_normal.append('-Wall')
cflags_normal.append('-W')
cflags_normal.append('-Wno-unused-parameter')
cflags_normal.append('-Werror')
cflags_normal.append('-Wno-narrowing')
cflags_normal.append('-Wpointer-arith')
cflags_normal.append('-Wcast-qual')

# Extra flags

cflags_extra = []
cflags_extra.append('-Wcast-align')
cflags_extra.append('-Wwrite-strings')
cflags_extra.append('-Wconversion')
# cflags_extra.append('-Winline')
cflags_extra.append('-Wnon-virtual-dtor')
cflags_extra.append('-Wsign-promo')
cflags_extra.append('-Wchar-subscripts')
cflags_extra.append('-Wold-style-cast')
cflags_extra.append('-Wunreachable-code')
cflags_extra.append('-Wshadow')

if IS_GCC:
       cflags_extra.append('-Wno-pmf-conversions')
elif IS_CLANG:
       cflags_extra.append('-Wno-int-conversions')

# Difficult flags

cflags_difficult = []
cflags_difficult.append('-pedantic')
cflags_difficult.append('-Weffc++')
cflags_difficult.append('-Wredundant-decls')
cflags_difficult.append('-Woverloaded-virtual')
cflags_difficult.append('-Wctor-dtor-privacy')

# Default flags (common for release/debug)

if IS_GCC:
	env.Append(CCFLAGS = '-std=c++11')
else:
	env.Append(CCFLAGS = '-std=c++11')

env.Append(CCFLAGS = '-fPIC')
env.Append(CCFLAGS = cflags_normal)
env.Append(CCFLAGS = cflags_extra)

if IS_CLANG:
	env.AppendUnique(CCFLAGS=('-isystem', '/usr/include/smartmet/newbase'))
	env.AppendUnique(CCFLAGS=('-isystem', '/usr/include/Eigen'))
	env.AppendUnique(CCFLAGS=('-isystem', '/opt/llvm-5.0.0/include'))
	env.AppendUnique(CCFLAGS=('-isystem', '/opt/llvm-5.0.0/include/c++/v1'))

# Linker flags

env.Append(LINKFLAGS = ['-rdynamic','-Wl,--as-needed'])

# Defines

env.Append(CPPDEFINES=['UNIX'])

if env['HAVE_CUDA']:
        env.Append(CPPDEFINES=['HAVE_CUDA'])
if env['HAVE_S3']:
        env.Append(CPPDEFINES=['HAVE_S3'])

env.Append(NVCCDEFINES=['HAVE_CUDA'])

env.Append(NVCCFLAGS = ['-m64', '-Xcompiler', '-fPIC'])
env.Append(NVCCFLAGS = ['-Wno-deprecated-declarations'])
env.Append(NVCCFLAGS = ['-gencode=arch=compute_35,code=sm_35'])
env.Append(NVCCFLAGS = ['-gencode=arch=compute_52,code=sm_52'])
env.Append(NVCCFLAGS = ['-gencode=arch=compute_60,code=sm_60'])

#if IS_CLANG:
#	env.Append(NVCCFLAGS = ['-ccbin=clang++'])
#	env.Append(NVCCFLAGS = ['-std=c++14'])

#else:
env.Append(NVCCFLAGS = ['-std=c++11'])

for flag in cflags_normal:
	if flag == '-Wcast-qual':
		continue
	env.Append(NVCCFLAGS = ['-Xcompiler', flag])

env.Append(NVCCPATH = [env['WORKSPACE'] + '/himan-lib/include']) # cuda-helper
env.Append(NVCCPATH = [env['WORKSPACE'] + '/himan-plugins/include'])
env.Append(NVCCPATH = ['/usr/include/smartmet/newbase'])

# Other

build_dir = ""

env.Append(NOCUDA = NOCUDA)

if RELEASE:
	env.Append(CPPDEFINES = ['NDEBUG'])
	env.Append(CCFLAGS = ['-O2'])

	env.Append(NVCCFLAGS = ['-O2'])
	env.Append(NVCCDEFINES = ['NDEBUG'])

	build_dir = 'build/release'

if DEBUG:
	env.Append(CCFLAGS = ['-O0'])
	env.Append(CCFLAGS = ['-ggdb', '-g3'])	
	#env.Append(CCFLAGS = cflags_difficult)
	env.Append(CPPDEFINES = ['DEBUG'])

	# Cuda
	env.Append(NVCCFLAGS = ['-O0','-g','-G'])
	env.Append(NVCCDEFINES = ['DEBUG'])

	build_dir = 'build/debug'

#
# https://bitbucket.org/scons/scons/wiki/PhonyTargets
#
def PhonyTargets(env = None, **kw):
        if not env: env = DefaultEnvironment()
        for target,action in kw.items():
                env.AlwaysBuild(env.Alias(target, [], action))

PhonyTargets(CPPCHECK = 'cppcheck --std=c++11 --enable=all -I ./include -I ../himan-lib/include ./')
PhonyTargets(SCANBUILD = 'scan-build make debug')

Export('env build_dir')
