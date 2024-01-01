#
# SConscript for himan-lib

import os
import distro

# Should also get compiler version here but it seems to be rather
# complicated with python subprocess -module

env = Environment(ENV = {'PATH' : os.environ['PATH']})

env['OS_NAME'] = distro.name()
env['OS_VERSION'] = float(distro.version())
env['IS_RHEL'] = (env['OS_NAME'] == "Red Hat Enterprise Linux" or env['OS_NAME'] == "CentOS Linux" or env['OS_NAME'] == "Rocky Linux")
env['IS_SLES'] = (env['OS_NAME'] == "SUSE Linux Enterprise Server")

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

env['HAVE_CEREAL'] = False

if os.path.isfile('/usr/include/cereal/cereal.hpp'):
	env['HAVE_CEREAL'] = True

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

if env['OS_VERSION'] < 9:
    librarypaths.append('/usr/lib64/boost169')

librarypaths.append('/usr/gdal35/lib')
env.Append(LIBPATH = librarypaths)

# Libraries

env.Append(LIBS = ['z', 'bz2', 'stdc++fs'])

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

cpp_standard = 'c++11'

if env['OS_VERSION'] >= 8:
    cpp_standard = 'c++17'

env.Append(CCFLAGS = '-std=' + cpp_standard)
env.Append(CCFLAGS = '-fPIC')
env.Append(CCFLAGS = cflags_normal)
env.Append(CCFLAGS = cflags_extra)

if env['OS_VERSION'] < 9:
    env.AppendUnique(CCFLAGS=('-isystem', '/usr/include/boost169'))

env.AppendUnique(CCFLAGS=('-isystem', '/usr/gdal35/include'))
env.AppendUnique(CCFLAGS=('-isystem', '/usr/include/eigen3'))

if IS_CLANG:
	env.AppendUnique(CCFLAGS=('-isystem', '/usr/include/smartmet/newbase'))
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
if env['HAVE_CEREAL']:
        env.Append(CPPDEFINES=['HAVE_CEREAL'])

env.Append(NVCCDEFINES=['HAVE_CUDA'])

env.Append(NVCCFLAGS = ['-m64', '-Xcompiler', '-fPIC'])
env.Append(NVCCFLAGS = ['-Wno-deprecated-declarations'])
env.Append(NVCCFLAGS = ['-gencode=arch=compute_60,code=sm_60'])
env.Append(NVCCFLAGS = ['-gencode=arch=compute_70,code=sm_70'])
env.Append(NVCCFLAGS = ['-gencode=arch=compute_80,code=sm_80'])

env.AppendUnique(NVCCFLAGS = ['-std=' + cpp_standard])

if env['OS_VERSION'] < 9:
    env.AppendUnique(NVCCFLAGS = ('-isystem', '/usr/include/boost169'))

env.AppendUnique(NVCCFLAGS = ('-isystem', '/usr/gdal35/include'))

for flag in cflags_normal:
	if flag == '-Wcast-qual':
		continue
	env.Append(NVCCFLAGS = ['-Xcompiler', flag])

# thrust and cuda combined producer warnings like:
# warning #20012-D: __device__ annotation is ignored on a function("vector") that is explicitly defaulted on its first declaration
# disable these warnings
env.Append(NVCCFLAGS = ['--diag-suppress', 20012])
env.Append(NVCCPATH = [env['WORKSPACE'] + '/himan-lib/include']) # cuda-helper
env.Append(NVCCPATH = [env['WORKSPACE'] + '/himan-plugins/include'])
env.Append(NVCCPATH = ['/usr/include/smartmet/newbase'])

# Other

build_dir = ""

env.Append(NOCUDA = NOCUDA)

if RELEASE:
	env.Append(CPPDEFINES = ['NDEBUG'])
	env.Append(CCFLAGS = ['-O2', '-g'])

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

PhonyTargets(CPPCHECK = 'cppcheck --std=c++17 --enable=all -I ./include -I ../himan-lib/include ./')
PhonyTargets(SCANBUILD = 'scan-build make debug')

Export('env build_dir')
