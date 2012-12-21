LIB = tpot \
      tk2tc \
      neons \
      grib \
      cache \
      fetcher \
      writer \
      querydata \
      util \
      vvmms

MAINFLAGS = -Wall -W -Wno-unused-parameter -Werror

EXTRAFLAGS = -Wpointer-arith \
	-Wcast-qual \
	-Wcast-align \
	-Wwrite-strings \
	-Winline \
	-Wnon-virtual-dtor \
	-Wno-pmf-conversions \
	-Wsign-promo \
	-Wchar-subscripts \
	-Wold-style-cast

#	-Wconversion \

DIFFICULTFLAGS = -pedantic -Weffc++ -Wredundant-decls -Wshadow -Woverloaded-virtual -Wunreachable-code -Wctor-dtor-privacy

CC = g++
# NVCC = nvcc

# Default compiler flags

CFLAGS = -fPIC -std=c++0x -DUNIX -O2 -DNDEBUG $(MAINFLAGS)
LDFLAGS = 

# Special modes

CFLAGS_DEBUG = -fPIC -std=c++0x -DUNIX -O0 -g -DDEBUG $(MAINFLAGS) $(EXTRAFLAGS)
CFLAGS_PROFILE = -DUNIX -O2 -g -pg -DDEBUG $(MAINFLAGS)

LDFLAGS_DEBUG =
LDFLAGS_PROFILE =

# nvcc
# Assume 64 bit os

# NVCCFLAGS = -m64 -Xcompiler -fPIC
# NVCCFLAGS_DEBUG = -m64 -g -G

# CUDA_PATH       ?= /opt/cuda
# CUDA_INC_PATH   ?= $(CUDA_PATH)/include
# CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
# CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

INCLUDES = -I include \
           -I$(includedir) \
           -I../himan-lib/include \
           -I../fmidb/include \
           -I../fmigrib/include \
           -I../fminc/include \
           -I$(HOME)/local/include \
           -I/usr/include/oracle/11.2/client64 \
           -I/usr/include/smartmet/newbase

# CUDA_INCLUDES = -I include \
# 		-I $(CUDA_INC_PATH) \
# 		-I /home/partio/NVIDIA_GPU_Computing_SDK/C/common/inc

LIBS =  -L$(libdir) \
	-L../fmigrib/lib \
	-L../fminc/lib \
	-L../fmidb/lib \
        -L/usr/lib/oracle/11.2/client64/lib \
	/usr/lib64/libsmartmet_newbase.a \
        -lfmidb \
	-lfmigrib \
	-lfminc \
        -lboost_system \
        -lboost_filesystem \
        -lboost_regex \
	-lpng \
	-ljasper \
	-lclntsh \
	$(HOME)/local/lib/libgrib_api.a \
        -lpthread

# CUDA_LIBS = -L $(CUDA_LIB_PATH) \
#            -lcudart \
#            /home/partio/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a

# Common library compiling template

# Installation directories

processor := $(shell uname -p)

ifeq ($(origin PREFIX), undefined)
  PREFIX = /usr
else
  PREFIX = $(PREFIX)
endif

ifeq ($(processor), x86_64)
  libdir = $(PREFIX)/lib64
else
  libdir = $(PREFIX)/lib
endif

objdir = obj
libdir = lib

includedir = $(PREFIX)/include

ifeq ($(origin BINDIR), undefined)
  bindir = $(PREFIX)/bin
else
  bindir = $(BINDIR)
endif

# rpm variables

CWP = $(shell pwd)
BIN = $(shell basename $(CWP))

# Special modes

ifneq (,$(findstring debug,$(MAKECMDGOALS)))
  CFLAGS = $(CFLAGS_DEBUG)
  LDFLAGS = $(LDFLAGS_DEBUG)
endif

ifneq (,$(findstring profile,$(MAKECMDGOALS)))
  CFLAGS = $(CFLAGS_PROFILE)
  LDFLAGS = $(LDFLAGS_PROFILE)
endif

# Compilation directories

vpath %.cpp source
# vpath %.cu source
vpath %.h include
vpath %.o $(objdir)

# How to install

INSTALL_LIB = install -m 775
INSTALL_DATA = install -m 664

# The files to be compiled

SRCS = $(patsubst source/%,%,$(wildcard *.cpp source/*.cpp))
# CUSRCS = $(patsubst source/%,%,$(wildcard *.cu source/*.cu))

HDRS = $(patsubst include/%,%,$(wildcard *.h include/*.h))
OBJS = $(SRCS:%.cpp=%.o)
# CUOBJS = $(SRCS:%.cu=%.o)

# For rm

OBJFILES = $(OBJS:%.o=obj/%.o)
# CUOBJFILES = $(CUOBJS:%.o=obj/%.o)

MAINSRCS = $(LIB:%=%.cpp)
SUBSRCS = $(filter-out $(MAINSRCS),$(SRCS))
SUBOBJS = $(SUBSRCS:%.cpp=%.o)
SUBOBJFILES = $(SUBOBJS:%.o=obj/%.o)

# CUSUBOBJS = $(CURCS:%.cu=%.o)
# CUSUBOBJFILES = $(SUBOBJS:%.o=obj/%.o)

# For make depend:

ALLSRCS = $(wildcard *.cpp source/*.cpp)

.PHONY: test rpm

# The rules

all: objdir $(LIB)
debug: objdir $(LIB)
release: objdir $(LIB)
profile: objdir $(LIB)

$(LIB): % : $(SUBOBJS) %.o
	$(CC) $(LDFLAGS) -shared -o $(libdir)/lib$(@).so obj/$@.o $(SUBOBJFILES) $(LIBS) $(CUDA_LIBS)
# 	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(CUDA_INCLUDES) -o obj/tpot_cuda.o -c source/tpot_cuda.cu

#$(CUSUBOBJFILES) :
#	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f $(libdir)/*.so $(libdir)/*.a $(OBJFILES) $(CUOBJFILES) *~ source/*~ include/*~

install:
#	$(INSTALL_DATA) $(libdir)/lib$(LIB).a $(NEONS_LIB)

depend:
	gccmakedep -fDependencies -- $(CFLAGS) $(INCLUDES) -- $(ALLSRCS)

objdir:
	@mkdir -p $(objdir)
	@mkdir -p $(libdir)


.SUFFIXES: $(SUFFIXES) .cpp .cu

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(objdir)/$@ $<

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $(objdir)/$@ $<

-include Dependencies
