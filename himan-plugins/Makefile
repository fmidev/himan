LIB = himan-plugins

SCONS_FLAGS=-j 2

# How to install

INSTALL_DATA = install -m 664

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
	   -I/usr/local/grib_api/include \
           -I../himan-lib/include \
           -I../fmidb/include \
           -I../fmigrib/include \
           -I../fminc/include \
           -I$(HOME)/local/include \
           -I/usr/include/oracle \
           -I/usr/include/smartmet/newbase \
	   -I../database/include

# CUDA_INCLUDES = -I include \
# 		-I $(CUDA_INC_PATH) \
# 		-I /home/partio/NVIDIA_GPU_Computing_SDK/C/common/inc

LIBS =  -L$(libdir) \
	-L../fmigrib/lib \
	-L../fminc/lib \
	-L../fmidb/lib \
        -L/usr/lib64/oracle \
	-L/usr/local/grib_api/lib \
	/usr/lib64/libsmartmet_newbase.a \
	-lfmigrib \
	-lfminc \
        -lboost_system \
        -lboost_filesystem \
        -lboost_regex \
	-lpng \
	-ljasper \
	-lclntsh \
	-Llibgrib_api \
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

rpmsourcedir = /home/partio/rpmbuild/SOURCES

# The rules

all release: 
	scons $(SCONS_FLAGS)
debug: 
	scons $(SCONS_FLAGS) --debug-build

clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean
	if [ -a $(LIB).spec ]; \
        then \
          tar -C ../ --exclude .svn \
                   -cf $(rpmsourcedir)/$(LIB).tar $(LIB) ; \
          gzip -f $(rpmsourcedir)/$(LIB).tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB).tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p  $(DESTDIR)/$(INSTALL_TARGET)
	for plugin in build/release/*.so; do \
		echo installing $$plugin ; \
                $(INSTALL_DATA) $$plugin $(DESTDIR)/$(INSTALL_TARGET); \
	done;

