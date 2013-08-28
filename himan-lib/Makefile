LIB = himan

SCONS_FLAGS=-j 2

EXTRAFLAGS = -Wpointer-arith \
	-Wcast-qual \
	-Wcast-align \
	-Wwrite-strings \
	-Wconversion \
	-Winline \
	-Wnon-virtual-dtor \
	-Wno-pmf-conversions \
	-Wsign-promo \
	-Wchar-subscripts \
	-Wold-style-cast

DIFFICULTFLAGS = -pedantic -Weffc++ -Wredundant-decls -Wshadow -Woverloaded-virtual -Wunreachable-code -Wctor-dtor-privacy

CC = g++

# Default compiler flags

CFLAGS = -fPIC -std=c++0x -DUNIX -O2 -DNDEBUG $(MAINFLAGS)
LDFLAGS = -Wl,-export-dynamic 

# Special modes

CFLAGS_DEBUG = -fPIC -std=c++0x -DUNIX -O0 -g -DDEBUG $(MAINFLAGS) $(EXTRAFLAGS)
CFLAGS_PROFILE = -DUNIX -O2 -g -pg -DNDEBUG $(MAINFLAGS)

LDFLAGS_DEBUG =
LDFLAGS_PROFILE =

workspace = $(HOME)/workspace

INCLUDES = -I include \
	   		-I/usr/local/grib_api/include \
           	-I$(includedir) \
	   		-I/usr/local/lib \
           	-I/usr/include/oracle \
           	-I$(workspace)/himan-plugins/include \
           	-I/usr/include/smartmet/newbase \
           	-I$(workspace)/fmigrib/include \
           	-I$(workspace)/fmidb/include \
           	-I$(workspace)/database/include \
	   		-I/usr/include/oracle/11.2/client64/

LIBS =  	-L$(libdir) \
        	-L/usr/lib64 \
			-L/usr/local/lib \
			-L../fmigrib/lib \
        	-L /usr/lib64/oracle \
			/usr/lib64/libsmartmet_newbase.a \
        	-lclntsh \
			-ldl -licuuc \
			-lz -lbz2 \
			-lpthread \
        	-lboost_system \
        	-lboost_filesystem \
        	-lboost_iostreams \
        	-lboost_thread \
        	-lboost_program_options \
        	-lboost_regex \
#	-Wl,-whole-archive \
#	-Wl,-no-whole-archive

# Common library compiling template

# Installation directories

processor := $(shell uname -p)

ifeq ($(origin PREFIX), undefined)
  PREFIX = /usr
else
  PREFIX = $(PREFIX)
endif

ifeq ($(processor), x86_64)
  LIBDIR = $(PREFIX)/lib64
else
  LIBDIR = $(PREFIX)/lib
endif

objdir = obj
libdir = lib

includedir = $(PREFIX)/include

ifeq ($(origin BINDIR), undefined)
  bindir = $(PREFIX)/bin
else
  bindir = $(BINDIR)
endif

# How to install

INSTALL_DATA = install -m 664

rpmsourcedir = /home/partio/rpmbuild/SOURCES

INSTALL_TARGET = /usr/lib64

# The rules

all release: 
	scons $(SCONS_FLAGS)
debug: 
	scons $(SCONS_FLAGS) --debug-build

clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean
	if [ -a $(LIB)-lib.spec ]; \
        then \
          tar -C ../ --exclude .svn \
                   -cf $(rpmsourcedir)/$(LIB)-lib.tar $(LIB)-lib ; \
          gzip -f $(rpmsourcedir)/$(LIB)-lib.tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB)-lib.tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	if [ -f "build/release/lib$(LIB).so" ]; then \
		$(INSTALL_DATA) build/release/lib$(LIB).so $(DESTDIR)/$(INSTALL_TARGET); \
	fi;
