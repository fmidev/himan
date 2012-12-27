LIB = himan

MAINFLAGS = -Wall -W -Wno-unused-parameter -Werror

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
           -I$(includedir) \
           -I/usr/include/oracle \
           -I$(workspace)/himan-plugins/include \
           -I/usr/include/smartmet/newbase \
           -I$(workspace)/fmigrib/include \
           -I$(workspace)/fmidb/include \
	   -I/usr/include/oracle/11.2/client64/

LIBS =  -L$(libdir) \
        -L/usr/lib64 \
	-L../fmigrib/lib \
        -L /usr/lib64/oracle \
	/usr/lib64/libsmartmet_newbase.a \
        -lclntsh \
        /usr/lib64/libboost_system.a \
        /usr/lib64/libboost_filesystem.a \
        /usr/lib64/libboost_iostreams.a \
        /usr/lib64/libboost_thread.a \
	/usr/lib64/libboost_program_options.a \
	/usr/lib64/libboost_regex.a \
	-ldl -licuuc \
	-lz -lbz2 \
	-lpthread \

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

# rpm variables

CWP = $(shell pwd)
BIN = $(shell basename $(CWP))

rpmsourcedir = /home/partio/rpmbuild/SOURCES

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
vpath %.h include
vpath %.o $(objdir)

# How to install

INSTALL_LIB = install -m 775
INSTALL_DATA = install -m 664

# The files to be compiled

SRCS = $(patsubst source/%,%,$(wildcard *.cpp source/*.cpp))
HDRS = $(patsubst include/%,%,$(wildcard *.h include/*.h))
OBJS = $(SRCS:%.cpp=%.o)

OBJFILES = $(OBJS:%.o=obj/%.o)

MAINSRCS = $(LIB:%=%.cpp)
SUBSRCS = $(filter-out $(MAINSRCS),$(SRCS))
SUBOBJS = $(SUBSRCS:%.cpp=%.o)
SUBOBJFILES = $(SUBOBJS:%.o=obj/%.o)

# For make depend:

ALLSRCS = $(wildcard *.cpp source/*.cpp)

.PHONY: test rpm

# The rules

all: objdir $(LIB)
debug: objdir $(LIB)
release: objdir $(LIB)
profile: objdir $(LIB)

$(LIB): $(SUBOBJS)
	$(CC) $(LDFLAGS) -shared -o lib/lib$(LIB).so $(SUBOBJFILES) $(LIBS)
	ar rcs $(libdir)/lib$(LIB).a $(SUBOBJFILES)

clean:
	rm -f $(libdir)/*.so $(libdir)/*.a $(OBJFILES) *~ source/*~ include/*~

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
	mkdir -p  $(DESTDIR)/$(LIBDIR)
	$(INSTALL_DATA) lib/libhiman.so $(DESTDIR)/$(LIBDIR)/libhiman.so
	$(INSTALL_DATA) lib/libhiman.a  $(DESTDIR)/$(LIBDIR)/libhiman.a

depend:
	gccmakedep -fDependencies -- $(CFLAGS) $(INCLUDES) -- $(ALLSRCS)

objdir:
	@mkdir -p $(objdir)
	@mkdir -p $(libdir)


.SUFFIXES: $(SUFFIXES) .cpp

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(objdir)/$@ $<

-include Dependencies
