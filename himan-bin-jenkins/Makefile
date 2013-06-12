PROG = himan

SCONS_FLAGS=-j 4

# How to install

INSTALL_PROG = install -m 755

CC = g++

# Default compiler flags

CFLAGS = -std=c++0x -DUNIX -O2 -DNDEBUG $(MAINFLAGS)
LDFLAGS = -Wl,--export-dynamic

# Special modes

CFLAGS_DEBUG = -std=c++0x -DUNIX -O0 -g -DDEBUG $(MAINFLAGS) $(EXTRAFLAGS)
CFLAGS_PROFILE = -std=c++0x -DUNIX -O2 -g -pg -DNDEBUG $(MAINFLAGS)

LDFLAGS_DEBUG = -Wl,--export-dynamic
LDFLAGS_PROFILE =

workspace = $(HOME)/workspace

INCLUDES = -I include \
           -I$(includedir) \
           -I$(workspace)/himan-lib/include \
           -I$(workspace)/himan-plugins/include \
           -I/usr/include/smartmet/newbase \
	   -I/usr/local/lib
	
LIBS =  -L$(libdir) \
        -L$(workspace)/himan-lib/lib \
	-lhiman 

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

rpmsourcedir = /home/partio/rpmbuild/SOURCES

INSTALL_TARGET = /usr/bin

# The rules

all release: 
	scons $(SCONS_FLAGS)
debug: 
	scons $(SCONS_FLAGS) --debug-build

clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean
	if [ -a $(PROG)-bin.spec ]; \
        then \
          tar -C ../ --exclude .svn \
                   -cf $(rpmsourcedir)/$(PROG)-bin.tar $(PROG)-bin ; \
          gzip -f $(rpmsourcedir)/$(PROG)-bin.tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(PROG)-bin.tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	$(INSTALL_PROG) build/release/himan $(DESTDIR)/$(INSTALL_TARGET)

test:	debug
	cd regression && sh test_all.sh
