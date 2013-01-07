PROB = himan

SCONS_FLAGS=-j 4

# How to install

INSTALL_DATA = install -m 664

# rpm variables

CWP = $(shell pwd)
BIN = $(shell basename $(CWP))

rpmsourcedir = /home/partio/rpmbuild/SOURCES

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
                   -cf $(rpmsourcedir)/$(PROG)-lib.tar $(PROG)-lib ; \
          gzip -f $(rpmsourcedir)/$(PROG)-lib.tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(PROG)-lib.tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	#mkdir -p  $(DESTDIR)/$(PROGDIR)
	scons $(SCONS_FLAGS) /usr/bin

objdir:
	@mkdir -p $(objdir)
	@mkdir -p $(libdir)


