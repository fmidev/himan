PROG = himan

SCONS_FLAGS=-j 4

# How to install

INSTALL_PROG = install -m 755

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
