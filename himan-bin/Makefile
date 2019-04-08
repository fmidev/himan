PROG = himan

SCONS_FLAGS=-j 1

# How to install

INSTALL_PROG = install -m 755

# rpm variables

CWP = $(shell pwd)
BIN = $(shell basename $(CWP))

rpmsourcedir = /tmp/$(shell whoami)/rpmbuild

INSTALL_TARGET = /usr/bin

.SILENT:createlink

# The rules

all release: createlink
	scons $(SCONS_FLAGS)
debug: createlink
	scons $(SCONS_FLAGS) --debug-build
nocuda: createlink
	scons $(SCONS_FLAGS) --no-cuda-build
clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~ ; rm -f scons_common.py
createlink:
	if [ ! -e scons_common.py ]; then \
	  ln -fs ../scons_common.py; \
	fi;
rpm:    clean createlink
	mkdir -p $(rpmsourcedir) ; \
        if [ -a $(PROG)-bin.spec ]; \
        then \
          tar -C ../ --dereference --exclude-vcs \
                   -zcf $(rpmsourcedir)/$(PROG)-bin.tar.gz $(PROG)-bin ; \
          rpmbuild -ta $(rpmsourcedir)/$(PROG)-bin.tar.gz ; \
          rm -f $(rpmsourcedir)/$(LIB).tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	$(INSTALL_PROG) build/release/himan $(DESTDIR)/$(INSTALL_TARGET)

test:	debug
	cd regression && sh test_all.sh
