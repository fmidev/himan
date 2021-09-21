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

ifeq ($(VERSION),)
  VERSION=$(shell date -u +%y).$(shell date -u +%m | sed 's/^0*//').$(shell date -u +%d | sed 's/^0*//')
endif

ifeq ($(RELEASE),)
  RELEASE=$(shell date -u +%H%M).$(shell git rev-parse --short HEAD)
endif

# The rules

all release: createlink
	scons-3 $(SCONS_FLAGS)
debug: createlink
	scons-3 $(SCONS_FLAGS) --debug-build
nocuda: createlink
	scons-3 $(SCONS_FLAGS) --no-cuda-build
check: createlink
	scons-3 CPPCHECK
	scons-3 SCANBUILD
clean:  createlink
	scons-3 -c ; scons-3 --debug-build -c ; rm -f *~ source/*~ include/*~ ; rm -f scons_common.py
createlink:
	if [ ! -e scons_common.py ]; then \
	  ln -fs ../scons_common.py; \
	fi;
rpm:    clean
	cd ../himan-lib && make
	cd ../himan-bin && rpmbuild --build-in-place --define="version $(VERSION)" --define="release $(RELEASE)" -bb himan-bin.spec

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	$(INSTALL_PROG) build/release/himan $(DESTDIR)/$(INSTALL_TARGET)

test:	debug
	cd regression && sh test_all.sh
