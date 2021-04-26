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
check: createlink
	scons CPPCHECK
	scons SCANBUILD
clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~ ; rm -f scons_common.py
createlink:
	if [ ! -e scons_common.py ]; then \
	  ln -fs ../scons_common.py; \
	fi;
rpm:    clean createlink
	cd ../himan-lib && make
	cd ../himan-bin && rpmbuild --build-in-place -bb himan-bin.spec

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	$(INSTALL_PROG) build/release/himan $(DESTDIR)/$(INSTALL_TARGET)

test:	debug
	cd regression && sh test_all.sh
