LIB = himan-plugins

NPROC=$(shell nproc)

SCONS_FLAGS=-j $(NPROC)

# How to install

rpmsourcedir = /tmp/$(shell whoami)/rpmbuild

.SILENT:createlink

VERSION=$(shell date -u +%y).$(shell date -u +%m | sed 's/^0*//').$(shell date -u +%d | sed 's/^0*//')
RELEASE=$(shell date -u +%H%M).$(shell git rev-parse --short HEAD)

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
	rpmbuild --build-in-place --define="version $(VERSION)" --define="release $(RELEASE)" -bb himan-plugins.spec
install:
	mkdir -p  $(DESTDIR)/usr/lib64/himan-plugins $(DESTDIR)/usr/include/himan/plugins
	@for plugin in build/release/*.so; do \
                install -m 755 $$plugin $(DESTDIR)/usr/lib64/himan-plugins; \
	done;
	@for h in $(shell ls include/*); do install -m 644 $$h $(DESTDIR)/usr/include/himan/plugins; done


