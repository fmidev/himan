LIB = himan

NPROC=$(shell nproc)

SCONS_FLAGS=-j $(NPROC)

rpmsourcedir=/tmp/$(shell whoami)/rpmbuild

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
	rpmbuild --build-in-place -bb himan-lib.spec
install:
	mkdir -p $(DESTDIR)/usr/lib64 $(DESTDIR)/usr/include/himan
	@if [ -f "build/release/lib$(LIB).so" ]; then \
		install -m 755 build/release/lib$(LIB).so $(DESTDIR)/usr/lib64; \
	fi
	@for h in $(shell ls include/*); do install -m 644 $$h $(DESTDIR)/usr/include/himan; done
