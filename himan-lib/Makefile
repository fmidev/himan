LIB = himan

NPROC=$(shell nproc)

SCONS_FLAGS=-j $(NPROC)

# How to install

INSTALL_DATA = install -m 755

#rpmsourcedir = /home/partio/rpmbuild/SOURCES
rpmsourcedir=/tmp/$(shell whoami)/rpmbuild

INSTALL_TARGET = /usr/lib64

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
	mkdir -p $(rpmsourcedir) ; \
	if [ -a $(LIB)-lib.spec ]; \
        then \
          tar -C ../ --dereference --exclude-vcs \
                   -zcf $(rpmsourcedir)/$(LIB)-lib.tar.gz $(LIB)-lib ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB)-lib.tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	if [ -f "build/release/lib$(LIB).so" ]; then \
		$(INSTALL_DATA) build/release/lib$(LIB).so $(DESTDIR)/$(INSTALL_TARGET); \
	fi;
