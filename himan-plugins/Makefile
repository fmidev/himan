LIB = himan-plugins

NPROC=$(shell nproc)

SCONS_FLAGS=-j $(NPROC)

# How to install

rpmsourcedir = /tmp/$(shell whoami)/rpmbuild

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
	if [ -a $(LIB).spec ]; \
        then \
          tar -C ../ --dereference --exclude-vcs \
                   -zcf $(rpmsourcedir)/$(LIB).tar.gz $(LIB) ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB).tar.gz ; \
          rm -f $(rpmsourcedir)/$(LIB).tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p  $(DESTDIR)/usr/lib64/himan-plugins $(DESTDIR)/usr/include/himan/plugins
	@for plugin in build/release/*.so; do \
                install -m 755 $$plugin $(DESTDIR)/usr/lib64/himan-plugins; \
	done;
	@for h in $(shell ls include/*); do install -m 644 $$h $(DESTDIR)/usr/include/himan/plugins; done


