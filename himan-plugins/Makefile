LIB = himan-plugins

SCONS_FLAGS=-j 8

# How to install

INSTALL_DATA = install -m 664

INSTALL_TARGET = /usr/lib64/himan-plugins

rpmsourcedir = /tmp/$(shell whoami)/rpmbuild

# The rules

all release: 
	scons $(SCONS_FLAGS)
debug: 
	scons $(SCONS_FLAGS) --debug-build
nocuda: 
	scons $(SCONS_FLAGS) --no-cuda-build

clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean
	mkdir -p $(rpmsourcedir) ; \
	if [ -a $(LIB).spec ]; \
        then \
          tar -C ../ --exclude .svn \
                   -cf $(rpmsourcedir)/$(LIB).tar $(LIB) ; \
          gzip -f $(rpmsourcedir)/$(LIB).tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB).tar.gz ; \
          rm -f $(rpmsourcedir)/$(LIB).tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p  $(DESTDIR)/$(INSTALL_TARGET)
	for plugin in build/release/*.so; do \
		echo installing $$plugin ; \
                $(INSTALL_DATA) $$plugin $(DESTDIR)/$(INSTALL_TARGET); \
	done;

