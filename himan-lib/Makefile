LIB = himan

SCONS_FLAGS=-j 2

# How to install

INSTALL_DATA = install -m 664

rpmsourcedir = /home/partio/rpmbuild/SOURCES

INSTALL_TARGET = /usr/lib64

# The rules

all release: 
	scons $(SCONS_FLAGS)
debug: 
	scons $(SCONS_FLAGS) --debug-build

clean:
	scons -c ; scons --debug-build -c ; rm -f *~ source/*~ include/*~

rpm:    clean
	if [ -a $(LIB)-lib.spec ]; \
        then \
          tar -C ../ --exclude .svn \
                   -cf $(rpmsourcedir)/$(LIB)-lib.tar $(LIB)-lib ; \
          gzip -f $(rpmsourcedir)/$(LIB)-lib.tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB)-lib.tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p $(DESTDIR)/$(INSTALL_TARGET)
	if [ -f "build/release/lib$(LIB).so" ]; then \
		$(INSTALL_DATA) build/release/lib$(LIB).so $(DESTDIR)/$(INSTALL_TARGET); \
	fi;
