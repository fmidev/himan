LIB = himan-plugins

SCONS_FLAGS=-j 2

# How to install

INSTALL_DATA = install -m 664

# rpm variables

CWP = $(shell pwd)
BIN = $(shell basename $(CWP))

rpmsourcedir = /home/partio/rpmbuild/SOURCES

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
                   -cf $(rpmsourcedir)/$(LIB).tar $(LIB) ; \
          gzip -f $(rpmsourcedir)/$(LIB).tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB).tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p  $(DESTDIR)/$(LIBDIR)
#	$(INSTALL_DATA) lib/libhiman.so $(DESTDIR)/$(LIBDIR)/libhiman.so
#	$(INSTALL_DATA) lib/libhiman.a  $(DESTDIR)/$(LIBDIR)/libhiman.a

