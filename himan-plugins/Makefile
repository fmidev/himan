LIB = plugins

SCONS_FLAGS=-j 6

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
                   -cf $(rpmsourcedir)/$(LIB)-lib.tar $(LIB)-lib ; \
          gzip -f $(rpmsourcedir)/$(LIB)-lib.tar ; \
          rpmbuild -ta $(rpmsourcedir)/$(LIB)-lib.tar.gz ; \
        else \
          echo $(rpmerr); \
        fi;

install:
	mkdir -p  $(DESTDIR)/$(LIBDIR)
#	$(INSTALL_DATA) lib/libhiman.so $(DESTDIR)/$(LIBDIR)/libhiman.so
#	$(INSTALL_DATA) lib/libhiman.a  $(DESTDIR)/$(LIBDIR)/libhiman.a

objdir:
	@mkdir -p $(objdir)
	@mkdir -p $(libdir)


