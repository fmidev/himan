PROG = himan-scripts

# How to install

rpmsourcedir = /tmp/$(shell whoami)/rpmbuild

VERSION=$(shell date -u +%y).$(shell date -u +%m | sed 's/^0*//').$(shell date -u +%d | sed 's/^0*//')
RELEASE=$(shell date -u +%H%M).$(shell git rev-parse --short HEAD)

# The rules

rpm:    
	mkdir -p $(rpmsourcedir)
	tar -C ../ --dereference --exclude-vcs \
                   -zcf $(rpmsourcedir)/$(PROG).tar.gz $(PROG) ; \
          rpmbuild --define="version $(VERSION)" --define="release $(RELEASE)" -ta $(rpmsourcedir)/$(PROG).tar.gz ; \
          rm -f $(rpmsourcedir)/$(LIB).tar.gz ; \
