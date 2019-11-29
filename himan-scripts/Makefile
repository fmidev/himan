PROG = himan-scripts

# How to install

rpmsourcedir = /tmp/$(shell whoami)/rpmbuild

# The rules

rpm:    
	mkdir -p $(rpmsourcedir)
	tar -C ../ --dereference --exclude-vcs \
                   -zcf $(rpmsourcedir)/$(PROG).tar.gz $(PROG) ; \
          rpmbuild -ta $(rpmsourcedir)/$(PROG).tar.gz ; \
          rm -f $(rpmsourcedir)/$(LIB).tar.gz ; \
