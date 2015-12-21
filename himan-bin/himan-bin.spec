%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define BINNAME himan-bin
Summary: himan executable
Name: %{BINNAME}
Version: 15.12.21
Release: 1%{dist}.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: himan-lib >= 15.11.6
Requires: boost-program-options
Requires: boost-system
Requires: boost-regex
Requires: boost-iostreams
Requires: boost-thread
Requires: libsmartmet-newbase >= 15.11.10
Requires: libsmartmet-smarttools >= 15.4.9
Requires: jasper
Requires: grib_api >= 1.14.0
Requires: oracle-instantclient-basic
Requires: gdal >= 1.11.0
Requires: bzip2-libs
Requires: zlib
BuildRequires: redhat-rpm-config
BuildRequires: boost-devel >= 1.55
BuildRequires: gcc-c++ >= 4.8.2
BuildRequires: cuda-7-5
BuildRequires: scons
Provides: himan

AutoReqProv:	no

%description
FMI himan -- hilojen manipulaatio -- executable

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-bin" 

%build
make

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0775)
%{_bindir}/himan

%changelog
* Mon Dec 21 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.21-1.fmi
- himan lib API change
* Thu Nov 19 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.19-1.fmi
- Remove calculated data from memory once plugin is finished
* Wed Sep 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.30-1.fmi
- Cuda 7.5
* Tue Sep  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.8-1.fmi
- plugin inheritance hierarchy modifications
* Wed Sep  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.2-1.fmi
- grib_api 1.14
* Mon Aug 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.24-1.fmi
- New command line option --no-cuda-interpolation
* Wed May 27 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.27-1.fmi
- New command line option --compression,-c 
* Mon May 11 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.11-1.fmi
- Link with cuda 6.5 due to performance issues (HIMAN-96)
* Tue Apr 28 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.28-1.fmi
- Linking with Cuda 7 (HIMAN-96)
* Fri Apr 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.10-1.fmi
- Link with boost 1.57 and dynamic version of newbase
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-1.fmi
- New release
* Mon Mar 30 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.3.30-1.fmi
- New release
* Tue Feb 10 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.10-1.fmi
- Changes in himan-lib headers
* Tue Feb  3 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.03-1.fmi
- Bugfix in unstagger
- Changes in querydata
- Changes in CSV
- Changes in NCL
* Mon Jan 26 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.26-1.fmi
- CSV plugin
- RHEL7 compatibility
* Wed Jan  7 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.7-1.fmi
- New options -N and -R
* Wed Dec 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.17-1.fmi
- Changes in himan-lib headers
* Mon Dec  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.8-1.fmi
- Large internal changes in himan-lib
* Mon Dec  1 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.1-1.fmi
- Changes in himan-lib headers
* Tue Nov 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.25-1.fmi
- Initial support for enabling cuda packing/unpacking with separate switches
* Thu Nov 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.13-1.fmi
- Changes in himan-lib headers
* Tue Nov 04 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.4-1.fmi
- Changes in himan-lib headers
* Thu Oct 30 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.30-1.fmi
- Changes in himan-lib headers
* Tue Oct 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.28-1.fmi
- Changes in himan-lib headers
* Mon Oct 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.20-1.fmi
- Changes in himan-lib headers
* Mon Oct  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.6-1.fmi
- Changes in himan-lib headers
* Wed Sep 24 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.24-1.fmi
- Changes in himan-lib headers
* Tue Sep 23 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.23-1.fmi
- Changes in himan-lib headers
* Mon Sep  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.8-1.fmi
- Compling and linkin with -pie (HIMAN-51)
* Tue Aug 12 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.12-1.fmi
- Backwards compatibility for renamed plugins
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-2.fmi
- Fix bug that forced himan to an eternal loop
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-1.fmi
- Add overall timings on himan execution
- Add cuda functionality since plugin pcuda is removed
* Fri Jun 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.18-1.fmi
- Initial build with Cuda6 (HIMAN-57)
* Thu Jun  5 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.5-1.fmi
- Changes in himan-lib
* Tue May  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.6-1.fmi
- Changes in himan-lib
* Tue Apr 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.29-1.fmi
- Extra support for renamed plugin 'kindex' (new name 'stability')
* Mon Apr  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.7-1.fmi
- Changes in himan-lib
* Tue Mar 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.18-1.fmi
- Changes in himan-lib
* Mon Mar 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.17-1.fmi
- Small change in himan-lib/configuration
* Wed Mar 12 2014  Mikko Partio <mikko.partio@fmi.fi> - 14.3.12-1.fmi
- Updated configuration file options (source_geom_name, file_type)
* Tue Feb 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.25-1.fmi
- Add support for setting cuda device id
* Tue Jan  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.7-1.fmi
- Changes in himan-lib
* Wed Nov 13 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.13-1.fmi
- Fix crashing himan caused by missing cudar and clntsh libraries at linking stage
* Tue Nov 12 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.12-1.fmi
- Extra support for renamed plugin 'precipitation' (new name 'split_sum')
* Wed Oct  2 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.2-1.fmi
- Changes in himan-lib
* Mon Sep 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.23-1.fmi
- Changes in configuration and json_parser
* Tue Sep  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.3-1.fmi
- Fix bug that crashes himan (unresolved cuda-symbols at excutable)
* Fri Aug 30 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.30-1.fmi
- Latest changes
- First compilation on scout.fmi.fi
* Fri Aug 16 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.16-1.fmi
- Latest changes
- First release for masala-cluster
* Mon Mar 11 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.3.11-1.fmi
- Latest changes
* Thu Feb 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.21-1.fmi
- Latest changes
* Mon Feb 18 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.18-1.fmi
- Latest changes
* Tue Feb  5 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.5-1.fmi
- Latest changes
* Thu Jan 31 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.31-1.fmi
- Latest changes
* Mon Jan 14 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.14-1.fmi
- Initial build
