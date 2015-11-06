%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define LIBNAME himan-lib
Summary: himan core library
Name: %{LIBNAME}
Version: 15.11.6
Release: 1%{dist}.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: libfmidb >= 15.10.9
BuildRequires: zlib-devel
BuildRequires: bzip2-devel
BuildRequires: boost-devel >= 1.55
BuildRequires: redhat-rpm-config
BuildRequires: cub
BuildRequires: libfmidb-devel >= 15.10.9
BuildRequires: gcc-c++ >= 4.8.2
BuildRequires: cuda-7-5
BuildRequires: libsmartmet-newbase-devel >= 15.10.26
BuildRequires: scons
Provides: libhiman.so

%description
FMI himan -- hila manipulaatio -- core library

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-lib" 

%build
make

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0644)
%{_libdir}/libhiman.so

%changelog
* Fri Nov  6 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.6-1.fmi
- Configuration option cache_limit
* Fri Oct 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.30-1.fmi
- Changes in regular_grid and matrix
* Mon Oct 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.26-1.fmi
- Some changes to metutil
* Mon Sep 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.30-1.fmi
- cuda 7.5
* Mon Sep 28 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.28-1.fmi
- New release 
* Wed Sep  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.9-1.fmi
- Fix for dewpoint calculation when RH=0
- Remove obsolete code from fetcher
* Tue Sep  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.8-1.fmi
- plugin inheritance hierarchy modifications
* Wed Sep  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.2-1.fmi
- fmidb api change
* Mon Aug 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.24-1.fmi
- Additions to numerical integration functions
* Mon Aug 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.10-1.fmi
- New interpolation method (not enabled)
- New vertical integration namespace
* Mon Jun 22 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.6.22-1.fmi
- Changes in metutil and modifier
* Wed May 27 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.27-1.fmi
- Improving external packing support
- Fixes and additions to metutil
* Mon May 11 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.11-1.fmi
- gzip and bzip2 support for grib (STU-2152)
- Link with cuda 6.5 due to performance issues (HIMAN-96)
* Tue Apr 28 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.28-1.fmi
- Linking with Cuda 7 (HIMAN-96)
* Fri Apr 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.24-1.fmi
- Linking with newer fmidb
* Mon Apr 13 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.13-1.fmi
- Better debug output to matrix
* Fri Apr 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.10-1.fmi
- Link with boost 1.57 and dynamic version of newbase
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-1.fmi
- Major update to add forecast type based calculations
* Mon Mar 30 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.3.30-1.fmi
- New release
* Wed Feb 18 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.18-1.fmi
- New release
* Mon Feb 16 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.16-1.fmi
- Fix regular_grid LatLon() with +x-y scanning mode
* Tue Feb 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.10-1.fmi
- Changes in json_parser
* Tue Feb 03 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.03-1.fmi
- Changes in util 
* Mon Jan 26 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.26-1.fmi
- RHEL7 adjustment
- Other minor fixes
* Wed Jan  7 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.7-1.fmi
- Changes related to radon access
* Fri Jan  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.2-1.fmi
- Changes in modifier
* Mon Dec 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.29-1.fmi
- Minor changes
* Thu Dec 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.18-1.fmi
- Introducing irregular_grid
* Wed Dec 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.17-1.fmi
- Supporting lists in plugin-specific configuration directives
- station-class
* Mon Dec  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.8-1.fmi
- Large internal changes
* Mon Dec  1 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.1-1.fmi
- New modifier
* Tue Nov 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.25-1.fmi
- Removed logger from classes that don't need it
- Add initial cuda grib packing support (not enabled yet)
* Thu Nov 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.13-1.fmi
- Removed support for NFmiPoint
- No more double memory allocation for cuda plugins
* Tue Nov 04 2014 Andreas Tack <andreas.tack@fmi.fi> - 14.11.4-1.fmi
- Add modifier type to string
* Thu Oct 30 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.30-1.fmi
- Simplifying info-class more (HIMAN-69)
* Tue Oct 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.28-1.fmi
- Simplifying info-class more (HIMAN-69)
* Mon Oct 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.20-1.fmi
- Simplifying info-class more (HIMAN-69)
* Thu Oct 16 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.16-1.fmi
- Simplifying info-class (HIMAN-69)
* Mon Oct 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.13-1.fmi
- Changes in modifier_integral
- Changes in unpacking grib
* Mon Oct  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.6-1.fmi
- HIMAN-69
* Tue Sep 30 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.30-1.fmi
- Linking with newer grib_api that possibly fixes HIMAN-58
* Wed Sep 24 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.24-1.fmi
- Add ForecastStep() to class configuration
* Tue Sep 23 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.23-2.fmi
- Add Type() to modifier
* Tue Sep 23 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.23-1.fmi
- Adding Filter2D() function
- More functionality in metutil
- modifier updates with (modifer_findvalue)
- simple_packed updates: no more memory allocation in library calls
* Thu Aug 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.28-1.fmi
- Dewpoint calculation moved to metutil
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-2.fmi
- Fixes for modifier
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-1.fmi
- Timer time resolution change from microseconds to milliseconds
- modifier_integral class added 
* Wed Jun 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.18-1.fmi
- Initial build with Cuda6 (HIMAN-57)
* Thu Jun  5 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.5-1.fmi
- Minor changes
* Fri May  9 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.8-1.fmi
- Minor changes
* Tue May  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.6-1.fmi
- Add still unfinished namespace metutil
- Replacing some #includes with forward declarations
- Consistent way of implementing factories internal pointer handling
* Tue Apr 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.29-1.fmi
- Small changes
* Thu Apr 10 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.10-1.fmi
- Add forecast hour as sub directory to ref storage file name 
* Mon Apr  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.7-1.fmi
- Small changes
* Thu Mar 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.20-1.fmi
- Small changes
* Tue Mar 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.18-1.fmi
- Name change for level top
* Mon Mar 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.17-1.fmi
- Small change to configuration
* Wed Mar 12 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.12-1.fmi
- Updated configuration file options (source_geom_name, file_type)
- Initial support for grib packing in cuda
* Tue Feb 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.25-1.fmi
- Add support for setting cuda device id
* Tue Feb 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.18-1.fmi
- Bugfixes
* Mon Feb 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.17-1.fmi
- Added functionality to util::
- info::Merge() fixes
* Tue Feb 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.11-1.fmi
- Added info_simple
- Changes in some of util:: namespace meteorological functions
* Mon Jan 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.20-1.fmi
- Fixes for util::RelativeTopgraphy()
* Tue Jan  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.7-1.fmi
- Accumulated changes in many classes wrt hybrid_height and preform_hybrid
- Link with grib_api 1.11.0
* Wed Dec 11 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.12.1-1.fmi
- Changes in modifier
* Mon Nov 25 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.25-1.fmi
- Latest changes
* Tue Nov 12 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.12-1.fmi
- Changes in modifier
* Mon Oct 14 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.14-1.fmi
- Fix to HIMAN-13
* Tue Oct  8 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.8-1.fmi
- Small changes
* Wed Oct  2 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.2-1.fmi
- Small changes wrt logging
* Thu Sep 26 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.6-1.fmi
- Change in himan::level internals
* Wed Sep 25 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.25-1.fmi
- Check cuda calls for errors
* Mon Sep 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.23-1.fmi
- Fix for HIMAN-15
* Thu Sep  5 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.5-1.fmi
- Functions to support preform_pressure
* Tue Sep  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.3-1.fmi
- Fix bug that crashes himan (unresolved cuda-symbols at excutable)
* Fri Aug 30 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.30-1.fmi
- Latest changes
- First compilation on scout.fmi.fi
* Thu Aug 22 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.22-1.fmi
- Latest changes
* Wed Aug 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.21-1.fmi
- Latest changes
- Linking with new version of fmigrib to avoid grib_api bug crashing the 
  program (SUP-592 @ http://software.ecmwf.int)
* Fri Aug 16 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.16-1.fmi
- Latest changes
- First release for masala-cluster
* Mon Mar 11 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.3.11-1.fmi
- Latest changes
* Thu Feb 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.21-1.fmi
- Latest changes
* Tue Feb 19 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.19-1.fmi
- Latest changes
* Mon Feb 18 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.18-1.fmi
- Latest changes
* Tue Feb  5 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.5-1.fmi
- Latest changes
* Thu Jan 31 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.31-1.fmi
- Latest changes
* Thu Jan 24 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.24-1.fmi
- Latest changes
* Wed Jan 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.23-1.fmi
- Use debug build
* Mon Jan 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.21-1.fmi
- Bugfixes
* Tue Jan 15 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.15-1.fmi
- First attempt for production-ready release
* Thu Dec 27 2012 Mikko Partio <mikko.partio@fmi.fi> - 12.12.27-1.fmi
- Initial build
