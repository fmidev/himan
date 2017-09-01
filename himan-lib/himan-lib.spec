%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define LIBNAME himan-lib
Summary: himan core library
Name: %{LIBNAME}
Version: 17.8.21
Release: 1.el7.fmi
License: MIT
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: libfmidb >= 17.8.10
Requires: libfmigrib >= 17.4.6
Requires: gdal

%if %{defined suse_version}
BuildRequires: bzip2
Requires: grib_api
%else
BuildRequires: bzip2-devel
BuildRequires: redhat-rpm-config
BuildRequires: cub
BuildRequires: cuda-8-0
BuildRequires: gcc-c++ >= 4.8.2
Requires: eccodes
%endif
BuildRequires: libfmidb-devel >= 17.8.10
BuildRequires: libfmigrib-devel >= 17.4.6
BuildRequires: zlib-devel
BuildRequires: boost-devel >= 1.53
BuildRequires: smartmet-library-newbase-devel >= 17.4.4
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
* Mon Aug 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.21-1.fmi
- General code cleanup
- Level type fixes
- Adding async execution mode for plugins
* Mon Aug 14 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.14-1.fmi
- Database access optimization
* Thu Aug  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.3-1.fmi
- Minor changes to ensemble, matrix
* Tue Aug  1 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.1-1.fmi
- Removing logger_factory
* Mon Jul 17 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.7.17-1.fmi
- Removing timer_factory
* Thu Jun 22 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.22-1.fmi
- Bugfix for packed_data
* Wed Jun 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.21-1.fmi
- Minor tweaks to configuration and ensemble
* Thu Jun  1 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.1-1.fmi
- Tweak to csv reading
* Tue May 30 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.30-1.fmi
- Tweaks to Wobf code
* Tue May 23 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.23-1.fmi
- Support for info serialization
* Mon May 15 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.15-1.fmi
- Updates to previ / station data handling
* Tue Apr 18 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.18-1.fmi
- Changes to database plugin headers
* Tue Apr 11 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.11-1.fmi
- Disable database access when doing newbase interpolation
* Thu Apr  6 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.6-1.fmi
- New fmigrib
- New fmidb
- New newbase
- New eccodes
- Add nodatabase mode 
- Improved findheight_gt/lt modifiers
* Mon Apr  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.3-1.fmi
- metutil: Add moist lift approximation function 
* Wed Mar 29 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.29-1.fmi
- Add functionality to ensemble
* Wed Mar 15 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.15-1.fmi
- Increased air parcel lift accuracy
* Fri Feb 17 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.17-1.fmi
- raw_time and time_ensemble fixes
* Mon Feb 13 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.13-1.fmi
- New newbase
- time_lagged ensemble
- Improved aux file read
- Leap year fix for raw_time
- CAPE MoistLift optimization
- Apply scale+base when writing querydata
* Mon Jan 30 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.30-1.fmi
- Dependency to open sourced Newbase
- Fix for Lambert vector component rotation
* Tue Jan  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.3-1.fmi
- Additional ensemble methods to luatool
* Thu Dec 29 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.29-1.fmi
- Reworked vector component rotation
* Thu Dec 15 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.15-1.fmi
- Min2D implementation
* Fri Dec  9 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.9-1.fmi
- SLES accomodations
- New key origintimes for configuration files
- fmidb API change
* Wed Dec  7 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.7-1.fmi
- Cuda 8
* Tue Nov 22 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.22-8.fmi
- ensemble works with missing values
* Tue Nov  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.8-1.fmi
- MEPS compatibility
* Tue Nov  1 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.1-1.fmi
- Fixes and additions to modifier
* Thu Oct 27 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.27-1.fmi
- New level 'depth layer'
* Wed Oct 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.26-1.fmi
- Introducing time_ensemble
* Mon Oct 24 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.24-1.fmi
- SLES compatibility fixes
- General code cleanup
* Thu Oct  6 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.6-1.fmi
- Fix in json_parser
* Wed Oct  5 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.5-1.fmi
- Support grib index reading
* Wed Sep 28 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.28-1.fmi
- Lambert projection support
* Mon Sep 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.12-1.fmi
- Latest time fixes for json_parser
* Thu Sep  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.8-1.fmi
- fmigrib api change
* Wed Aug 31 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.31-1.fmi
- New release
* Tue Aug 30 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.30-1.fmi
- New release
* Tue Aug 23 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.23-1.fmi
- New release
* Mon Aug 15 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.15-1.fmi
- New fmigrib
* Wed Aug 10 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.10-1.fmi
- New release
* Mon Jun 27 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.20-1.fmi
- Fixes to metutil
- Ensemble changes
* Mon Jun 20 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.20-1.fmi
- Smarttool compatibility functions to metutil
- Ensemble changes
* Thu Jun 16 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.16-1.fmi
- New release
* Mon Jun 13 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.13-1.fmi
- Forecast type fixes
* Thu Jun  9 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.9-1.fmi
- Ensemble helper additions
- Option to explicitly select grib1 output in json
* Mon Jun  6 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.6-1.fmi
- New ensemble helper
- New method of passing parameters from json to plugin
- New method Max2D() at numerical_functions
* Thu May 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.26-1.fmi
- fmidb header change
* Thu May 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.12-1.fmi
- New newbase
* Mon May  2 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.2-1.fmi
- Changes in metutil and point
* Tue Apr 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.26-1.fmi
- New release
* Tue Apr 19 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.19-1.fmi
- Filter2D() GPU version
* Thu Apr 14 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.14-1.fmi
- Moving Filter2D() to numerical_functions
- ThetaE signature change
* Fri Apr  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.8-1.fmi
- Simplifying code, thanks partly to Cuda 7.5 and C++11
* Mon Apr  4 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.4-1.fmi
- Fix reading of latest analysistime from Radon
* Tue Feb 23 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.23-1.fmi
- Fix reading of empty packed grids
* Mon Feb 22 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.22-1.fmi
- New release
* Thu Feb 18 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.18-1.fmi
- Minor change in util
* Fri Feb 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.12-1.fmi
- New newbase and fmidb
* Thu Jan 14 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.14-1.fmi
- modifier fixes
* Tue Jan  5 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.5-1.fmi
- Support for dynamic memory allocation
* Mon Jan  4 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.4-1.fmi
- New release
* Mon Dec 21 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.21-1.fmi
- API change for metutil
* Thu Dec 17 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.17-1.fmi
- API change for info
* Tue Dec  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.8-1.fmi
- Minor changes
* Wed Nov 25 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.25-2.fmi
- More modifier_mean tweaks
* Wed Nov 25 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.25-1.fmi
- Modifier tweak
* Fri Nov 20 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.20-1.fmi
- Reducing memory footpring
* Thu Nov 19 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.19-1.fmi
- New release
* Fri Nov  6 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.6-1.fmi
- Configuration option cache_limit
* Fri Oct 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.30-1.fmi
- Changes in regular_grid and matrix
* Mon Oct 26 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.26-1.fmi
- Some changes to metutil
* Wed Sep 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.30-1.fmi
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
