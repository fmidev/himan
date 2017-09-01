%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define LIBNAME himan-plugins
Summary: himan-plugins library
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
Requires: oracle-instantclient-basic >= 11.2.0.3.0
Requires: himan-lib >= 17.8.14
Requires: lua >= 5.1.4
Requires: unixODBC
Requires: libfmigrib >= 17.7.31
Requires: libfmidb >= 17.8.10
Requires: smartmet-library-newbase >= 17.4.4
Requires: libpqxx

%if %{defined suse_version}
Requires: libjasper
Requires: grib_api
%else
BuildRequires: gdal-devel
BuildRequires: gcc-c++ >= 4.8.2
BuildRequires: cuda-8-0
BuildRequires: eccodes-devel
BuildRequires: redhat-rpm-config
BuildRequires: cusp >= 0.5.1
Requires: jasper-libs
Requires: eccodes
%endif
BuildRequires: libfmidb-devel >= 17.8.10
BuildRequires: libfmigrib-devel >= 17.7.31
BuildRequires: smartmet-library-newbase-devel >= 17.4.4
BuildRequires: scons
BuildRequires: libluabind >= 0.9.3-3
BuildRequires: boost-devel >= 1.53
BuildRequires: scons
BuildRequires: oracle-instantclient-devel >= 11.2.0.3.0

%description
FMI himan-plugins -- hila manipulaatio -- plugin library

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-plugins" 

%build
make

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0644)
%{_libdir}/himan-plugins/libabsolute_humidity.so
%{_libdir}/himan-plugins/libcache.so
%{_libdir}/himan-plugins/libcape.so
%{_libdir}/himan-plugins/libcloud_code.so
%{_libdir}/himan-plugins/libcsv.so
%{_libdir}/himan-plugins/libdensity.so
%{_libdir}/himan-plugins/libdewpoint.so
%{_libdir}/himan-plugins/libfetcher.so
%{_libdir}/himan-plugins/libfog.so
%{_libdir}/himan-plugins/libfractile.so
%{_libdir}/himan-plugins/libgrib.so
%{_libdir}/himan-plugins/libgust.so
%{_libdir}/himan-plugins/libhitool.so
%{_libdir}/himan-plugins/libhybrid_height.so
%{_libdir}/himan-plugins/libhybrid_pressure.so
%{_libdir}/himan-plugins/libicing.so
%{_libdir}/himan-plugins/libmonin_obukhov.so
%{_libdir}/himan-plugins/libluatool.so
%{_libdir}/himan-plugins/libneons.so
%{_libdir}/himan-plugins/libncl.so
%{_libdir}/himan-plugins/libpop.so
%{_libdir}/himan-plugins/libpot.so
%{_libdir}/himan-plugins/libprecipitation_rate.so
%{_libdir}/himan-plugins/libpreform_hybrid.so
%{_libdir}/himan-plugins/libpreform_pressure.so
%{_libdir}/himan-plugins/libprobability.so
%{_libdir}/himan-plugins/libqnh.so
%{_libdir}/himan-plugins/libquerydata.so
%{_libdir}/himan-plugins/libradon.so
%{_libdir}/himan-plugins/librelative_humidity.so
%{_libdir}/himan-plugins/libseaicing.so
%{_libdir}/himan-plugins/libsplit_sum.so
%{_libdir}/himan-plugins/libstability.so
%{_libdir}/himan-plugins/libtke.so
%{_libdir}/himan-plugins/libtpot.so
%{_libdir}/himan-plugins/libtransformer.so
%{_libdir}/himan-plugins/libturbulence.so
%{_libdir}/himan-plugins/libunstagger.so
%{_libdir}/himan-plugins/libvisibility.so
%{_libdir}/himan-plugins/libvvms.so
%{_libdir}/himan-plugins/libweather_symbol.so
%{_libdir}/himan-plugins/libweather_code_1.so
%{_libdir}/himan-plugins/libweather_code_2.so
%{_libdir}/himan-plugins/libwindvector.so
%{_libdir}/himan-plugins/libwriter.so

%changelog
* Mon Aug 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.21-1.fmi
- General code cleanup
- Level type fixes
- Adding async execution mode for plugins
* Mon Aug 14 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.14-1.fmi
- Database access optimization
- split_sum cleanup
- visibility fix for inconsistent precipitation parameter access wrt preform_hybrid
* Tue Aug  8 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.8-1.fmi
- Calculate meps hybrid level height using geopotential
* Thu Aug  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.3-1.fmi
- Re-execute radon SELECT if deadlock occurs
* Tue Aug  1 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.1-1.fmi
- Removing logger_factory
- One less memory allocation when reading grib
* Mon Jul 17 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.7.17-1.fmi
- Removing timer_factory
- Manual ANALYZE when first inserting to radon
* Wed Jun 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.21-3.fmi
- Fix for neons grib2 metadata
* Wed Jun 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.21-2.fmi
- Fix fractile so that missing data does not stop Himan execution
* Wed Jun 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.21-1.fmi
- Previ data fetch optimizations
- POT version 2.1
- Fix probability so that missing data does not stop Himan execution
- Correct thread count for hybrid_height/ensembles
* Thu Jun 15 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.15-1.fmi
- More accurate MOL
- Cape plugin support for meps
* Mon Jun 12 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.12-2.fmi
- New producer 243 for per-member post-processed ECMWF ENS
* Mon Jun 12 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.12-1.fmi
- cape-plugin fixes
- preform grib2 numbering encode/decode moved to grib-plugin
* Thu Jun  8 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.8-1.fmi
- More fine-tuning of visibility
- Fix for gust/meps
- Write radon metadata firectly to table partition
* Fri Jun  2 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.2-1.fmi
- Fine-tuning visibility
* Thu Jun  1 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.6.1-1.fmi
- Per-station limits for probability
- hybrid_pressure fix for ENS data
* Wed May 31 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.31-1.fmi
- Fix for CAPE500m starting values
* Tue May 30 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.30-1.fmi
- Tweak to visibility
- Support for ENS per-member calculations
- Fix for MUCIN
* Tue May 23 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.23-1.fmi
- Update to visibility
* Tue May 16 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.16-1.fmi
- dewpoint produces now TD-K
* Mon May 15 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.15-1.fmi
- Updated CSV reading and writing
- Additions to luatool
* Tue Apr 18 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.18-1.fmi
- pop plugin: allow previous ECMWF forecast to be missing
- Remove crash when listing plugins and database password was not set
* Tue Apr 11 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.11-1.fmi
- Cape plugin fix for GPU code bug
* Thu Apr  6 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.6-2.fmi
- Hotfix for database fetch issue
* Thu Apr  6 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.6-1.fmi
- Further safeguards for fractile & NaNs 
- Add nodatabase mode
- New newbase
- New fmigrib
- New fmidb
- New eccodes
* Tue Apr  4 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.4-1.fmi
- fractile another NaN fix
* Mon Apr  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.3-1.fmi
- gust Hirlam support
- cape fix EL level search
- fractile fix NaN issues
* Thu Mar 29 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.30-1.fmi
- Hotfix for fetcher database access
* Wed Mar 29 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.29-1.fmi
- Hotfix for sticky param cache, code refactoring
* Mon Mar 27 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.27-1.fmi
- Lagged ensemble support for luatool
- Sticky param cache for fetcher
* Wed Mar 15 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.15-1.fmi
- General cleanup
- MUCAPE starting value fix 
* Tue Mar  7 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.3.7-1.fmi
- Improved performance of cache read
- Additions to split_sum
* Thu Feb 23 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.23-1.fmi
- Revised gust
* Tue Feb 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.21-1.fmi
- grib plugin refactoring
- Removed roughness plugin
* Fri Feb 17 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.17-1.fmi
- raw_time changes in himan-lib
* Mon Feb 13 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.13-1.fmi
- New newbase
- time_lagged ensemble
- Improved aux file read
- Leap year fix for raw_time
- CAPE MoistLift optimization
- Apply scale+base when writing querydata
- Added plugin documentation
* Mon Jan 30 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.30-1.fmi
- Dependency to open sourced Newbase
* Tue Jan 17 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.17-1.fmi
- VerticalHeightGreaterThan/LessThan added to hitool 
  and luatool
- Ground level type fixes for frost lua
* Thu Jan 12 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.12-2.fmi
- Another bugfix 
* Thu Jan 12 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.12-1.fmi
- Bugfix for preform_hybrid (STU-4906)
* Tue Jan  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.3-1.fmi
- Standard deviation for fractile plugin
* Thu Dec 29 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.29-1.fmi
- Reworked vector component rotation
* Tue Dec 20 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.20-1.fmi
- Hotfix for querydata plugin crashhh
* Mon Dec 19 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.19-2.fmi
- Map grib1 level 103 to mean sea
* Mon Dec 19 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.19-1.fmi
- Three hour snow accumulation (split_sum)
* Thu Dec 15 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.15-1.fmi
- Min2D exposure to luatool
* Fri Dec  9 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.9-1.fmi
- SLES accomodations
- fmidb API change
* Wed Dec  7 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.7-2.fmi
- Cuda 8
* Wed Dec  7 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.7-1.fmi
- ensemble in luatool
* Tue Nov 22 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.22-1.fmi
- fractile and probability mising value changes
- fixing non-cuda build for windvector
* Thu Nov 10 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.10-1.fmi
- Fractile bugfix
* Tue Nov  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.8-2.fmi
- hybrid_height bugfix
* Tue Nov  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.8-1.fmi
- MEPS compatibility
* Tue Nov  1 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.1-1.fmi
- Custom fractiles
* Thu Oct 27 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.27-1.fmi
- Hotfix for radon query problem
* Wed Oct 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.26-1.fmi
- Introducing time_ensemble for fractile
- radon supports level_value2
* Mon Oct 24 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.24-2.fmi
- Update to visibility/Harmonie
* Mon Oct 24 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.24-1.fmi
- SLES compatibility fixes
- General code cleanup
* Thu Oct  6 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.6-1.fmi
- Bugfix for visibility
* Wed Oct  5 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.5-1.fmi
- Preform hybrid v2.7.2
- Support grib index reading
* Tue Oct  4 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.4-1.fmi
- Visibility v1.2.1
- Unstagger cuda version
* Thu Sep 29 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.29-1.fmi
- Visibility v1.2
- Harmonie mixing ratios in correct name and unit (kg/kg)
* Wed Sep 28 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.28-1.fmi
- Ensemble mean calculation
- Different CAPE&CIN variations are separated with levels (not par ids)
* Thu Sep 15 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.15-1.fmi
- New sums for split_sum
* Mon Sep 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.9.12-1.fmi
- Fix in visibility
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
- Fixes to pot
- Renaming si to cape
* Fri Jul  1 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.7.1-1.fmi
- More changes to pop
* Thu Jun 30 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.30-1.fmi
- Changes to pop
* Mon Jun 27 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.27-1.fmi
- New plugin probability
- New plugin pop
- Changes si (metutil)
* Thu Jun 23 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.23-1.fmi
- Update to pot
* Mon Jun 20 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.20-1.fmi
- Change si to use smarttool thetae function
* Thu Jun 16 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.16-1.fmi
- Minor fixes to radon, luatool, fractile
* Mon Jun 13 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.13-1.fmi
- Fixes to forecast type in several plugins
* Thu Jun  9 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.9-1.fmi
- New plugin fractile
- Cache support for ensemble members
* Mon Jun  6 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.6-1.fmi
- New grib_api
- New pot
* Thu May 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.26-1.fmi
- LCL,LFC,EL metric height (HIMAN-123)
- Change in fmidb headers
* Tue May 17 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.17-1.fmi
- Another fix for visibility
* Mon May 16 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.16-1.fmi
- Fix for visibility
* Thu May 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.12-1.fmi
- Hirlam support for visibility
- New newbase
* Mon May  2 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.5.2-1.fmi
- Cuda for si-plugin
* Wed Apr 27 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.27-1.fmi
- New release
* Tue Apr 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.26-1.fmi
- Cuda version of si-plugin (partial support)
* Thu Apr 21 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.21-1.fmi
- Remove swapping code from querydata-plugin (HIMAN-120)
* Wed Apr 20 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.20-1.fmi
- Visibility fixes
* Tue Apr 19 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.19-1.fmi
- Simplified interpolation/Swap code (HIMAN-120)
* Fri Apr 15 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.15-1.fmi
- New release
* Thu Apr 14 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.14-2.fmi
- ThetaE change
- Filter2D changed namespaces
* Thu Apr 14 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.14-1.fmi
- Enabling debug symbols for release builds
* Tue Apr 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.12-1.fmi
- Initial import of visibility
* Mon Apr 11 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.11-1.fmi
- New release
* Fri Apr  8 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.8-1.fmi
- Simplified auxiliary plugin includes
- Removed race condition from si-plugin
* Thu Apr  7 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.7-1.fmi
- Do not use fast_math at si by default
* Tue Apr  5 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.5-1.fmi
- Bugfix for gust
* Mon Apr  4 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.4.4-1.fmi
- Totally new gust plugin
- Seaicing is now index instead of raw value
- Smoothening for CAPE+CIN
* Thu Mar 17 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.3.17-1.fmi
- SW radiation support for split_sum
- Fix area issue with helmi
* Tue Feb 23 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.23-1.fmi
- Fix si for forecast step=0
- Do not write empty grids at hybrid_pressure
* Mon Feb 22 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.22-1.fmi
- Changes to si
* Thu Feb 18 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.18-1.fmi
- New release
* Tue Feb 16 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.16-1.fmi
- Changes to si
* Fri Feb 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.12-1.fmi
- Changes to si
- New newbase and fmidb
* Wed Feb  3 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.3-1.fmi
- Fix split_sum crash
* Mon Feb  1 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.1-1.fmi
- Adding support for potential precipitation form
- split_sum performance improvements
* Mon Jan 18 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.18-1.fmi
- Performance improvements for hybrid_height
* Thu Jan 14 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.14-1.fmi
- Changes in turbulence
* Mon Jan 11 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.11-1.fmi
- Allowing mu Cape to be zero
* Tue Jan  5 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.5-1.fmi
- Support for dynamic memory allocation
- Simplified grib writing code
* Mon Jan  4 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.1.4-1.fmi
- Fixes in grib metadata creation for accumulated parameters and Harmonie
- Fix for relative_humidity
* Mon Dec 21 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.22-1.fmi
- relative_humidity-optimizations
* Mon Dec 21 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.21-1.fmi
- si-optimizations
* Thu Dec 17 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.17-1.fmi
- Free memory after plugin is finished
- Select source data for si-plugin
* Tue Dec 15 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.15-1.fmi
- Fix for POT
* Wed Dec  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.9-1.fmi
- Fix lnsp issue with analysis time
* Tue Dec  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.8-1.fmi
- Optimizations for hybrid_pressure and hybrid_height
* Fri Nov 20 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.25-2.fmi
- Removing debug code from preform_hybrid
* Fri Nov 20 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.25-1.fmi
- Changes in himan-lib
* Fri Nov 20 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.20-1.fmi
- Changes in himan-lib
* Fri Nov 13 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.13-1.fmi
- Friday hotfix for split_sum: all-missing grids were written to db due to mismatching virtual function signature
* Tue Nov 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.10-1.fmi
- Minor fixes to radon plugin
- Change of TI2 formula
* Fri Nov  6 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.6-1.fmi
- Add limit to cache
* Mon Nov  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.2-1.fmi
- Fix to work distribution when primary dimension is time
- Minor changes to hybrid_height
* Fri Oct 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.30-1.fmi
- Performance optimization for hybrid_height
* Thu Oct 29 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.29-1.fmi
- Bugfix for luatool (SOL-3219)
* Mon Oct 26 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.26-1.fmi
- Fix POT
* Wed Oct 14 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.14-1.fmi
- Remove mandatory jpeg packing of grib2 files
* Fri Oct  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.9-1.fmi
- Fix writing of grib in radon-only environment
- Use radondb libary to fetch geom info
- Fix grib plugin to use correct database for fetching data
- Remove redundant luatool parameter number fetching function
* Thu Oct  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.10.8-1.fmi
- tke completed
- hybrid_pressure changed to use T not Q
* Wed Sep 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.30-1.fmi
- Cuda 7.5
* Mon Sep 28 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.28-1.fmi
- Initial version of tke plugin
* Wed Sep  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.9-1.fmi
- neons and radon to luatool
* Tue Sep  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.8-1.fmi
- Support passing options to writer plugins
* Wed Sep  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.2-1.fmi
- fmidb api change
- grib_api 1.14
* Mon Aug 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.24-1.fmi
- Augmenting interpolation methods
- Add support for setting specific missing value to grib
* Mon Aug 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.10-1.fmi
- Adding native interpolation methods
* Mon Jun 22 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.6.22-2.fmi
- Performance improvements in hybrid_height and tpot
* Mon Jun 22 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.6.22-1.fmi
- Fix to cache
* Tue Jun 16 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.6.16-1.fmi
- Fix to relative_humidity
* Mon Jun  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.6.8-1.fmi
- RNETLW-WM2 to split_sum
* Wed Jun  3 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.6.3-1.fmi
- Change formula for tpot/thetaw
- Other minor changes
* Wed May 27 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.27-1.fmi
- New plugin: pot (HIMAN-100)
- Change tpot to use new formula for theta e
* Mon May 11 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.11-1.fmi
- Link with cuda 6.5 due to performance issues (HIMAN-96)
* Tue Apr 28 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.28-1.fmi
- Linking with Cuda 7 (HIMAN-96)
* Mon Apr 27 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.27-1.fmi
- sql fix for radon 
* Fri Apr 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.24-1.fmi
- Linking with newer fmigrib and fmidb
* Mon Apr 13 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.13-1.fmi
- Minor fix to preform_hybrid
* Fri Apr 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.10-1.fmi
- Link with boost 1.57 and dynamic version of newbase
* Thu Apr  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.9-1.fmi
- Bugfix in sequential job distribution (hybrid_height)
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-3.fmi
- Bugfix in job distribution
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-2.fmi
- Bugfix for radon insert sql
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-1.fmi
- Major update to add forecast type based calculations
- Land-sea mask for fetcher&transformer
* Thu Apr  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.2-1.fmi
- Bugix to luatool
- preform_pressure algorithm update 
* Mon Mar 30 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.3.30-1.fmi
- Bugfix to turbulence
* Mon Mar 16 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.3.16-1.fmi
- Bugfix to preform_hybrid
* Wed Mar 11 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.3.11-1.fmi
- Changing cache-plugin to check for existing cache item before inserting
* Mon Mar  9 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.3.9-1.fmi
- Minor changes in preform_pressure and preform_hybrid
* Tue Feb 17 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.17-1.fmi
- Small fix in luatool
- Link with grib_api 1.13.0
* Mon Feb 16 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.16-2.fmi
- Fix in windvector_cuda grid point coordinate handling
* Mon Feb 16 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.16-1.fmi
- Small changes in luatool
* Tue Feb 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.2.10-1.fmi
- Cosmetic changes
* Wed Feb  4 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.4-2.fmi
- Add turbulence
* Tue Feb  3 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.03-2.fmi
- Turbulence plugin
- Bugfix in unstagger
- Changes in NCL
- Changes in CSV
- Changes in querydata
* Mon Jan 26 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.26-1.fmi
- CSV plugin
- RHEL7 compatibility
- Other minor fixes
* Wed Jan  7 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.7-2.fmi
- Changes to accomodate radon
* Wed Jan  7 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.7-1.fmi
- Changes to accomodate radon
* Fri Jan  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.2-2.fmi
- Changes in modifier
* Fri Jan  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.2-1.fmi
- Fix for SOL-2166
* Mon Dec 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.29-3.fmi
- Disable radon due to problems
* Mon Dec 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.29-2.fmi
- Updated fmidb
* Mon Dec 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.29-1.fmi
- Changes in luatool, hitool
* Thu Dec 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.18-1.fmi
- Fixes in qnh
- Irregular grid in himan-lib
* Wed Dec 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.17-2.fmi
- Missed linking with odbc-library
* Wed Dec 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.17-1.fmi
- New plugin: qnh
- New plugin: radon
- New plugin: luatool
* Mon Dec  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.8-1.fmi
- Large internal changes in himan-lib
* Mon Dec  1 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.1-1.fmi
- Harmonie support for gust
* Thu Nov 27 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.27-1.fmi
- Fix in gust
* Tue Nov 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.25-1.fmi
- Initial support for cuda grib packing (disabled for now)
- Fixes in hitool
* Tue Nov 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.18-1.fmi
- Fixes in gust
* Thu Nov 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.13-1.fmi
- Enable hybrid level height check in hitool
- Replace double allocation of memory in cuda plugins with memory registration
* Tue Nov 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.10-1.fmi
- Fixes in split_sum, preform_pressure and preform_hybrid
* Tue Nov 04 2014 Andreas Tack <andreas.tack@fmi.fi> - 14.11.4-1.fmi
- Fixes in split_sum and monin_obukhov
* Thu Oct 30 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.30-1.fmi
- Fixes in split_sum and monin_obukhov
* Tue Oct 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.28-1.fmi
- HIMAN-69
* Tue Oct 21 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.21-1.fmi
- Fixes in unstagger
* Mon Oct 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.20-1.fmi
- New plugin unstagger
- Changes in fetcher related above
* Thu Oct 16 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.16-1.fmi
- Icing formula modified (HIMAN-77)
* Tue Oct 14 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.14-1.fmi
- New plugin monin obukhov length
- Fix EC hybrid pressure
* Mon Oct 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.13-1.fmi
- Fix EC snow accumulation unit recognition
- Always use cuda grib unpacking if cuda device is present
* Thu Oct  9 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.9-1.fmi
- Fix in writer and grib-plugins when writing hybrid levels with level value > 127
* Mon Oct  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.6-1.fmi
- Changes in himan-lib
* Tue Sep 30 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.30-1.fmi
- Linking with newer grib_api that possibly fixes HIMAN-58
- Newer version of fmigrib that possibly fixes another grib_api related bug (RPINTAII-39)
* Fri Sep 26 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.26-1.fmi
- Stability cuda version done
- Stability-si calculation disabled by default
- New plugin gust (initial version)
- preform_hybrid does not throw runtime_error if hitool fails
* Thu Sep 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.25-1.fmi
- Limiting EC highest hybrid lever number to 24
- Fix in weather_code_1 / 3h step
* Wed Sep 24 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.24-1.fmi
- EC support for weather_code_1
- Fixes in hitool::Stratus()
* Tue Sep 23 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.23-1.fmi
- Adding wind bulk shear to stability
- New SSICING-N parameter for seaicing
- Fixed hybrid_height for ECMWF
- Pre-allocate memory for simple_packed::Unpack()
- New interpolation method (NFmiQueryInfo::InterpolatedValue())
- Improvements in querydata plugins
- Improvements in hitool (additional overloads)
- Using HPStringToLevelType at transformer
* Thu Aug 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.28-1.fmi
- Fix for HIMAN-62
* Fri Aug 22 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.22-1.fmi
- Hotfix for GFS/GEM longitude coordinate issue
* Tue Aug 12 2014  Mikko Partio <mikko.partio@fmi.fi> - 14.8.12-1.fmi
- Fixes to hybrid_height and cache
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-2.fmi
- Logging fix on windvector
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-1.fmi
- Misc fixes WRT ECMWF
- Removed pcuda plugin
- Renamed fmi_weather_symbol_1 to weather_code_2
- Rename rain_type to weather_code_1
- Renamed cloud_type to cloud_code
- Cuda support for transformer, stability
- "Developer friendly" plugins
* Mon Jun 23 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.23-1.fmi
- Fixes in transformer
* Wed Jun 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.18-1.fmi
- New plugin: transformer (HIMAN-37)
- Initial build with Cuda6 (HIMAN-57)
* Thu Jun  5 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.5-1.fmi
- New plugin: fmi_weather_symbol_1 (HIMAN-52)
- Bugfix in fog/ECMWF (STU-1366)
- Bugfix in preform_pressure (HIMAN-55)
* Fri May 16 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.16-1.fmi
- Bugfix in cloud_type/ECMWF
* Tue May 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.13-2.fmi
- Bugfix in rain_type/ECMWF
* Tue May 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.13-1.fmi
- Bugfix in cloud_type/ECMWF
* Fri May  9 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.9-1.fmi
- Bugfix in grib-plugin
- Bugfix in stability-plugin/LI
* Wed May  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.7-2.fmi
- Bugfix in split_sum
* Wed May  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.7-1.fmi
- Cuda-enabled relative_humidity
- Improved error logging in pcuda
* Tue May  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.6-1.fmi
- weather_symbol finished (HIMAN-36)
- kindex renamed to stability (HIMAN-49)
- Changing shared_ptr<T> to const shared_ptr<T>& in many places (HIMAN-50)
- Fix subtle bug with grib bitmap unpacking
- Initial version of relative_humidity cuda calculation (not enabled yet)
* Mon Apr 14 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.14-1.fmi
- New plugin: weather_symbol (beta) (HIMAN-36)
- Changes to preform_pressure (HIMAN-48)
* Thu Apr 10 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.10-1.fmi
- Add parameter unit to dewpoint and relative_humidity
* Mon Apr  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.7-2.fmi
- Link against smarttools
* Mon Apr  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.7-1.fmi
- Three new plugins: si, roughness and absolute_humidity
- Parameter change in precipitation_rate
* Wed Mar 26 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.26-1.fmi
- Vertical coordinate fix for density and precipitation_rate
* Thu Mar 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.20-1.fmi
- New plugin: precipitation_rate
* Tue Mar 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.18-1.fmi
- Minor fixes
* Mon Mar 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.17-1.fmi
- Bugfix in compiled_plugin_base::Unpack() (VALVONTA-112)
- Multi-param overloads for hitool
- Minor bugfixes
* Wed Mar 12 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.12-1.fmi
- New plugin: density
- Bugfix for preform_hybrid
- Attemping to fix HIMAN-25
* Fri Feb 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.28-3.fmi
- Support EC ground surface parameter PGR-PA for preform_pressure
* Fri Feb 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.28-2.fmi
- Fix HIMAN-22
* Fri Feb 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.28-1.fmi
- fetcher: do level transform separately for each producer (of more than one present)
* Tue Feb 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.25-2.fmi
- Bugfix for tpot/hybrid (OPER-494)
* Tue Feb 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.25-1.fmi
- Add flag for separating cuda-enabled plugins
- Fixes for tpot@cuda
- Fixes for preform_hybrid
- Add automatic level conversion to fetcher
- Bugfix and optimization for dewpoint
* Tue Feb 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.18-1.fmi
- Fix for HIMAN-21
* Mon Feb 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.17-1.fmi
- Bugfix for preform_pressure (HIMAN-38)
- Add functionality to querydata
* Thu Feb 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.13-1.fmi
- Bugfix for preform_hybrid and hitool
* Tue Feb 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.11-1.fmi
- Bugfix for preform_hybrid
- tpot results in correct unit (kelvin)
- Change split_sum calculation logic when calculating for Harmonie
- BIG change in cuda calculation
* Mon Feb  3 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.3-1.fmi
- Bugfix for preform_pressure
- Add support for solid precipitation and graupel for split_sum
* Mon Jan 27 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.27-1.fmi
- Fix for relative humidity pressure scaling 
- Bugfix for preform_pressure
- Some changes in ncl
* Mon Jan 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.20-2.fmi
- Add millimeter-support to vvms
* Mon Jan 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.20-1.fmi
- Fixes in rain_type
- New formula for hirlam in hybrid_height
- Fixes in ncl
- New formula for preform_pressure and preform_hybrid
- Fix in grib when reading polster
- Tuning for fmidb (HIMAN-17)
* Tue Jan  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.7-1.fmi
- Change in split_sum rate calculation (HIMAN-26)
- Lots of internal changes related to boilerplate code removal (HIMAN-28) 
- Final touches on hitool and preform_hybrid (HIMAN-27)
- Fast mode for hybrid_height (HIMAN-20)
- Link with grib_api 1.11.0
* Wed Dec 11 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.12.11-1.fmi
- Changes in pcuda, hitool and preform_hybrid
* Mon Nov 25 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.25-1.fmi
- Fixes related to upcoming scandinavia area Harmonie
* Wed Nov 13 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.13-1.fmi
- Source parameter fixes for tpot and icing
* Tue Nov 12 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.12-1.fmi
- New plugin preform_hybrid (not finished yet)
- Rename plugin 'precipitation' to 'split_sum'
- Fix for split_sum ground level determination
- Other bugfixes
* Mon Oct 14 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.14-1.fmi
- Leveltype fixes in hybrid_pressure and precipitation
- Add hitool-plugin
* Wed Oct  9 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.9-1.fmi
- Support kurkuma.fmi.fi
* Tue Oct  8 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.8-1.fmi
- relative_humidity plugin
* Wed Oct  2 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.2-1.fmi
- Fix for preform_pressure
* Thu Sep 26 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.26-1.fmi
- Fix for windvector hybrid level handling
* Wed Sep 25 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.25-1.fmi
- Another fix for HIMAN-16
* Tue Sep 24 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.24-1.fmi
- Fix for HIMAN-16
* Mon Sep 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.23-1.fmi
- Fix for HIMAN-15
* Thu Sep  5 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.5-1.fmi
- New plugin: preform_pressure
* Tue Sep  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.3-2.fmi
- Attempt to fix HIMAN-14
* Tue Sep  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.3-1.fmi
- Fix bug that crashes himan (unresolved cuda-symbols at excutable)
* Fri Aug 30 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.30-1.fmi
- Latest changes
- First compilation on scout.fmi.fi
* Thu Aug 22 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.22-1.fmi
- Latest changes
* Wed Aug 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.21-1.fmi
- Latest changes
- Linking with new version of fmigrib to avoid grib_api bug crashing the program 
  (SUP-592 @ http://software.ecmwf.int)
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
* Thu Jan 24 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.24-1.fmi
- One new plugin + other changes
* Wed Jan 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.23-1.fmi
- Two new plugins
- Use debug build
* Mon Jan 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.21-1.fmi
- Bugfixes
* Tue Jan 15 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.15-1.fmi
- First attempt for production-ready release
* Thu Dec 27 2012 Mikko Partio <mikko.partio@fmi.fi> - 12.12.27-1.fmi
- Initial build
