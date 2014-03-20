%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define LIBNAME himan-plugins
Summary: himan-plugins library
Name: %{LIBNAME}
Version: 14.3.20
Release: 1.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: jasper-libs
Requires: grib_api >= 1.11.0
Requires: oracle-instantclient-basic >= 11.2.0.3.0
BuildRequires: boost-devel >= 1.54
BuildRequires: scons
BuildRequires: libsmartmet-newbase >= 13.9.26
BuildRequires: grib_api-devel >= 1.10.4
BuildRequires: redhat-rpm-config
BuildRequires: cuda-5-5
BuildRequires: oracle-instantclient-devel >= 11.2.0.3.0

%if %{distnum} == 5
BuildRequires: gcc44-c++ >= 4.4.6
BuildRequires: gcc44-c++ < 4.7
%else
BuildRequires: gcc-c++ >= 4.4.6
BuildRequires: gcc-c++ < 4.7
%endif

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
%{_libdir}/himan-plugins/libcache.so
%{_libdir}/himan-plugins/libcloud_type.so
%{_libdir}/himan-plugins/libdensity.so
%{_libdir}/himan-plugins/libdewpoint.so
%{_libdir}/himan-plugins/libfetcher.so
%{_libdir}/himan-plugins/libfog.so
%{_libdir}/himan-plugins/libgrib.so
%{_libdir}/himan-plugins/libhitool.so
%{_libdir}/himan-plugins/libhybrid_height.so
%{_libdir}/himan-plugins/libhybrid_pressure.so
%{_libdir}/himan-plugins/libicing.so
%{_libdir}/himan-plugins/libkindex.so
%{_libdir}/himan-plugins/libneons.so
%{_libdir}/himan-plugins/libncl.so
%{_libdir}/himan-plugins/libpcuda.so
%{_libdir}/himan-plugins/libprecipitation_rate.so
%{_libdir}/himan-plugins/libpreform_hybrid.so
%{_libdir}/himan-plugins/libpreform_pressure.so
%{_libdir}/himan-plugins/libquerydata.so
%{_libdir}/himan-plugins/librain_type.so
%{_libdir}/himan-plugins/librelative_humidity.so
%{_libdir}/himan-plugins/libseaicing.so
%{_libdir}/himan-plugins/libsplit_sum.so
%{_libdir}/himan-plugins/libtk2tc.so
%{_libdir}/himan-plugins/libtpot.so
%{_libdir}/himan-plugins/libvvms.so
%{_libdir}/himan-plugins/libwindvector.so
%{_libdir}/himan-plugins/libwriter.so

%changelog
* Thu Mar 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.20-1.fmi
- New plugin: precipitation_rate
* Mon Mar 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.18-1.fmi
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
