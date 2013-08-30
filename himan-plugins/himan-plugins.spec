%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define dist el5

%if %{distnum} == 6
%define dist el6
%endif

%define LIBNAME himan-plugins
Summary: himan-plugins library
Name: %{LIBNAME}
Version: 13.8.22
Release: 1.%{dist}.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: jasper-libs
Requires: grib_api >= 1.10.4
Requires: oracle-instantclient-basic >= 11.2.0.3.0
BuildRequires: boost-devel >= 1.52
BuildRequires: scons
BuildRequires: libsmartmet-newbase >= 12.4.18-1
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
%{_libdir}/himan-plugins/libdewpoint.so
%{_libdir}/himan-plugins/libfetcher.so
%{_libdir}/himan-plugins/libfog.so
%{_libdir}/himan-plugins/libgrib.so
%{_libdir}/himan-plugins/libhybrid_height.so
%{_libdir}/himan-plugins/libhybrid_pressure.so
%{_libdir}/himan-plugins/libicing.so
%{_libdir}/himan-plugins/libkindex.so
%{_libdir}/himan-plugins/libneons.so
%{_libdir}/himan-plugins/libncl.so
%{_libdir}/himan-plugins/libpcuda.so
%{_libdir}/himan-plugins/libprecipitation.so
%{_libdir}/himan-plugins/libquerydata.so
%{_libdir}/himan-plugins/librain_type.so
%{_libdir}/himan-plugins/libseaicing.so
%{_libdir}/himan-plugins/libtk2tc.so
%{_libdir}/himan-plugins/libtpot.so
%{_libdir}/himan-plugins/libvvms.so
%{_libdir}/himan-plugins/libwindvector.so
%{_libdir}/himan-plugins/libwriter.so

%changelog
* Thu Aug 22 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.22-1.el6.fmi
- Latest changes
* Wed Aug 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.21-1.el6.fmi
- Latest changes
- Linking with new version of fmigrib to avoid grib_api bug crashing the program 
  (SUP-592 @ http://software.ecmwf.int)
* Fri Aug 16 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.16-1.el6.fmi
- Latest changes
- First release for masala-cluster
* Mon Mar 11 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.3.11-1.el5.fmi
- Latest changes
* Thu Feb 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.21-1.el5.fmi
- Latest changes
* Mon Feb 18 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.18-1.el5.fmi
- Latest changes
* Tue Feb  5 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.5-1.el5.fmi
- Latest changes
* Thu Jan 31 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.31-1.el5.fmi
- Latest changes
* Thu Jan 24 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.24-1.el5.fmi
- One new plugin + other changes
* Wed Jan 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.23-1.el5.fmi
- Two new plugins
- Use debug build
* Mon Jan 21 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.21-1.el5.fmi
- Bugfixes
* Tue Jan 15 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.15-1.el5.fmi
- First attempt for production-ready release
* Thu Dec 27 2012 Mikko Partio <mikko.partio@fmi.fi> - 12.12.27-1.el6.fmi
- Initial build
