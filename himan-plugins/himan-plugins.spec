%define is_rhel_5 %(grep release /etc/redhat-release | cut -d ' ' -f 3 | head -c 1 | grep 5 || echo 0)
%define is_rhel_6 %(grep release /etc/redhat-release | cut -d ' ' -f 3 | head -c 1 | grep 6 || echo 0)

%define dist el5

%if %is_rhel_6
%define dist el6
%endif

%define LIBNAME himan-plugins
Summary: himan-plugins library
Name: %{LIBNAME}
Version: 13.2.18
Release: 1.%{dist}.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
BuildRequires: libsmartmet-newbase >= 12.4.18-1
BuildRequires: boost-devel
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: jasper-libs
Requires: grib_api
BuildRequires: boost-devel >= 1.49
BuildRequires: scons

%if is_rhel_5
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
make debug

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0644)
%{_libdir}/himan-plugins/libcache.so
%{_libdir}/himan-plugins/libdewpoint.so
%{_libdir}/himan-plugins/libfetcher.so
%{_libdir}/himan-plugins/libgrib.so
%{_libdir}/himan-plugins/libicing.so
%{_libdir}/himan-plugins/libkindex.so
%{_libdir}/himan-plugins/libneons.so
%{_libdir}/himan-plugins/libpcuda.so
%{_libdir}/himan-plugins/libprecipitation.so
%{_libdir}/himan-plugins/libquerydata.so
%{_libdir}/himan-plugins/libseaicing.so
%{_libdir}/himan-plugins/libtk2tc.so
%{_libdir}/himan-plugins/libtpot.so
%{_libdir}/himan-plugins/libvvms.so
%{_libdir}/himan-plugins/libwindvector.so
%{_libdir}/himan-plugins/libwriter.so

%changelog
* Fri Feb 18 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.2.18-1.el5.fmi
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
