%define LIBNAME himan-plugins
Summary: himan-plugins library
Name: %{LIBNAME}
Version: 13.1.14
Release: 1.el6.fmi
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
BuildRequires: gcc-c++ >= 4.4.6
BuildRequires: scons
Provides: libcache.so

%description
FMI himan-plugins -- hila manipulaatio -- plugin library

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-plugins" 

%build
make %{_smp_mflags} 

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0775)
%{_libdir}/himan-plugins/libcache.so
%{_libdir}/himan-plugins/libfetcher.so
%{_libdir}/himan-plugins/libgrib.so
%{_libdir}/himan-plugins/libicing.so
%{_libdir}/himan-plugins/libneons.so
%{_libdir}/himan-plugins/libpcuda.so
%{_libdir}/himan-plugins/libquerydata.so
%{_libdir}/himan-plugins/libtk2tc.so
%{_libdir}/himan-plugins/libtpot.so
%{_libdir}/himan-plugins/libvvms.so
%{_libdir}/himan-plugins/libwriter.so

%changelog
* Wed Jan 14 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.14-1.el6.fmi
- Latest release
* Thu Dec 27 2012 Mikko Partio <mikko.partio@fmi.fi> - 12.12.27-1.el6.fmi
- Initial build
