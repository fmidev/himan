%define LIBNAME himan-lib
Summary: himan core library
Name: %{LIBNAME}
Version: 12.12.27
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
Provides: libhiman.so
Provides: libhiman.a

%description
FMI himan -- hila manipulaatio -- core library

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-lib" 

%build
make %{_smp_mflags} 

%install
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0775)
%{_libdir}/libhiman.so
%{_libdir}/libhiman.a

%changelog
* Thu Dec 27 2012 Mikko Partio <mikko.partio@fmi.fi> - 12.12.27-1.el6.fmi
- Initial build
