%define BINNAME himan-bin
Summary: himan core library
Name: %{BINNAME}
Version: 12.12.27
Release: 1.el6.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: himan-lib
Provides: himan

%description
FMI himan -- hila manipulaatio -- executable

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-bin" 

%build
make %{_smp_mflags} 

%install
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0775)
%{_bindir}/himan

%changelog
* Thu Dec 27 2012 Mikko Partio <mikko.partio@fmi.fi> - 12.12.27-1.el6.fmi
- Initial build
