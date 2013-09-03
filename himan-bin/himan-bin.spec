%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define dist el5

%if %{distnum} == 6
%define dist el6
%endif

%define BINNAME himan-bin
Summary: himan executable
Name: %{BINNAME}
Version: 13.9.3
Release: 1.%{dist}.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: himan-lib >= 13.8.16
BuildRequires: redhat-rpm-config

%if %{distnum} == 5
BuildRequires: gcc44-c++ >= 4.4.6
BuildRequires: gcc44-c++ < 4.7
%else
BuildRequires: gcc-c++ >= 4.4.6
BuildRequires: gcc-c++ < 4.7
%endif

BuildRequires: scons
BuildRequires: boost-devel >= 1.52
BuildRequires: libsmartmet-newbase >= 12.4.18-1

Provides: himan

%description
FMI himan -- hila manipulaatio -- executable

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
* Tue Sep  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.3-1.el6.fmi
- Fix bug that crashes himan (unresolved cuda-symbols at excutable)
* Fri Aug 30 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.30-1.el6.fmi
- Latest changes
- First compilation on scout.fmi.fi
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
* Mon Jan 14 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.14-1.el5.fmi
- Initial build
