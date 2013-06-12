%define is_rhel_5 %(grep release /etc/redhat-release | cut -d ' ' -f 3 | head -c 1 | grep 5 || echo 0)
%define is_rhel_6 %(grep release /etc/redhat-release | cut -d ' ' -f 3 | head -c 1 | grep 6 || echo 0)

%define dist el5

%if %is_rhel_6
%define dist el6
%endif

%define BINNAME himan-bin
Summary: himan executable
Name: %{BINNAME}
Version: 13.3.11
Release: 1.%{dist}.fmi
License: FMI
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: himan-lib

%if is_rhel_5
BuildRequires: gcc44-c++ >= 4.4.6
BuildRequires: gcc44-c++ < 4.7
%else
BuildRequires: gcc-c++ >= 4.4.6
BuildRequires: gcc-c++ < 4.7
%endif

BuildRequires: scons
BuildRequires: boost-devel
BuildRequires: libsmartmet-newbase >= 12.4.18-1

Provides: himan

%description
FMI himan -- hila manipulaatio -- executable

%prep
rm -rf $RPM_BUILD_ROOT

%setup -q -n "himan-bin" 

%build
make debug

%install
mkdir -p $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0775)
%{_bindir}/himan

%changelog
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
