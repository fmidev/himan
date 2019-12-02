%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define LIBNAME himan-scripts
Summary: himan-scripts collection
Name: %{LIBNAME}
Version: 19.11.29
Release: 1%{dist}.fmi
License: MIT
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: lua >= 5.1.4
Requires: himan-plugins

%define debug_package %{nil}

%description
Himan -- hilojen manipulaatio -- scripts collection

%prep
rm -rf $RPM_BUILD_ROOT
%setup -q -n "himan-scripts" 

%build

%install
mkdir -p $RPM_BUILD_ROOT/usr/share/himan-scripts
cp *.lua $RPM_BUILD_ROOT/usr/share/himan-scripts
chmod 644 $RPM_BUILD_ROOT/usr/share/himan-scripts/*

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,0644)
%{_datadir}/himan-scripts/*.lua

%changelog
* Fri Nov 29 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.11.29-1.fmi
- Initial build
