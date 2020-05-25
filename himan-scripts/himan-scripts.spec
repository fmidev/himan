%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define LIBNAME himan-scripts
Summary: himan-scripts collection
Name: %{LIBNAME}
Version: 20.5.25
Release: 1%{dist}.fmi
License: MIT
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: lua >= 5.1.4
Requires: himan-plugins >= 20.3.2
Requires: himan-lib >= 20.3.2

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
* Mon May 25 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.5.25-1.fmi
- Add meps-cloudiness.lua
* Wed May 13 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.5.13-1.fmi
- Change to snow-depth/ecmwf
* Thu May  7 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.5.7-1.fmi
- Change to ceiling.lua
* Mon Mar 16 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.3.16-1.fmi
- Minor fixes and enhancements
* Fri Mar  6 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.3.6-1.fmi
- Fix for snow.lua (and rename to snow-depth.lua)
* Mon Mar  2 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.3.2-1.fmi
- Improved lagged ensemble configuration
* Wed Feb 26 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.2.27-1.fmi
- Skip missing values in neighborhood probabilites
* Wed Feb 26 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.2.26-1.fmi
- Fixes to probability filters
* Tue Feb 18 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.2.18-2.fmi
- Fix to LVP.lua
* Tue Feb 18 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.2.18-1.fmi
- Updates to emc.lua and cloutype.lua
* Tue Feb  4 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.2.4-1.fmi
- CB/TCU v1.6
* Wed Jan 29 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.1.29-1.fmi
- Specify forecast type when needed
* Tue Jan 21 2020 Mikko Partio <mikko.partio@fmi.fi> - 20.1.21-1.fmi
- Add nearby-weather.lua
* Mon Dec 30 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.12.30-1.fmi
- Fix to snowload
* Fri Nov 29 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.11.29-1.fmi
- Initial build
