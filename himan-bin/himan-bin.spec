%define distnum %(/usr/lib/rpm/redhat/dist.sh --distnum)

%define BINNAME himan-bin
Summary: himan executable
Name: %{BINNAME}
Version: 19.10.25
Release: 1%{dist}.fmi
License: MIT
Group: Development/Tools
URL: http://www.fmi.fi
Source0: %{name}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot-%(%{__id_u} -n)
Requires: glibc
Requires: libgcc
Requires: libstdc++
Requires: himan-lib >= 19.5.6
Requires: himan-plugins
Requires: gdal >= 1.11.0
Requires: zlib
Requires: libfmidb
BuildRequires: boost-devel >= 1.66

%if %{defined suse_version}
Requires: libjasper
%else
BuildRequires: redhat-rpm-config
BuildRequires: gcc-c++ >= 4.8.2
BuildRequires: cuda-9-1
Requires: jasper
Requires: boost-program-options
Requires: boost-system
Requires: boost-regex
Requires: boost-iostreams
Requires: boost-thread
Requires: bzip2-libs

%endif
BuildRequires: scons
Provides: himan

AutoReqProv:	no

%description
Himan -- hilojen manipulaatio -- executable

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
* Fri Oct 25 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.10.25-1.fmi
- Remove options -R and -N
* Mon May  6 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.5.6-1.fmi
- Add class time_duration
* Tue Feb 19 2019 Mikko Partio <mikko.partio@fmi.fi> - 19.2.19-1.fmi
- configuration api change
* Wed Nov  7 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.11.7-1.fmi
- More timing output formatting improvements
* Thu Nov  1 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.11.1-1.fmi
- Timing output formatting improvements
- Remove HPVersionNumber
* Tue Oct 16 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.10.16-1.fmi
- Improved statistics upload
* Mon Oct 15 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.10.15-1.fmi
- ss_state table changes
* Wed Oct 10 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.10.10-1.fmi
- Refactoring plugin_configuration
* Fri Oct  5 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.10.5-1.fmi
- Fix for invalid statistics json
* Tue Aug  7 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.8.7-1.fmi
- RHEL7.5 build
* Wed May  2 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.5.2-1.fmi
- Stricter compiler warning flags
* Mon Apr 16 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.4.16-1.fmi
- Built with cuda 9.1
* Tue Apr 10 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.4.10-1.fmi
- New boost
* Mon Mar 26 2018 Mikko Partio <mikko.partio@fmi.fi> - 18.3.26-1.fmi
- Support for uploading run statistics to database
* Mon Dec  4 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.12.4-1.fmi
- Support for ss_state table updates
* Mon Sep 25 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.9.25-1.fmi
- Remove Oracle support
* Tue Sep 12 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.9.12-1.fmi
- Add stack trace functionality
* Mon Sep 11 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.9.11-1.fmi
- Replace kFloatMissing with nan
* Tue Aug 29 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.29-1.fmi
- boost 1.65
* Mon Aug 21 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.21-1.fmi
- General code cleanup
* Mon Aug 14 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.14-1.fmi
* json_parser interface change
* Tue Aug  8 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.8-1.fmi
- Changing cmd line option -s argument optional
* Thu Aug  3 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.3-1.fmi
- Disabling cuda interpolation if device is missing
* Tue Aug  1 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.8.1-1.fmi
- Removing logger_factory
* Mon Jul 17 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.7.17-1.fmi
- Removing timer_factory
* Tue May 23 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.23-1.fmi
- New release
* Mon May 15 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.5.15-1.fmi
- New release
* Tue Apr 18 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.18-1.fmi
- Remove cuda cmd line options if cuda libraries are not present
  at build time
* Thu Apr  6 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.4.6-1.fmi
- Add nodatabase mode
* Mon Feb 13 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.2.13-1.fmi
- New newbase
- time_lagged ensemble
- Improved aux file read
- Leap year fix for raw_time
- CAPE MoistLift optimization
- Apply scale+base when writing querydata
* Mon Jan 30 2017 Mikko Partio <mikko.partio@fmi.fi> - 17.1.30-1.fmi
- Remove dependency to Newbase
* Wed Dec  7 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.12.7-1.fmi
- Cuda 8
* Tue Nov 22 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.22-1.fmi
- New release
* Tue Nov  1 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.11.1-1.fmi
- New release
* Thu Oct 27 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.27-1.fmi
- New release
* Wed Oct 26 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.26-1.fmi
- New release
* Mon Oct 24 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.24-1.fmi
- General code cleanup
* Thu Oct  6 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.10.6-1.fmi
- New release
* Tue Aug 23 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.8.23-1.fmi
- New release
* Mon Jun  6 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.6.6-1.fmi
- New release
* Thu Feb 18 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.18-1.fmi
- Remove dependency to smarttools
* Fri Feb 12 2016 Mikko Partio <mikko.partio@fmi.fi> - 16.2.12-1.fmi
- New newbase
* Mon Dec 21 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.12.21-1.fmi
- himan lib API change
* Thu Nov 19 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.11.19-1.fmi
- Remove calculated data from memory once plugin is finished
* Wed Sep 30 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.30-1.fmi
- Cuda 7.5
* Tue Sep  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.8-1.fmi
- plugin inheritance hierarchy modifications
* Wed Sep  2 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.9.2-1.fmi
- grib_api 1.14
* Mon Aug 24 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.8.24-1.fmi
- New command line option --no-cuda-interpolation
* Wed May 27 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.27-1.fmi
- New command line option --compression,-c 
* Mon May 11 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.5.11-1.fmi
- Link with cuda 6.5 due to performance issues (HIMAN-96)
* Tue Apr 28 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.28-1.fmi
- Linking with Cuda 7 (HIMAN-96)
* Fri Apr 10 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.10-1.fmi
- Link with boost 1.57 and dynamic version of newbase
* Wed Apr  8 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.4.8-1.fmi
- New release
* Mon Mar 30 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.3.30-1.fmi
- New release
* Tue Feb 10 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.10-1.fmi
- Changes in himan-lib headers
* Tue Feb  3 2015 Andreas Tack <andreas.tack@fmi.fi> - 15.2.03-1.fmi
- Bugfix in unstagger
- Changes in querydata
- Changes in CSV
- Changes in NCL
* Mon Jan 26 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.26-1.fmi
- CSV plugin
- RHEL7 compatibility
* Wed Jan  7 2015 Mikko Partio <mikko.partio@fmi.fi> - 15.1.7-1.fmi
- New options -N and -R
* Wed Dec 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.17-1.fmi
- Changes in himan-lib headers
* Mon Dec  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.8-1.fmi
- Large internal changes in himan-lib
* Mon Dec  1 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.12.1-1.fmi
- Changes in himan-lib headers
* Tue Nov 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.25-1.fmi
- Initial support for enabling cuda packing/unpacking with separate switches
* Thu Nov 13 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.13-1.fmi
- Changes in himan-lib headers
* Tue Nov 04 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.11.4-1.fmi
- Changes in himan-lib headers
* Thu Oct 30 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.30-1.fmi
- Changes in himan-lib headers
* Tue Oct 28 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.28-1.fmi
- Changes in himan-lib headers
* Mon Oct 20 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.20-1.fmi
- Changes in himan-lib headers
* Mon Oct  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.10.6-1.fmi
- Changes in himan-lib headers
* Wed Sep 24 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.24-1.fmi
- Changes in himan-lib headers
* Tue Sep 23 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.23-1.fmi
- Changes in himan-lib headers
* Mon Sep  8 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.9.8-1.fmi
- Compling and linkin with -pie (HIMAN-51)
* Tue Aug 12 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.12-1.fmi
- Backwards compatibility for renamed plugins
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-2.fmi
- Fix bug that forced himan to an eternal loop
* Mon Aug 11 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.8.11-1.fmi
- Add overall timings on himan execution
- Add cuda functionality since plugin pcuda is removed
* Wed Jun 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.18-1.fmi
- Initial build with Cuda6 (HIMAN-57)
* Thu Jun  5 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.6.5-1.fmi
- Changes in himan-lib
* Tue May  6 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.5.6-1.fmi
- Changes in himan-lib
* Tue Apr 29 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.29-1.fmi
- Extra support for renamed plugin 'kindex' (new name 'stability')
* Mon Apr  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.4.7-1.fmi
- Changes in himan-lib
* Tue Mar 18 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.18-1.fmi
- Changes in himan-lib
* Mon Mar 17 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.3.17-1.fmi
- Small change in himan-lib/configuration
* Wed Mar 12 2014  Mikko Partio <mikko.partio@fmi.fi> - 14.3.12-1.fmi
- Updated configuration file options (source_geom_name, file_type)
* Tue Feb 25 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.2.25-1.fmi
- Add support for setting cuda device id
* Tue Jan  7 2014 Mikko Partio <mikko.partio@fmi.fi> - 14.1.7-1.fmi
- Changes in himan-lib
* Wed Nov 13 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.13-1.fmi
- Fix crashing himan caused by missing cudar and clntsh libraries at linking stage
* Tue Nov 12 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.11.12-1.fmi
- Extra support for renamed plugin 'precipitation' (new name 'split_sum')
* Wed Oct  2 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.10.2-1.fmi
- Changes in himan-lib
* Mon Sep 23 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.23-1.fmi
- Changes in configuration and json_parser
* Tue Sep  3 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.9.3-1.fmi
- Fix bug that crashes himan (unresolved cuda-symbols at excutable)
* Fri Aug 30 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.8.30-1.fmi
- Latest changes
- First compilation on scout.fmi.fi
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
* Mon Jan 14 2013 Mikko Partio <mikko.partio@fmi.fi> - 13.1.14-1.fmi
- Initial build
