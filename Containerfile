FROM docker.io/rockylinux/rockylinux:8

RUN rpm -ivh https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm \
             https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
    dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf -y module disable postgresql && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y update && \
    dnf -y install \
	himan-bin \
	himan-lib \
	himan-plugins \
        himan-scripts \
	wget jq && \
    dnf -y clean all
