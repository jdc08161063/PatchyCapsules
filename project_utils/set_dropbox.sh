#!/usr/bin/env bash




cd
wget -O dropbox.tar.gz "http://www.dropbox.com/download/?plat=lnx.x86_64"
tar -xvzf dropbox.tar.gz
.dropbox-dist/dropboxd &


#ls -n

ln -s Dropbox/ResearchDepartment/ResearchProjects/GAMMA/Data/ ~/.gamma_link
echo 'export GAMMA_DATA_ROOT="~/.gamma_link/"' >> ~/.bashrc