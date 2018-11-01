# Quick setup; tested only on Ubuntu 16.04.
# Get the prerequisite utilities to download and unzip and compile if you don't already have them
sudo apt-get install wget zip clang
# Get terra
wget https://github.com/zdevito/terra/releases/download/release-2016-03-25/terra-Linux-x86_64-332a506.zip
unzip terra-Linux-x86_64-332a506.zip
mv terra-Linux-x86_64-332a506 ../terra
rm rm terra-Linux-x86_64-332a506.zip
# Test terra
cd ../terra/share/terra/tests
../../../bin/terra run
cd ../../../../Opt

