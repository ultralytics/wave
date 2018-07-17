#!/bin/bash
# Run this script on a brand new GCP VM. Recommend Ubuntu 18.04 LTS, 15 GB Persistent SSD, P100 GPU, Skylake CPU.
sudo apt update -y
sudo apt autoremove -y

# Install Drive FUSE wrapper. https://github.com/astrada/google-drive-ocamlfuse
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:alessandro-strada/ppa
sudo apt update -y
sudo apt install -y google-drive-ocamlfuse

# Generate creds for the Drive FUSE library.
google-drive-ocamlfuse -headless -id 280545512213-gaaccol136ti4bt46g1imip3is7h10ph.apps.googleusercontent.com -secret l1aNm3ecGgKOit3vQ_TwBhNM
#Create a directory and mount Google Drive using that directory.
mkdir drive
google-drive-ocamlfuse drive
#fusermount -u drive  # unmount

# Install Linux Programs
sudo apt install -y git unzip python3-pip screen vim

# Install Python Packages
pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
pip3 install -U numpy scipy #tensorflow # plotly  # wave
pip3 install -U opencv-python exifread tqdm # bokeh  # velocity

# GPU driver install P100 and K80
#sudo apt install ubuntu-drivers-common -y
# sudo ubuntu-drivers autoinstall

# GPU driver install on *Ubuntu 16.04 LTS* from https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
# The 16.04 installer works with 16.10.
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get install cuda-9-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1

# Shutdown
sudo apt autoremove -y
rm -rf vminstall.bash
sudo shutdown now

# Extras
# find MATLAB/ -size +100M -ls
