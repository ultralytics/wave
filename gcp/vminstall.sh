##!/bin/bash
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

sudo apt install -y unzip
sudo apt install -y git
sudo apt install -y python3-pip

# Install Python Packages
pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

pip3 install -U numpy scipy tensorflow # plotly  # wave
pip3 install -U opencv-python exifread tqdm # bokeh  # velocity


# GPU driver install P100 and K80
sudo apt install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

## GPU driver install V100
##https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1710&target_type=debnetwork
#curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/cuda-repo-ubuntu1710_9.2.88-1_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu1710_9.2.88-1_amd64.deb
#sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub
#sudo apt-get update -y
#sudo apt-get install -y cuda
#sudo nvidia-smi -pm 1

# Shutdown
sudo apt autoremove -y
rm -rf vminstall.bash
sudo shutdown now


# Extras
# find MATLAB/ -size +100M -ls
