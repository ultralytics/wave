#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Run this script on a brand new GCP VM. Recommend Ubuntu 18.04 LTS, 15 GB Persistent SSD, P100 GPU, Skylake CPU.
sudo apt update -y
sudo apt autoremove -y

# Install Drive FUSE wrapper. https://github.com/astrada/google-drive-ocamlfuse
sudo apt-get install dirmngr
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:alessandro-strada/ppa
sudo apt-get update -y
sudo apt-get install -y google-drive-ocamlfuse

# Generate creds for the Drive FUSE library.
google-drive-ocamlfuse -headless -id 280545512213-gaaccol136ti4bt46g1imip3is7h10ph.apps.googleusercontent.com -secret l1aNm3ecGgKOit3vQ_TwBhNM
#Create a directory and mount Google Drive using that directory.
mkdir drive
google-drive-ocamlfuse drive
#fusermount -u drive  # unmount

# Install Linux Programs
sudo apt install -y git unzip python3-pip screen vim

# Install Python Packages
pip3 install -U numpy scipy                                                               #tensorflow # plotly  # wave
pip3 install -U opencv-python exifread tqdm                                               # bokeh  # velocity
pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl # CUDA 9.1
# pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  # CUDA 9.0
pip3 install torchvision

# GPU driver install P100 and K80
sudo apt install python3-opencv -y
# sudo apt-get install python-opencv -y --allow-unauthenticated
sudo apt install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

# GPU driver install on *Ubuntu 16.04 LTS* from https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script
# echo "Checking for CUDA and installing."
# if ! dpkg-query -W cuda-9-0; then
# sudo curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
# sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo apt-get update
# sudo apt-get install cuda-9-0 -y
# fi
# # Enable persistence mode
# sudo nvidia-smi -pm 1
