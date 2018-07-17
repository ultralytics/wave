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
pip3 install -U numpy scipy #tensorflow # plotly  # wave
pip3 install -U opencv-python exifread tqdm # bokeh  # velocity
pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  # CUDA 9.1
# pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  # CUDA 9.0
pip3 install torchvision

# GPU driver install P100 and K80
sudo apt install python3-opencv
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


# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1531850600&Signature=FvYp1qDdadcfOAF2ELmHSJXaRYkq~R2KrRi0Fk3akL1~UZWiCm26QjLh52e11Ga-99GNAkLylXauCgt0k17hmw2aMrMDW-z9Qo9hGQV-BkYEimhd~dyybOqqGJ3ZWG3CmeesHHJ7ScdDpv9aIxZTNo-QUSABA8g5X2oMs96RWOy-GnAw09W8liBIoLAfeoGcqOubvY7vOMtFeFgFatzmMSoLPQ-Y8Zv2bGpQyih-pd7A2S0VAE3ccDwvjKgdOgYeuZLXBNaF5Wy~-JNX2RdaqaXmLO42P3soxT5FnCnGbLYoVAI7K6-mtlttcw0VOTMXqWvoN8QOsdZenREhKfJ0iw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._*

# convert all .tif to .bmp
sudo rm -rf train_images/659.tif train_images/769.tif
sudo rm -rf yolo
git clone https://github.com/ultralytics/yolo
cd yolo
python3
from utils import datasets
datasets.convert_tif2bmp('../train_images')
exit()

# Shutdown
sudo apt autoremove -y
rm -rf vminstall.bash
sudo shutdown now


