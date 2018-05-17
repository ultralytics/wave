# Run this script on a brand new GCP VM. Recommend Ubuntu 17.10, 20 GB Persistent disk, P100 GPU, Skylake CPU.

sudo apt update -y
sudo apt install -y git
sudo apt install -y python3-pip
# pip3 install virtualenv
# virtualenv venv

# Install Drive FUSE wrapper. https://github.com/astrada/google-drive-ocamlfuse
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:alessandro-strada/ppa
sudo apt update -y
sudo apt install -y google-drive-ocamlfuse

# Generate creds for the Drive FUSE library.
google-drive-ocamlfuse -headless -id 280545512213-gaaccol136ti4bt46g1imip3is7h10ph.apps.googleusercontent.com -secret l1aNm3ecGgKOit3vQ_TwBhNM

# Create a directory and mount Google Drive using that directory.
# mkdir drive
# google-drive-ocamlfuse drive
# fusermount -u drive  # unmount

# PYTORCH FOR CUDA 9
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
pip3 install numpy scipy tensorflow plotly

# GPU DRIVER INSTALL https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script
#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
# The 17.04 installer works with 17.10.
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-9-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1
# Verify driver is installed
nvidia-smi

sudo shutdown

# find MATLAB/ -size +100M -ls
