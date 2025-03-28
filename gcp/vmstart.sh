#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Run this script on every GCP VM startup.
# If this is a new VM, run vminstall.bash first.
# Recommend running this script in a 'screen' (if connection drops run 'screen -r' to reattach)

# 0. Remove old files
rm -rf data
rm -rf results
rm -rf wave

# 1. unmount, delete, and remount Google Drive
fusermount -u drive
rm -rf drive
mkdir drive
google-drive-ocamlfuse drive

# 2. fresh clone repo
rm -rf wave && git clone https://github.com/ultralytics/wave
cp -r data/* wave/data && cd wave
python3 train.py --var 2

# 3. download training data
mkdir data
mkdir results
# wget -P data https://storage.googleapis.com/ultralytics/wavedata3ns.mat
wget -P data https://storage.googleapis.com/ultralytics/wavedata25ns.mat

# 4a. Run python and then copy results to drive
python3 -c 'import torch; print(torch.cuda.device_count())'
python3 wave/wave_pytorch.py
#python3 -c 'import sys;
#sys.path.append("/home/glenn_jocher1/wave");
#sys.path.append("/home/glenn_jocher1/wave/gcp");
#import wave_pytorch_gcp as a; a.tsnoact()'
cp -r results/. drive/results #copy results to fused Google Drive
