<img src="https://storage.googleapis.com/ultralytics/logo.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information on Ultralytics projects please visit:
http://www.ultralytics.com.


# Description

The https://github.com/ultralytics/wave repo contains **WA**veform **V**ector **E**xploitation code, a new method for particle physics detector readout and reconstruction based on Machine Learning and Deep Neural Networks.

# Requirements

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `scipy`
- `torch` >= 0.4.0
- `tensorflow` >= 1.8.0
- `plotly` (optional)

# Running
- WAVE in pytorch: `wave_pytorch.py` 
- WAVE in tensorflow: `wave_tf.py`
- WAVE in pytorch on Google Cloud Platform: `gcp/wave_pytorch_gcp.py`

![Alt](https://github.com/ultralytics/wave/blob/master/data/waveforms.png "waveforms")
![Alt](https://github.com/ultralytics/wave/blob/master/data/wave.png "training")


# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
