<img src="https://storage.googleapis.com/ultralytics/UltralyticsLogoName1000×676.png" width="200">  

# Introduction

This directory contains software developed by Ultralytics LLC. For more information on Ultralytics projects please visit:
http://www.ultralytics.com  

# WAVE

The https://github.com/ultralytics/wave repo contains **WA**veform **V**ector **E**xploitation code, a new method for particle physics detector readout and reconstruction based on Machine Learning and Deep Neural Networks.

# Requirements

Python 3.6 or later with the following packages installed:  

- ```numpy```
- ```scipy```
- ```pytorch``` >= 0.4.0
- ```tensorflow``` >= 1.8.0
- ```plotly``` (optional)

# Running WAVE
1. Install and/or update requirements:

    ```pip3 install numpy scipy torch tensorflow plotly --update```
    
    or
    
    ```pip3 install -U -r requirements.txt```

2. Run WAVE in pytorch:

    ```wave_pytorch.py``` 

    or tensorflow:

    ```wave_tfeager.py```

3. Fit particles better :)

![Alt](https://github.com/University-of-Hawaii-Physics/mtc/blob/master/Analysis/Ultralytics/mtcview.png "mtcView")
![Alt](https://github.com/ultralytics/wave/blob/master/data/wave.png "training")


# Contact

For questions or comments please contact Glenn Jocher at glenn.jocher@ultralytics.com or visit us at http://www.ultralytics.com/contact
