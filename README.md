# Introduction 🌊

Welcome to the [Ultralytics WAVE repository](https://github.com/ultralytics/wave)! Here you will find innovative **W**aveform **A**nalysis **V**ia **E**ducation (WAVE) code, a cutting-edge approach for detector readout and data reconstruction in particle physics. Utilizing state-of-the-art Machine Learning (ML) and Deep Neural Networks (DNNs), our methods aim to enhance the precision and efficiency of time-of-flight measurements.

The significance of this work extends beyond the realm of physics into various fields where waveform data is crucial. Stay tuned as we explore the potentials of this exciting technology together!

# Requirements 📋

Ensure that you have Python 3.7 or later on your system. Begin by installing the necessary dependencies via pip:

```bash
pip3 install -U -r requirements.txt
```

In your `requirements.txt` file, you'll need the following libraries:

- `numpy`: For handling numerical data efficiently.
- `scipy`: For scientific and technical computing.
- `torch` >= 0.4.0: PyTorch deep learning framework.
- `tensorflow` >= 1.8.0: TensorFlow ML platform.
- `plotly`: Optional, for creating interactive plots.

# Run 🚀

Here's how to run the WAVE models:

- WAVE in PyTorch:
  ```bash
  python wave_pytorch.py
  ```
- WAVE in TensorFlow:
  ```bash
  python wave_tf.py
  ```

If you're interested in deploying on Google Cloud Platform (GCP), utilize:
- WAVE in PyTorch on GCP:
  ```bash
  python gcp/wave_pytorch_gcp.py
  ```

Visualize the waveforms and training process through these plots:

![Waveform data representation](https://github.com/ultralytics/wave/blob/master/data/waveforms.png "Waveforms")
![Training visualization](https://github.com/ultralytics/wave/blob/master/data/wave.png "Training")

# Citation 📖

To cite our work, please refer to:

> Jocher, G., Nishimura, K., Koblanski, J. and Li, V. (2018). WAVE: Machine Learning for Full-Waveform Time-Of-Flight Detectors. Arxiv.org. Available at: [https://arxiv.org/abs/1811.05875](https://arxiv.org/abs/1811.05875).

By crediting our efforts, you contribute to the vitality of the open-source scientific community. Your acknowledgments support ongoing research and innovation.

# Contribute 🤝

We love your input! Ultralytics open-source efforts would not be possible without help from our community. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started, and fill out our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experience. Thank you 🙏 to all our contributors!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->
<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

# License 📜

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/licenses/) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://ultralytics.com/license).

Looking to apply our technology in a commercial context? We're here to facilitate your journey and ensure that our tools seamlessly integrate within your product ecosystem.

# Contact ☎️

Have a bug to report or a feature to suggest? Visit [GitHub Issues](https://github.com/ultralytics/wave/issues) for help. Also, feel free to join our vibrant [Discord](https://ultralytics.com/discord) community to engage in discussions and get your questions answered!

<div align="center">
  Connect with us on social media and join our tech-savvy circles:
  <br><br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
