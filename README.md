<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🌊 Introduction

Welcome to the [Ultralytics WAVE repository](https://github.com/ultralytics/wave)! This repository hosts the cutting-edge solution for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml)-driven analysis and interpretation of waveform data, particularly tailored for applications in [particle physics](https://en.wikipedia.org/wiki/Particle_physics). 🎉

Here, we introduce **WA**veform **V**ector **E**xploitation (WAVE), a novel approach leveraging [Deep Learning](https://www.ultralytics.com/glossary/deep-learning-dl) to readout and reconstruct signals from particle physics detectors. This open-source codebase aims to foster collaboration and innovation at the exciting intersection of ML and physics.

[![Ultralytics Actions](https://github.com/ultralytics/wave/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/wave/actions/workflows/format.yml) [![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics) [![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/) [![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🚀 Project Objectives

The primary goal of this project is to develop and share advanced [Machine Learning](https://www.ultralytics.com/glossary/machine-learning-ml) techniques applicable to full-waveform time-of-flight detectors. These methods are designed to enhance signal processing and interpretation, pushing the boundaries of particle physics research.

## 🌟 Key Features

- **Framework Flexibility**: Implementation of WAVE using both [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).
- **User-Friendly Codebase**: Designed for ease of use and adaptability to various research needs.
- **Cloud Integration**: Support for running WAVE experiments on [Google Cloud Platform (GCP)](https://cloud.google.com/).
- **Visualization Examples**: Sample images illustrating waveform analysis and training progress.

## 🔧 Requirements

Before diving into waveform vector exploitation with WAVE, ensure your environment meets the following prerequisites:

- [Python](https://www.python.org/) 3.7 or later.
- Essential Python packages installed via `pip3 install -U -r requirements.txt`:
  - `numpy`
  - `scipy`
  - `torch` (version 0.4.0 or later)
  - `tensorflow` (version 1.8.0 or later)
  - `plotly` (optional, for enhanced visualization)

You can easily install these packages using [pip](https://pip.pypa.io/en/stable/), the Python package installer.

## 🏃 Run Instructions

Execute the WAVE models using the provided scripts:

- **PyTorch Implementation**: Run `wave_pytorch.py` for the [PyTorch](https://pytorch.org/)-based model.
- **TensorFlow Implementation**: Use `wave_tf.py` if you prefer [TensorFlow](https://www.tensorflow.org/).
- **Google Cloud Deployment**: Explore `gcp/wave_pytorch_gcp.py` for running on [Google Cloud Platform](https://cloud.google.com/).

Visualize the intricacies of waveform signals and the training process with these example images:

![Waveform Signals](https://raw.githubusercontent.com/ultralytics/wave/main/data/waveforms.png)
![Training Visualization](https://raw.githubusercontent.com/ultralytics/wave/main/data/wave.png)

## 📜 Citation

If you utilize this code or the WAVE methodology in your research, please cite the original paper:

- Jocher, G., Nishimura, K., Koblanski, J. and Li, V. (2018). WAVE: Machine Learning for Full-Waveform Time-Of-Flight Detectors. _arXiv preprint arXiv:1811.05875_. Available at: [https://arxiv.org/abs/1811.05875](https://arxiv.org/abs/1811.05875).

## 🤝 Contribute

We highly value community contributions and invite you to participate in advancing this pioneering ML approach for physics! Whether it's fixing bugs, proposing new features, or improving documentation, your input is welcome. Learn how to contribute by reading our [Contributing Guide](https://docs.ultralytics.com/help/contributing/). We also encourage you to share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge thank you 🙏 to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics provides two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license) [open-source license](https://github.com/ultralytics/wave/blob/main/LICENSE) is ideal for students and researchers, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/wave/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license allows for the integration of Ultralytics software and AI models into commercial products and services. Visit [Ultralytics Licensing](https://www.ultralytics.com/license) for more information.

## 📬 Contact Us

For bug reports, feature requests, and contributions, please use [GitHub Issues](https://github.com/ultralytics/wave/issues). For broader questions and discussions about the WAVE project or other Ultralytics initiatives, join our vibrant community on [Discord](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
