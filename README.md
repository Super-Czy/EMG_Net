# EMG-Net: MRI Compressed Sensing Reconstruction Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.11+](https://img.shields.io/badge/pytorch-1.11+-orange.svg)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**EMG-Net** is a state-of-the-art deep learning framework implemented in PyTorch for Magnetic Resonance Imaging Compressed Sensing (MRI-CS) reconstruction. By tightly binding the physical data consistency of the MRI sampling process and embedding the powerful feature representation capabilities of **ConvNeXt V2** blocks, EMG-Net successfully reconstructs high-fidelity, artifact-free MR images directly from under-sampled K-space/image domains.

---

## 🎯 Key Features

- **Dual-Domain Constraints**: Alternates optimizations between the image domain (for deep priors, artifact removal, and texture preservation) and the K-space domain (for hard Data Consistency (DC) physical checks).
- **ConvNeXt V2 Powered**: Utilizes the modern ConvNeXt V2 convolutional block architecture enhanced by Global Response Normalization (GRN) to resolve over-smoothing issues and lock onto intricate cerebral cortex details.
- **Multi-Ratio Radial Sampling Support**: Pre-configured with clinical-grade radial sampling masks at 5%, 10%, 20%, 30%, and 40% under-sampling trajectories to validate robust performance across multiple acceleration factors.

---

## 🛠️ Environment Setup

Ensure your local system has a CUDA-compatible PyTorch environment. You can rapidly build up the production defense line with the following commands:

```bash
# Clone the repository
git clone git@github.com:Super-Czy/EMG_Net.git
cd EMG_Net

# Install required dependencies
pip install torch torchvision numpy scipy pillow opencv-python matplotlib

# test
python Test_EMG_Net.py
