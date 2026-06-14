# EMG-Net: Edge-Memory Guided Deep Unrolling for Structurally Faithful Accelerated MRI Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.11+](https://img.shields.io/badge/pytorch-1.11+-orange.svg)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**EMG-Net** is a state-of-the-art deep learning framework implemented in PyTorch for Magnetic Resonance Imaging Compressed Sensing (MRI-CS) reconstruction. By tightly binding the physical data consistency of the MRI sampling process, EMG-Net successfully reconstructs high-fidelity, artifact-free MR images directly from under-sampled K-space/image domains.

---

## Environment Setup

Ensure your local system has a CUDA-compatible PyTorch environment. You can rapidly build up the production defense line with the following commands:

```bash
# Clone the repository
git clone git@github.com:Super-Czy/EMG_Net.git
cd EMG_Net

# Create and activate a conda environment
conda create -n emg_net python=3.8
conda activate emg_net

# Install Pytorch and specific CUDA-related packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install additional research dependencies
pip install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    opencv-python==4.12.0.88 \
    matplotlib==3.7.5 \
    pillow==10.4.0 \
    einops==0.8.1 \
    nibabel==5.2.1 \
    timm==1.0.19 \
    lpips==0.1.4 \
    fvcore==0.1.5.post20221221 \
    yacs==0.1.8 \
    scikit-image==0.21.0 \
    h5py==3.11.0 \
    tqdm==4.67.1

# test
python Test_EMG_Net.py
