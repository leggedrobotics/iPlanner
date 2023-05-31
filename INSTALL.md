# iPlanner and Imperative Training Installation Guide

This guide provides step-by-step instructions on how to setup your environment and install all necessary packages required for iPlanner and Imperative Training.

## Manual Installation Steps

If you're opting for a manual installation, follow these steps:

1. Create a new conda environment named 'iplanner' with Python 3.8 using the command:
    ```bash
    conda create -n iplanner python=3.8
    ```

2. Activate the 'iplanner' conda environment with the following command:
    ```bash
    conda activate iplanner
    ```

3. Install PyTorch and Torchvision using the command:
    ```bash
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4. Install numpy version 1.23.5 using conda:
    ```bash
    conda install -c anaconda numpy==1.23.5 nbconvert
    ```

5. Install rospkg, wandb, and PyYAML from the conda-forge channel using the command:
    ```bash
    conda install -c conda-forge rospkg wandb pyyaml==6.0
    ```

6. Install additional necessary packages using pip3:
    ```bash
    pip3 install pypose open3d opencv-python rosnumpy
    ```

## Installation Using a YAML File

As an alternative to the manual installation, you can set up your environment and install all necessary packages at once using the `iplanner_env.yaml` file:

1. Create and activate the environment from the `iplanner_env.yml` file using the command:
    ```bash
    conda env create -f iplanner_env.yml
    ```

If you encounter any issues during the installation process, please open an issue on the project's GitHub page.
