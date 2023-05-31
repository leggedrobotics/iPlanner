# iPlanner and Imperative Training Installation Guide

This guide provides step-by-step instructions on how to setup your environment and install all necessary packages required for iPlanner and Imperative Training.

## Manual Installation

Follow the steps below for a manual installation:

1. Create a new conda environment named 'iplanner' with Python 3.8:
    ```
    conda create -n iplanner python=3.8
    ```

2. Activate the 'iplanner' conda environment:
    ```
    conda activate iplanner
    ```

3. Install the required packages with pip3:
    ```
    pip3 install torch torchvision pypose open3d opencv-python rosnumpy
    ```

4. Install numpy version 1.23.5 with conda:
    ```
    conda install -c anaconda numpy==1.23.5
    ```

5. Install rospkg and wandb from the conda-forge channel:
    ```
    conda install -c conda-forge rospkg wandb
    ```

## Using a YAML file

You can also setup your environment by installing all necessary packages at once using the `iplanner_env.yaml` file:

1. Create and activate the environment from the `iplanner_env.yml` file:
    ```
    conda env create -f iplanner_env.yml
    ```

If you encounter any issues during the installation process, please open an issue on the project's GitHub page.
