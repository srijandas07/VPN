# VPN: Learning Video-Pose Embedding for Activities of Daily Living (ECCV 2020)

## News


## Introduction

This repo contains implementation of the paper Video-Pose Embedding For Activities of Daily Living (VPN) in Keras. VPN works with a base 3D video understanding model for feature extraction from I3D and adds Pose based Attention Network to weight the importance of video features towards activity recognition. 

### VPN Architectural Overview
![](image.png)

## Results
We show the results of VPN on four activity recognition datasets of varied complexity - Smarthomes, NTU-60, NTU-120 and NUCLA. 

## Get Started
First,  

### Dataset Instructions 
* environment.yaml file will be provided


### Pre-Trained Models

|    Model   |    Dataset    |                           Weights                           |
|------------|---------------|-------------------------------------------------------------|
|     VPN    |  Smarthomes   | [Google Drive](https://drive.google.com/drive/u/0/my-drive) |
|     VPN    |  NTU-60       | [Google Drive](https://drive.google.com/drive/u/0/my-drive) |
|     VPN    |  NTU-120      | [Google Drive](https://drive.google.com/drive/u/0/my-drive) |
|     VPN    |  NUCLA        | [Google Drive](https://drive.google.com/drive/u/0/my-drive) |


### Train
To train VPN with I3D as backbone on Smarthomes, execute the below line


### Test
To test VPN with I3D as backbone on Smarthomes, execute the below line


## Citing VPN
    @misc{das2020vpn,
        title={VPN: Learning Video-Pose Embedding for Activities of Daily Living},
        author={Srijan Das and Saurav Sharma and Rui Dai and Francois Bremond and Monique Thonnat},
        year={2020},
        eprint={2007.03056},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
