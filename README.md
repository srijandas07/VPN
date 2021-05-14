# VPN: Learning Video-Pose Embedding for Activities of Daily Living (ECCV 2020)
Note: VPN++ Pytorch code is coming soon!!! Stay tuned...

## Contributors 

[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/0)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/0)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/1)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/1)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/2)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/2)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/3)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/3)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/4)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/4)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/5)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/5)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/6)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/6)[![](https://sourcerer.io/fame/srv902/srijandas07/VPN/images/7)](https://sourcerer.io/fame/srv902/srijandas07/VPN/links/7)

## News
Codes have been restructured and ready for benchmarking on datasets.

## Introduction

This repository contains implementation of the paper Video-Pose Embedding For Activities of Daily Living (VPN) in Keras. VPN works with a base 3D video understanding model such as Inception3D for feature extraction and adds Pose based Attention Network to weight the importance of video features towards activity recognition. 

### VPN Architectural Overview
![](image.png)

## Results
We show the results of VPN on four activity recognition datasets - Smarthomes, NTU-60, NTU-120 and NUCLA for different evaluation protocols. Currently, in this repo only I3D backbone is supported. 

|    Backbone   |    Dataset    |   Protocol  |    Clip Width   |   Accuracy (%)   |                           Model                             |
|:-------------:|:-------------:|:-----------:|:---------------:|:----------------:|:-----------------------------------------------------------:|
|     I3D       |  Smarthomes   |      CS     |       64        |       60.8       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|     I3D       |  Smarthomes   |     CV1     |       64        |       43.8       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|     I3D       |  Smarthomes   |     CV2     |       64        |       53.5       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|     I3D       |  NTU-60       |      CS     |       64        |       93.5       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|     I3D       |  NTU-60       |      CV     |       64        |       96.2       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|  ResNext-101  |  NTU-60       |      CS     |       64        |       95.5       |                                                             |
|  ResNext-101  |  NTU-60       |      CV     |       64        |       98.0       |                                                             |
|     I3D       |  NTU-120      |     CS1     |       64        |       86.3       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|     I3D       |  NTU-120      |     CS2     |       64        |       87.8       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |
|     I3D       |  NUCLA        |      CV     |       64        |       93.5       |[Google Drive](https://drive.google.com/drive/u/0/my-drive)  |


## Get Started
Before the start of VPN training, following steps should be completed

* Create a new or use the existing configuration files stored in `config` folder. The configuration files are specified by the type of model and the dataset to use. Refer to args defined in the main.py file for more details.

* Specify the paths of following files needed as input for VPN in the config yaml file.
    * Skeleton : 3D pose data stored as npz files for each video clip
    * CNN      : RGB video data
    * Splits   : Training, Validation and Test video data splits

* Make sure the necessary folders for storing model weights are created.

* Currently, only NTU60 and NTU120 is supported and config files and related files will be updated for other datasets Smarthomes, NUCLA later. 

## Train
To train VPN with I3D as backbone on NTU60, execute the below line. 

```
python main.py --dataset ntu60
```

## ToDos
- [x] Reorganize codebase to get started with model training quickly
- [ ] Add support for Smarthomes and NUCLA datasets 
- [ ] Benchmark results
- [ ] Update results 
- [ ] Upload Trained models
- [ ] Upload Demo videos for all datasets 
- [ ] Add support for other base 3D video models 

## Citing VPN
    @misc{das2020vpn,
        title={VPN: Learning Video-Pose Embedding for Activities of Daily Living},
        author={Srijan Das and Saurav Sharma and Rui Dai and Francois Bremond and Monique Thonnat},
        year={2020},
        eprint={2007.03056},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
