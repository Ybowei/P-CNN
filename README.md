## P-CNN : Prototype-CNN for Few-Shot Object Detection in Remote Sensing Images
Code for reproducing the results in the following paper, and the code is built on top of [MetaR-CNN](https://github.com/yanxp/MetaR-CNN)

P-CNN : Prototype-CNN for Few-Shot Object Detection in Remote Sensing Images

Gong Cheng, Bowei Yan, Peizhen Shi, Ke Li, Xiwen Yao, Lei Guo, and Junwei Han

<p align=center><img src="Prototype-CNN.jpg"/></p>

### License

For Academic Research Use Only!

### Requirements

+ python packages
  
  + Python = 3.6
  
  + PyTorch = 0.3.1
    
    *This project can not support pytorch 0.4, higher version will not recur results.*

  + Torchvision >= 0.2.0

  + cython

  + pyyaml

  + easydict

  + opencv-python

  + matplotlib

  + numpy

  + scipy

  + tensorboardX
  
+ CUDA 8.0

+ gcc >= 4.9

### Getting Started

Clone the repo:


    https://https://github.com/Ybowei/P-CNN.git



### Compilation

Compile the CUDA dependencies:


    cd {repo_root}/lib
    sh make.sh


It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align


### Data Preparation

create a data folder under the repo,

    cd {repo_root}
    mkdir data
    
