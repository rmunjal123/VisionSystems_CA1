# VisionSystems_CA1
In this project, we are doing the semantic segmentation of the surface defects.There are 3 main classes of defects which are indicated by the filename of the photos:
•	Blowhole
•	Cavity 
•	Crack

The model developed is using MaskRCNN state of the art method of instance segmentation. Please create the python environment as stated below to run the code in local machine.

Windows, linux:
1. conda create -n y2-s1-ca1 python=3.6 numpy=1.18.1 opencv=3.4.2 matplotlib=3.1.3 tensorflow=1.14.0 tensorflow-gpu=1.14.0 keras=2.3.1 keras-gpu=2.3.1 cudatoolkit=10.0 cudnn=7.6.5 scipy=1.4.1 scikit-learn=0.22.1 pillow=6.1.0 ipython=7.12.0 spyder=4.0.1 imgaug cython pathlib yaml pandas pydot graphviz seaborn

(Assume Nvidia Cuda 10.2 is installed according to your Nvidia driver version (see https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility), and cudnn 7.6.5 installed, get cudnn from https://developer.nvidia.com/cuda-downloads)

 2. conda activate y2-s1-ca1
 
 Using Pre-trained model for inference:
 Please downlaod the pre-trained model from https://nusu-my.sharepoint.com/:u:/g/personal/e0385030_u_nus_edu/EcNzOTG_nFdCnzMFjxfP4-MBskJRIJsqj0B91Mzo2pEZVg?email=tanjenhong%40nus.edu.sg&e=jIy3Tn and put the correct path in the inference code to test the model.
