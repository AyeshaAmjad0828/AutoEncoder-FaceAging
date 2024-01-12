# AutoEncoder-FaceAging

In this project, a simple tensorflow implemented autoencoder is trained on 50 young and 50 old faces to create a face aging application. 

## Train Data

[UTKFace](https://susanqq.github.io/UTKFace/) dataset is used for this exercise, particularly the repository of aligned and cropped faces. 

The images in this dataset are sorted into two folder "[data/utk/young](https://github.com/AyeshaAmjad0828/AutoEncoder-FaceAging/tree/main/data/utk/young)" and "[data/utk/old](https://github.com/AyeshaAmjad0828/AutoEncoder-FaceAging/tree/main/data/utk/old)". The code for this process is written in [preprocess-utkcropped.py](https://github.com/AyeshaAmjad0828/AutoEncoder-FaceAging/blob/main/preprocess-utkcropped.py)



## Autoencoder Training on UTKFace

Two architectures of autoencoder were trained using tensorflow library :

1. [VGG16 network](https://github.com/AyeshaAmjad0828/AutoEncoder-FaceAging/blob/main/Autoencoder-VGG.py) 
2. [A Simple CNN](https://github.com/AyeshaAmjad0828/AutoEncoder-FaceAging/blob/main/Autoencoder-Simple.py)

Data preprocessing steps such as resizing and normalization are performed on images in both young and old folders before passing it into the autoencoder network. 



## Results

The results of the VGG16 encoder and decoder network on UTKFace dataset are in [Results-Autoencoder-VGG.ipynb](https://github.com/AyeshaAmjad0828/AutoEncoder-FaceAging/blob/main/Results-Autoencoder-VGG.ipynb) notebook. 



> Training with a larger sample and more epochs can potentially result in better face aging capability of the autoencoder. 
