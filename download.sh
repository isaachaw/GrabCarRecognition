#!/bin/bash

# Download training data

mkdir -p ./data
pushd ./data

wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
tar xf cars_train.tgz

wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
tar xf car_devkit.tgz

popd

# Download weight for YOLOv3

mkdir -p ./weights
pushd ./weights

wget https://pjreddie.com/media/files/yolov3.weights

popd
