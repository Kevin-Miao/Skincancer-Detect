# Cancer Detection: Modeling

**Authors** [Maxwell Chen](mailto:maxhchen@berkeley.edu) and [Kevin Miao](mailto:kevinmiao@cs.berkeley.edu)

*This repository contains our final project for [CS194-80: Full Stack Deep Learning](bit.ly/berkeleyfsdl) taught at UC Berkeley by Pieter Abbeel, Sergey Karayev and Joshua Tobin.*

The deployment phase can be accessed here:

## File Structure

- `Archive` - Contains older versions and debugging notebooks/scripts.
- `HAM10000_metadata.csv` - Original metadata with diagnoses (unmodified from the original HAM10000 dataset)
- `annotation-v2.py` - This code contains the annotation script which outputs `final.csv` that is used as ground truth labels and bounding box areas by using the provided segmentation maps.
- `annotation.py` - Script for automatic annotation
- `dataset.py` - Pytorch dataset accompanied by transforms
- `disc.ipynb/py` - Debugging files
- `final.csv` - Ground Truth bounding box coordinates, paths and labels
- `mean-std.pt` - PyTorch Dictionary containing the mean/std of the training images
- `model_util.py` - Util functions for loading/reading models from state
- `setup.sh` - Shell script for setting up requirements and dependencies
- `state_loading.py` - Script for loading a state dictionary into a model
- `sweep.yaml` - Weights and Biases files for hyperparameter sweeping
- `train.ipynb` - Notebook for training debugging
- `train.py` - Official training script
- `transforms.py` - Image Transforms
- `util.py` - Contains util functions

## Dataset

`Tschandl, Philipp, 2018, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", https://doi.org/10.7910/DVN/DBW86T, Harvard Dataverse, V3, UNF:6:/APKSsDGVDhwPBWzsStU5A== [fileUNF]`

The basis of the project is the HAM10000 dataset which contains 10,015 images categorized in 7 different skin cancers along with supervised segmentation maps. 

Dataset can be downloaded here: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) 

## Model Architecture

The architecture being used is a pretrained `FasterRCNN` with a `ResNet50` backbone augmented with a Feature Pyramid Network. The model has been adapted from: [torchvision FasterRCNN Resnet-50 fpn pretrained on COCO](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html)


## Labels

We have the following diseases in our dataset which correspond to the respective indices. The last index, 7, is reserved for background.

**Dictionary** : `{'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}`


## Setup/Dependencies

This part of the project uses `python 3.8` in a `conda` environment with the following dependencies. The `setup.sh` file can be run to initiate the online environment.

