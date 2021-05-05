# FSDL-FinalProject

## File Structure

- `HAM10000_images_part_1` - Image Data Part 1
- `HAM10000_images_part_2` - Image Data Part 2
- `dataset/` - Data Directory
    - `HAM10000_metadata` - Tab Delimited File for Metadata
- `annotation.py` - Script for automatic annotation
- `raw_annotation.csv` - Raw annotation data
- `setup.sh` - Script for Setup/Installation
- **Jupyter Files** are for experimentation

## Dataset

Dataset can be found here: 

## Labels

`{'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}`

## Setup/Dependencies

This part of the project uses `python 3.8` in a `conda` environment with the following dependencies. The `setup.sh` file can be run to initiate the online environment.

- sklearn
- matplotlib
- open-cv
- scikit-image
- numpy
- pandas
- os
