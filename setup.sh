
# Create Conda Environment

conda create -n skin python=3.8.3

# Install Packages

pip install sklearn
pip install numpy
pip install pandas
pip install torch
pip install opencv-python
pip install scikit-image
pip install imgaug



git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0
cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
