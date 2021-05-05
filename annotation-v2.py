import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import *

if __name__ == "__main__":
    working_directory = [x for x in os.listdir() if 'HAM10000_images' in x]

    result = pd.DataFrame(columns=['path', 'image_id'])
    for directory in working_directory:
        for image_path in os.listdir(directory):
            path = os.path.join(directory, image_path)
            result = result.append(pd.DataFrame(
                [[path, image_path]], columns=result.columns), ignore_index=True)
    result['image_id'] = result['image_id'].apply(lambda x: x[:-4])
    final = pd.merge(result, pd.read_csv('dataset/HAM10000_metadata'),
                     left_on='image_id', right_on='image_id', how='inner')
    diagnoses = sorted(final['dx'].unique())
    mapping = {x: diagnoses.index(x) for x in diagnoses}
    inv_mapping = {mapping[k]: k for k in mapping}
    final['diagnosis'] = final['dx'].apply(lambda x: mapping[x])
    final.to_csv('final.csv')

    matrix = []
    for index, image in enumerate(final['image_id']):
        seg = np.array(Image.open(os.path.join(
            'dataset/HAM10000_segmentations_lesion_tschandl', image + '_segmentation.png')))/255.
        im = np.array(Image.open(final['path'][index]))/255.
        y, x = np.indices((450, 600))
        y_seg = np.matrix.flatten(y * seg)
        x_seg = np.matrix.flatten(x * seg)
        ymin = np.min(y_seg[y_seg != 0])
        ymax = np.max(y_seg[y_seg != 0])
        xmin = np.min(x_seg[x_seg != 0])
        xmax = np.max(x_seg[x_seg != 0])
        matrix += [[xmin, ymin, xmax, ymax]]
    matrix = np.array(matrix)

    for i, col in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
        final[col] = matrix[:, i]
    final.to_csv('final.csv')
