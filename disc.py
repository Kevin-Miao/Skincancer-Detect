import pandas as pd 
import os
final = pd.read_csv('final.csv')
paths = final['path']
broken = []
for p in paths:
    try:
        from PIL import Image
        Image.open('data/kevinmiao/' + p)
    except:
        broken.append(p)
print(broken)
