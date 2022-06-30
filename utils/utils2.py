import cv2
import os
import json
import random
from config import *
import numpy as np
import h5py
#Función para encontrar un path
def get_folder(path):
    folder = '/'
    for x in path.split('/')[1:-1]:
        folder = folder + x + '/'
    return folder

#Función para que imágenes salgan en orden correcto
def sort_names(image_names):
    # Find Indices
    idx1 = [int(name.split('_')[-1].split('.')[0]) for name in image_names]
    idx2 = [int(name.split('_')[-2].split('.')[0]) for name in image_names]
    
    unique_idx1 = np.unique(idx1)
    unique_idx2 = np.unique(idx2)
    
    if len(unique_idx1) > len(unique_idx2):
        idx = idx1
    else:
        idx = idx2

    # Create empty list to store the sorted names
    sorted_names = [None]*(max(idx)+1)

    # Sort names
    for i, j in enumerate(idx):
        sorted_names[j] = image_names[i]

    # Eliminate empty elements
    sorted_names = [name for name in sorted_names if name is not None]

    return sorted_names