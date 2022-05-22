import numpy as np
import os
import cv2
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from imutils import paths
from tqdm import tqdm
import sys

print(tf.__version__)
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#INPUT SIZE MODEL
input_shape = (224, 224)

# try:
# Load model TF1.x
model = keras.models.load_model('./files/vgg/model_folder/model_weights_032.h5')
print("\n"*2+"-- Model Loaded Successfully--"+"\n"*2)

video_path = './files/samples/GTA_V_1.mp4'
'''
#Load img & normalize
input_img1 = load_image(img_path, colorspace='RGB')
shape = input_img1.shape
input_img = input_img1/255
input_img =  cv2.resize(input_img, input_shape)

#Predict & normalize mask
output = model.predict(np.expand_dims(input_img, axis=0), steps=1)[0]

# Interprete Otuput
print(' ')
print('Alcohol Probability: {:0.1f}%'.format(100*output[0]))
print('Control Probability: {:0.1f}%'.format(100*output[1]))
print('Drugs   Probability: {:0.1f}%'.format(100*output[2]))
print('Sleef   Probability: {:0.1f}%'.format(100*output[3]))
'''
