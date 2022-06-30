import os
import tensorflow as tf
import cv2
import numpy as np
import json
import shutil
from collections import deque
from model_architecture import build_tools
from utils.image_folder import make_dataset
from PIL import Image
from utils.utils import data_tools
from utils.FPS import FPS
from config import *
from tqdm import tqdm

from packaging import version
from tensorflow import __version__ as tfver
using_tf2 = version.parse(tfver) >= version.parse("2.0.0")

# Allow memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if not using_tf2:
        session = tf.Session(config=config)
    print('\n'*2 + '--Memory Growth enabled--' + '\n'*2)

#Función para encontrar un path
def get_folder(path):
    folder = '/'
    for x in path.split('/')[1:-1]:
        folder = folder + x + '/'
    return folder

#Función para que imágenes salgan en orden correcto
def sort_names(image_names):
    # Find Indices
    idx = [int(name.split('_')[-1].split('.')[0]) for name in image_names]

    # Create empty list to store the sorted names
    sorted_names = [None]*(max(idx)+1)

    # Sort names
    for i, j in enumerate(idx):
        sorted_names[j] = image_names[i]

    # Eliminate empty elements
    sorted_names = [name for name in sorted_names if name is not None]

    return sorted_names

#Para que lea todas las imágenes del dataset completo:
dataset_dir = '/home/pame/PROYECT1/Vehicle_Collision-master/files/TEST'
output_dir = './files/test_set/'
name = '{}_{}_{}_e{}_ns{}.npz'.format(model_name, data_type, vers, test_epoch, nsequences)
output_file = os.path.join(output_dir, name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# *********** 1.A  TEST   ************
# ALCOHOL
alcohol_dir = os.path.join(dataset_dir, 'Test_alcohol')
alcohol_dataset = make_dataset(alcohol_dir)

# DRUG
drug_dir = os.path.join(dataset_dir, 'Test_drug')
drug_dataset = make_dataset(drug_dir)
# SLEEP
sleep_dir = os.path.join(dataset_dir, 'Test_sleep')
sleep_dataset = make_dataset(sleep_dir)
# CONTROL
control_dir = os.path.join(dataset_dir, 'Test_control')
control_dataset = make_dataset(control_dir)

#Listado de carpetas que hay
def carpetas (dataset):
    folders = []
    for archivo in dataset:
        folders.append(get_folder(archivo))
    folders = np.unique(folders)
    return(folders)

# *********** 2.A TEST FOLDERS ***********
folders_alcohol = carpetas((alcohol_dataset))
folders_drug = carpetas(drug_dataset)
folders_sleep = carpetas(sleep_dataset)
folders_control = carpetas(control_dataset)

# ********* Create model ***********
model_tools = build_tools()
network = model_tools.create_network(model_name)
print('\n--Network created successfully--\n')

#Load weights 
network.load_weights(checkpoint_path.format(epoch=test_epoch))
print('\n--Weights loaded successfully--\n')

# ****** FUNCIONES  *******
def inference (network, folder, clase = [0,1], tipo = 0, nsequences = None):
    predictions = []
    clase = np.array(clase, dtype='uint8')
    tipo = np.array(tipo, dtype='uint8')
    for register in tqdm(folder, desc = "Making predictions: "):
        sequence_dataset = make_dataset(str(register))
        sequence_dataset = sort_names(sequence_dataset)
        #print('SHAPE DE SEQUENCE DATASET',  np.shape(sequence_dataset))

        if nsequences == 'all':
            nsequences = len(sequence_dataset) - 8
        else:
            nsequences = min([nsequences, len(sequence_dataset) - 8])

        for i in range(nsequences):
            seq_tensor = []
            ind_1 = i
            ind_2 = ind_1+8
            for j in range(ind_1, ind_2):

                img = Image.open(sequence_dataset[j]).convert('RGB').resize((210,140))
                #Debería ser un (140,210,3)
                img = np.array(img)
                # adquiere el (8,140,210,3)
                seq_tensor.append(img)
            
            np_image_seqs = np.reshape(np.array(seq_tensor)/255,(1,time,height,width,color_channels))
            r = network.predict(np_image_seqs)
            r = np.squeeze(r, 0)

            data = [r, clase, tipo]
            predictions.append(data)
            #print('PREDICTIONS',predictions)
            print('SHAPE DE SEQUENCE DATASET',  np.shape(sequence_dataset))
            print('TAMAÑO DE PREDICTIONS', len(predictions))
            print('SHAPE DE PREDICTIONS', np.shape(predictions))
    # tamaño final (N,8,140,210,3)
    return(predictions)

# *************** 4.A  TEST FINAL **************
# No control
print('Alcohol:')
if data_type == 'iris_2classes':
    clase = [1,0]
    tipo = 0
if data_type == 'iris_4classes':
    clase = [1,0,0,0]
    tipo = 0


pred_alcohol = inference (network, folders_alcohol, clase = clase, tipo = tipo, nsequences = nsequences)

print('\nDrug:')
if data_type == 'iris_2classes':
    clase = [1,0]
    tipo = 0
if data_type == 'iris_4classes':
    clase = [0,0,1,0]
    tipo = 2
pred_drug = inference (network, folders_drug, clase = clase, tipo = tipo, nsequences = nsequences)
print('\nSleep:')
if data_type == 'iris_2classes':
    clase = [1,0]
    tipo = 0
if data_type == 'iris_4classes':
    clase = [0,0,0,1]
    tipo = 3
pred_sleep = inference (network, folders_sleep, clase = clase, tipo = tipo, nsequences = nsequences)

# Control
print('\nControl:')
if data_type == 'iris_2classes':
    clase = [0,1]
    tipo = 1
if data_type == 'iris_4classes':
    clase = [0,1,0,0]
    tipo = 1
pred_control = inference (network, folders_control, clase = clase, tipo = tipo, nsequences = nsequences)


# *** TEST ***
# V1: CONTROL + ALCOHOL, SIN MODIFICACIONES (26 MAYO)
all_predictions = pred_control + pred_alcohol + pred_drug + pred_sleep

predictions = []
clase = []
tipo = []


for p,c,t in all_predictions:
    predictions.append(p)
    clase.append(c)
    tipo.append(t)

predictions = np.array(predictions)
clase = np.array(clase)
tipo = np.array(tipo)

print('\nSaving dataset')

np.savez(output_file, predictions = predictions, clase = clase, tipo = tipo)

print('\nDataset packed successfully!!!\n')
