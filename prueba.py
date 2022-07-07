
from cgi import test
import os
import pathlib
from pickletools import uint8
from cv2 import resize
import numpy as np
from io import BytesIO
from PIL import Image
import sys
import cv2
import os.path
from os import path
from os import listdir
#import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils import shuffle
from utils.image_folder import make_dataset
from tqdm import tqdm
from config import *
from utils.utils2 import *


#Función para encontrar un path
#aqui estaba def_folder, ahora en utils2

#Función para que imágenes salgan en orden correcto
#aquí estaba sort_names, ahora en utils2

#Make dataset está en utils.image_folder

# Escoger set de partición
partition_set_inp = 'Validation' #Ultima creación, antes test y valid
partition_set_opt = 'valid_set'

#Para que lea todas las imágenes del dataset completo:
#dataset_dir = '/home/ubuntu/Github/Vehicle_Collision/files/iris_raw_dataset/Version_1.0'
dataset_dir = '/home/pame/PROYECT1/IRIS_DATA/One_Eye_DB_Classification_Separated_Class/Monocular_DB_Classification_Separated_Class/Version_1.0'
output_dir = os.path.join('./datasets_h5/iris_4classes_PRUEBADESECUENCIAS', partition_set_opt)
#iris_4classesv2 INCLUYE TIPO

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# *********** 1.A  TEST   ************
# ALCOHOL
alcohol_dir = os.path.join(dataset_dir, '02_Alcohol_DB', partition_set_inp)
alcohol_dataset = make_dataset(alcohol_dir)
# DRUG
drug_dir = os.path.join(dataset_dir, '03_Drug_DB', partition_set_inp)
drug_dataset = make_dataset(drug_dir)
# SLEEP
sleep_dir = os.path.join(dataset_dir, '04_Sleep_DB', partition_set_inp)
sleep_dataset = make_dataset(sleep_dir)
# CONTROL
control_dir = os.path.join(dataset_dir, '01_Control_DB', partition_set_inp)
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


def make_tensor_5D (folder, clase = [0,1], tipo = 0):
    tensor_5d = []
    clase = np.array(clase, dtype='uint8')
    tipo = np.array(tipo, dtype = 'uint8')
    for register in tqdm(folder, desc = "Making tensor 5D: "):
        sequence_dataset = make_dataset(str(register))
        #sequence_dataset = sort_names(sequence_dataset)    

        for i in range(len(sequence_dataset)-8):
            seq_tensor = []
            ind_1 = i
            ind_2 = ind_1+8
            for j in range(ind_1, ind_2):

                img = Image.open(sequence_dataset[j]).convert('RGB').resize((210,140))
                #Debería ser un (140,210,3)
                img = np.array(img)
                # adquiere el (8,140,210,3)
                seq_tensor.append(img)
            seq_tensor = np.array(seq_tensor, dtype='uint8')
            #print(seq_tensor.shape)

            data = [seq_tensor, clase, tipo]
            tensor_5d.append(data)
            #print(len(tensor_5d))
    # tamaño final (N,8,140,210,3)

    return(tensor_5d)

def split_tensor_5d (tensor_5d):
    batch_list = []
    for i in tqdm(range(0, len(tensor_5d)-8, 8), desc = "Splitting tensor 5D: "):
        batch = []
        gt = []
        ind_1 = i
        ind_2 = ind_1+8
        for j in range(ind_1, ind_2):
            batch.append(tensor_5d[j][0])
            gt.append(tensor_5d[j][1])
        batch = np.array(batch, dtype='uint8')
        gt = np.array(gt, dtype='uint8')
        batch_list.append([batch,gt])
    return(batch_list)


# *************** 4.A  TEST FINAL **************

# Control
print('\nControl:')
if data_type == 'iris_2classes':
    clase = [1,0]
    tipo = 1
elif data_type == 'iris_4classes':
    clase = [1,0,0,0]
    tipo = 1
print('CONTROL Val')
tensor_control = make_tensor_5D(folders_control, clase = clase, tipo = tipo)
print(len(tensor_control))

# No control
print('Alcohol:')
if data_type == 'iris_2classes':
    clase = [0,1]
    tipo = 0
elif data_type == 'iris_4classes':
    clase = [0,1,0,0]
    tipo = 0

tensor_alcohol = make_tensor_5D(folders_alcohol, clase = clase, tipo = tipo)
print(len(tensor_alcohol))


print('\nDrug:')
if data_type == 'iris_2classes':
    clase = [0,1]
    tipo = 0
elif data_type == 'iris_4classes':
    clase = [0,0,1,0]
    tipo = 2

tensor_drug = make_tensor_5D(folders_drug, clase = clase, tipo = tipo)
print(len(tensor_drug))

print('\nSleep:')
if data_type == 'iris_2classes':
    clase = [0,1]
    tipo = 0
elif data_type == 'iris_4classes':
    clase = [0,0,0,1]
    tipo = 3

tensor_sleep = make_tensor_5D(folders_sleep, clase = clase, tipo = tipo)
print(len(tensor_sleep))
'''

# *** TEST ***
all_data = tensor_control + tensor_alcohol + tensor_drug + tensor_sleep
np.random.shuffle(all_data)

print('\nForming batches')
#all_batches_test = split_tensor_5d(all_data)

print('\nSaving dataset')
for i,batch in enumerate(tqdm(all_batches_test, desc='Saving npz')):
    file_name = '{}.npz'.format(i)
    file_path = os.path.join(output_dir, file_name)
    name1 = batch[0]
    name2 = batch[1]
    np.savez(file_path, name1 = name1, name2= name2)

print('\nDataset packed successfully!!!\n')
'''