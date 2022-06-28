import os

epochs = 1000
time = 8
n_classes = 4
width, height, color_channels = (210, 140, 3)
number_of_hiddenunits = 32
batch_size = 16
save_freq = 1

data_type = 'iris_4classes'  #'iris_2classes' # 'cars' # 'iris_4classes' #
model_name = 'vgg' # 'inception' #
version = 'v2'

mode = 'train' #'test' # 'train' #
test_epoch = 387
nsequences=  'all' #1 #'all'

input_video = 'GTA_V_3.mp4' # 'GTA_V_1.mp4' # 'GTA_V_2.mp4' #

#config
model_ID = '{}_{}_{}'.format(data_type, model_name, version)
base_folder = os.path.abspath(os.curdir)
data_path = os.path.join(base_folder,'datasets_h5',data_type)
train_folder = os.path.join(data_path,'train_set')
test_folder = os.path.join(data_path,'test_set')
valid_folder = os.path.join(data_path,'valid_set')
model_save_folder = os.path.join(base_folder,'checkpoints',model_ID)
tensorboard_save_folder = os.path.join(base_folder,'tensorboard',model_ID)
checkpoint_path = os.path.join(model_save_folder,"model_weights_{epoch:03d}.h5")
