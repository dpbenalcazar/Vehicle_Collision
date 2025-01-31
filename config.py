import os

epochs = 1000
time = 8
n_classes = 2
width, height, color_channels = (210, 140, 3)
number_of_hiddenunits = 32
batch_size = 24
save_freq = 1

data_type = 'AC' # 'iris_4classes' #   #'iris_2classes' # 'cars' # AC (alcohol, control)
model_name = 'vgg' # 'inception' #
vers = 'v1'

mode = 'test' #'test' # 'train' #
test_epoch = 992
nsequences=  1 #1 #'all'

input_video = 'GTA_V_3.mp4' # 'GTA_V_1.mp4' # 'GTA_V_2.mp4' #

#config
model_ID = '{}_{}_{}'.format(data_type, model_name, vers)
base_folder = os.path.abspath(os.curdir)
data_path = os.path.join(base_folder,'datasets_h5',data_type)
train_folder = os.path.join(data_path,'train_set')
test_folder = os.path.join(data_path,'test_set')
valid_folder = os.path.join(data_path,'valid_set')
model_save_folder = os.path.join(base_folder,'checkpoints',model_ID)
tensorboard_save_folder = os.path.join(base_folder,'tensorboard',model_ID)
checkpoint_path = os.path.join(model_save_folder,"model_weights_{epoch:03d}.h5")
