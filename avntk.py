import os
import tensorflow as tf
import cv2
import numpy as np
import json
import shutil
from collections import deque
from model_architecture import build_tools
from utils.utils import data_tools
from utils.FPS import FPS
from config import *

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


# Create training folders
if mode == 'train':
    # Create checkpoint's folder
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    else:
        shutil.rmtree(model_save_folder)
        os.makedirs(model_save_folder)

    # Create tensorboard folder
    if not os.path.exists(tensorboard_save_folder):
        os.makedirs(tensorboard_save_folder)
    else:
        shutil.rmtree(tensorboard_save_folder)
        os.makedirs(tensorboard_save_folder)

# Callback for saving checkpoints
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                 save_weights_only=True, period=save_freq)

# Tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_folder, histogram_freq=0, write_graph=True,
                                                      write_images=False)

# Model trainier
def _trainer(network,train_generator,val_generator):
    network.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])
    network.save_weights(checkpoint_path.format(epoch=0))
    steps_per_epoch = len(os.listdir(train_folder)) // batch_size
    history =network.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                   validation_data=val_generator, validation_steps=1,
                                   callbacks=[cp_callback, tensorboard_callback]
                                   )
    with open(os.path.join(model_save_folder, 'training_logs.json'),'w') as w:
        json.dump(history.history,w)

# Inference over an input video
def inference(network, video_file):
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter(os.path.join(base_folder,'files','inference_video.mp4'),fourcc, 12.0, (width*4,height*4))
    cv2.namedWindow("CNN LSTMN Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CNN LSTMN Inference", 840,560) # 1440,810
    print("\nProcessing Frames ...")
    image_seq = deque([],8)
    cap = cv2.VideoCapture(video_file)
    counter = 0
    stat = 'safe'
    fps = FPS().start()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            _frame = cv2.resize(frame,(width,height))
            image_seq.append(_frame)
            if len(image_seq)==8:
                np_image_seqs = np.reshape(np.array(image_seq)/255,(1,time,height,width,color_channels))
                r = network.predict(np_image_seqs)
                stat = ['safe','collision'][np.argmax(r,1)[0]]

            cv2.putText(frame,stat, (230,230), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),3)
            #out.write(frame)
            cv2.imshow("CNN LSTMN Inference",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter+=1
            fps.update()
            print('Frame: {:03d} , Prediction: {}'.format(counter, stat))
        
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    fps.stop()
    print('\nSpeed Test Results:')
    print('  -Elapsed time: {:0.2f} s'.format(fps.elapsed()))
    print('  -Speed: {:0.2f} frames per second'.format(fps.fps()))
    print('  -Speed: {:0.2f} seconds per frame'.format(1/fps.fps()))
    return


if  __name__ == "__main__":

    # Create model
    model_tools = build_tools()
    network = model_tools.create_network(model_name)
    print('\n--Network created successfully--\n')

    if mode == 'train':
        # Read train and validation datasets
        train_generator = data_tools(train_folder,'train')
        valid_generator = data_tools(valid_folder,'valid')

        # Train the network
        _trainer(network, train_generator.batch_dispatch(), valid_generator.batch_dispatch())

    else:
        # Load weights
        network.load_weights(checkpoint_path.format(epoch=test_epoch))
        print('\n--Weights loaded successfully--\n')

        # Save Model
        #network.save('model_save_folder/model_weights_{:03d}.h5'.format(test_epoch))

        # Process input video
        inference(network, os.path.join(base_folder,'files/samples', input_video))

        #testing from batch
        '''
        test_generator = get_valid_data(test_folder)
        for img_seq,labels in test_generator:
            r = network.predict(img_seq)
            print ('accuracy',np.count_nonzero(np.argmax(r,1)==np.argmax(labels,1))/8)
        '''

    print('\n\nSuccessful {}!!! XD\n'.format(mode))
