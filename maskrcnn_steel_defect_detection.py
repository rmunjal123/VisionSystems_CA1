# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 00:31:53 2020

@author: Paul S.R
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

import mrcnn.visualize as visualize
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

import imgaug.augmenters as iaa


# =========================================================================== #
# 1. Define environment configurations
# =========================================================================== #

# ignore UserWarnongs
import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')

print(device_lib.list_local_devices())

WORKING_DIR = os.path.curdir
LOGS_DIR = os.path.join(WORKING_DIR, "logs")

# configure Tessorflow to maximize GPU capability
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# session stuff
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

plt.rcParams.update({'figure.figsize':(8,6), 'figure.dpi':150})

def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

# Set up 'ggplot' style
plt.style.use('ggplot')
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

# =========================================================================== #
# 2. Define Constants, Functions, and Classes
# =========================================================================== #

# Location of input images
input_img_folder = "dataset/all_images"

# Distint class values (numeric)
class_names = ['Blowhole', 'Cavity', 'Crack']


def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels, 
        and turns them into a 3d numpy array of shape (256, 1600, 4) 
    """
    # initialise an empty numpy array 
    mask = np.zeros((512,512,3), dtype=np.uint8)
    
    # building the masks
    for rle, label in zip(encodings, labels):
        
        # classes are [1, 2, 3], corresponding indeces are [0, 1, 2]
        index = label - 1
        
        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for 
        # numpy and openCV handling width and height in reverse order 
        mask[:,:,index] = rle_to_mask(rle).T
    
    return mask


def rle_to_mask(lre, shape=(512,512)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return 
    
    returns: numpy array with dimensions of shape parameter
    '''    
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])
    
    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1
    
    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]
    
    #print("run:",run_starts, run_ends)
    
    # build the mask
    h, w = shape
    mask = np.zeros(h*w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1
    
    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)


def lrSchedule(epoch):
    lr  = 1e-3
    
    if epoch > 100:
        lr  *= 0.5e-3
    elif epoch > 75:
        lr  *= 1e-3        
    elif epoch > 50:
        lr  *= 1e-2        
    elif epoch > 25:
        lr  *= 1e-1
        
    print('Learning rate: ', lr)
    return lr


class SteelDefectConfig(Config):

    # Give the configuration a recognizable name
    NAME = "steel_detection"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 # this defines the batch size in MaskRCNN, max we can go is 2, otherwise OOM error will be thrown

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + metal defects

    # Number of training steps per epoch - 100, 64
    STEPS_PER_EPOCH = 200

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7 # base 0.7
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # Discard inferior model weights
    SAVE_BEST_ONLY = True
    
    # Image resiing parameters. 
    # We don't need image resizing as our input is fixed 512x512 image
    # IMAGE_RESIZE_MODE = "none"
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    
    # Learning rate and momentum
    # LEARNING_RATE = 0.001 # SGD
    LEARNING_RATE = 0.0025 # Adam
    LEARNING_MOMENTUM = 0.9
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    # Number of epochs to run - 100
    EPOCH_SIZE = 150
    
# instantiating 
steel_defect_config = SteelDefectConfig()


# super class can be found here:
# https://github.com/matterport/Mask_RCNN/blob/v2.1/utils.py

class SteelDefectDataset(Dataset):
    
    def __init__(self, dataframe):
        
        # https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
        super().__init__(self)
        
        # needs to be in the format of our squashed df, 
        # i.e. image id and list of rle plus their respective label on a single row
        self.dataframe = dataframe
        
    def load_dataset(self):
        """ takes:
                - pandas df containing 
                    1) file names of our images 
                       (which we will append to the directory to find our images)
                    2) a list of rle for each image 
                       (which will be fed to our build_mask() 
                       function we also used in the eda section)         
            does:
                adds images to the dataset with the utils.Dataset's add_image() metho
        """
        
        # add our four classes
        for i in range(1,4): #range(1,5)
            self.add_class(source='', class_id=i, class_name=f'defect_{i}')
        
        # add the image to our utils.Dataset class
        for index, row in self.dataframe.iterrows():
            file_name = row.ImageId
            file_path = f'{input_img_folder}/{file_name}'
            
            assert os.path.isfile(file_path), 'File doesn\'t exist.'
            self.add_image(source='', 
                           image_id=file_name, 
                           path=file_path)
    
    def load_mask(self, image_id):
        """As found in: 
            https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py
        
        Load instance masks for the given image
        
        This function converts the different mask format to one format
        in the form of a bitmap [height, width, instances]
        
        Returns:
            - masks    : A bool array of shape [height, width, instance count] with
                         one mask per instance
            - class_ids: a 1D array of class IDs of the instance masks
        """
        
        # find the image in the dataframe
        row = self.dataframe.iloc[image_id]
        
        # extract function arguments
        rle = row['EncodedPixels']
        labels = row['ClassId']
        
        # create our numpy array mask
        mask = build_mask(encodings=rle, labels=labels)
        
        # we're actually doing semantic segmentation, so our second return value is a bit awkward
        # we have one layer per class, rather than per instance... so it will always just be 
        # 1, 2, 3. See the section on Data Shapes for the Labels.
        return mask.astype(np.bool), np.array([1, 2, 3], dtype=np.int32)
    

# =========================================================================== #
# Pre-process input mask_to_rle data
# =========================================================================== #

# reading in the training mask_to_rle set
data = pd.read_csv('mask_to_rle.csv')

# isolating the file name and class
data['ClassId'] = data['ClassId'].astype(np.uint8)

# storing a list of images without defects for later use and testing
no_defects = data[data['EncodedPixels'].isna()] \
                [['ImageId']] \
                .drop_duplicates()

# adding the columns so we can append (a sample of) the dataset if need be, later
no_defects['EncodedPixels'] = ''
no_defects['ClassId'] = np.empty((len(no_defects), 0)).tolist()
no_defects['Distinct Defect Types'] = 0
no_defects.reset_index(inplace=True)

# keep only the images with labels
squashed = data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)
#squashed = data

# squash multiple rows per image into a list
squashed = data[['ImageId', 'EncodedPixels', 'ClassId']] \
            .groupby('ImageId', as_index=False) \
            .agg(list) \

# count the amount of class labels per image
squashed['Distinct Defect Types'] = squashed.ClassId.apply(lambda x: len(x))
print(squashed['Distinct Defect Types'])

# display first ten to show new structure
print(squashed.head(10))



# =========================================================================== #
# Split train/test data
# =========================================================================== #

# stratified split to maintain the same class balance in both sets
train, validate = train_test_split(squashed, test_size=0.3, random_state=41)

print("Preparing dataset ............")

# instantiating training set
dataset_train = SteelDefectDataset(dataframe=train)
dataset_train.load_dataset()
dataset_train.prepare()
print('Size of training dataset:', len(dataset_train.image_info))

# instantiating validation set
dataset_validate = SteelDefectDataset(dataframe=validate)
dataset_validate.load_dataset()
dataset_validate.prepare()

print('Size of testing dataset:', len(dataset_validate.image_info))



# =========================================================================== #
# 3. Train the model and plot loss
# =========================================================================== #

# initialiazing model
model = MaskRCNN(mode='training', config=steel_defect_config, model_dir='modeldir')
#model.keras_model.summary()

print("Loading weights ............")

# we will retrain starting with the coco weights
model.load_weights('pretrained_models/severstal/mask_rcnn_severstal_0100.h5', 
                   by_name=True, 
                   exclude=['mrcnn_bbox_fc',
                            'mrcnn_class_logits', 
                            'mrcnn_mask',
                            'mrcnn_bbox'])


print("Training starts ............")

# training at last
epoch_size = steel_defect_config.EPOCH_SIZE

sometimes = lambda aug: iaa.Sometimes(0.3, aug)

imgaug_set = seq = iaa.Sequential([iaa.Fliplr(0.3), # horizontal flips
                                   sometimes(iaa.Crop(percent=(0, 0.1))), # random crops
                                   sometimes(iaa.Affine(
                                        scale={"x": (0.8, 1.1), "y": (0.8, 1.1)},
                                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                        rotate=(-30, 30)
                                    )),
                                   iaa.Grayscale(alpha=(0.0, 0.7))], 
                                  random_order=True)

LRScheduler = LearningRateScheduler(lrSchedule)
callbacks_list  = [LRScheduler]


model.train(dataset_train,
            dataset_validate,
            epochs=epoch_size,
            layers='heads',
            augmentation=imgaug_set,
            custom_callbacks=callbacks_list,
            learning_rate=steel_defect_config.LEARNING_RATE,
            optimizer='Adam')

train_hist = model.keras_model.history

print("Training is completed ............")


# Save loss to csv file
hist_df = pd.DataFrame(train_hist.history) 
with open('history.csv', mode='w') as f:
    hist_df.to_csv(f)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_size), train_hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_size), train_hist.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
plt.xticks([0,20,40,60,80,100,120])
plt.legend(loc="lower left")
plt.savefig("loss_plot.png")
plt.show()



# =========================================================================== #
# 4. Load a test image and do inference
# =========================================================================== #

# define the model for inference
# infer_rcnn = MaskRCNN(mode='inference', config=steel_defect_config, model_dir='modeldir')

# load trained model weights
# infer_rcnn.load_weights('pretrained_models/metal_surface_detection/mask_rcnn_steel_detection_0100.h5', by_name=True)
#infer_rcnn.keras_model.summary()

# load an inference image
# val_blh_img = img_to_array(load_img('dataset/all_images/blowhole_001.jpg'))
# val_cvt_img = img_to_array(load_img('dataset/all_images/cavity_001.jpg'))
# val_crk_img = img_to_array(load_img('dataset/all_images/crack_01.jpg'))

# make prediction
# results = infer_rcnn.detect([img], verbose=1)

# get dictionary for first prediction
# r = results[0]

# show photo with bounding boxes, masks, class labels and scores
# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids']-1, class_names, r['scores'])

