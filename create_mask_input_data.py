# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:47:00 2020

@author: rajpaul
"""
import cv2 as cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random

def load_images_from_folder(folder):
    images = []
    names = []
    encodings = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            names.append(filename)
            #print(filename)

            #x = np.array(img, dtype=np.uint8).reshape(img.size[::-1])
            #x = np.array(img, dtype=np.uint8)
            #print(x.shape)
            
            x = img // 255
            
            encodings.append(rle_encoding(x))
            
            #x = mask_to_rle(img)
            #encodings.append(rle_encoding(x))
            
    return images, names, encodings


def rle_to_mask(lre, shape): #shape=(1600,256)
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
    
    print("run:",run_starts, run_ends)
    
    # build the mask
    h, w = shape
    mask = np.zeros(h*w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1
    
    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def array_to_rle_str(arr):
    return str(arr).replace('[','').replace(']','').replace(',','')


def build_mask(encodings, labels, shapes):
    """ takes a pair of lists of encodings and labels, 
        and turns them into a 3d numpy array of shape (256, 1600, 4) 
    """
    
    # building the masks
    for rle, label, shape in zip(encodings, labels, shapes):
        
        h,w,d = shape
        # initialise an empty numpy array 
        mask = np.zeros((h,w,3), dtype=np.uint8)        
        
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label - 1
        
        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for 
        # numpy and openCV handling width and height in reverse order 
        mask[:,:,index] = rle_to_mask(rle,(w,h)).T
    
    return mask


def mask_to_contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """
    
    mask_layer=mask_layer.astype(np.uint8)
    
    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    im2, contours, hierarchy = cv2.findContours(mask_layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0,0,255), 2, cv2.LINE_AA) #color
        
    return image


def visualise_mask(img, mask):
    """ open an image and draws clear masks, so we don't lose sight of the 
        interesting features hiding underneath 
    """
    # reading in the image
    #image = cv.imread("train_images/"+file_name)

    # going through the 4 layers in the last dimension 
    # of our mask with shape (256, 1600, 4)
    image = img
    # print(image.shape)
    # print(mask.shape)
    # print(mask.shape[-1])
    
    for index in range(mask.shape[-1]):
        
        # indeces are [0, 1, 2, 3], corresponding classes are [1, 2, 3, 4]
        label = index + 1
    
        #print(label)
        mask_ = mask[:,:,index]
        print(mask_.shape)
    
        # add the contours, layer per layer 
        image = mask_to_contours(image, mask_, color=sns.color_palette()[label])   
        print("done")
        
    return image


def write_mask_to_csv(path_arr):
    lines = []        
    for p in range(len(path_arr)):
        (images, names, encodings) = load_images_from_folder(path_arr[p])
        
        for i in range(len(images)):
            print(names[i])
            lines.append([names[i].replace('png','jpg') , str(p+1), array_to_rle_str(encodings[i])])
            
    print("Loading is Done!!!")

    random.shuffle(lines)
    print("Shuffling is Done!!!")
        
    with open('mask_to_rle.csv', 'w', newline='') as csvfile:
        rle_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=',')
        rle_writer.writerow(['ImageId','ClassId', 'EncodedPixels'])    
        rle_writer.writerows(lines)
        
    print("Writing is Done!!!")


paths = ['C:\\NUS_ISS_MTech\\Year 2\\Semester 1\\1. Vision systems (VSE)\\CA 1\\dataset\\Blowhole\\masks',
         'C:\\NUS_ISS_MTech\\Year 2\\Semester 1\\1. Vision systems (VSE)\\CA 1\\dataset\\Cavity\\masks',
         'C:\\NUS_ISS_MTech\\Year 2\\Semester 1\\1. Vision systems (VSE)\\CA 1\\dataset\\Crack\\masks']

# Create rle csv file
write_mask_to_csv(paths)



### ---------------------------------------------------------------------------
### Calcualte and display Masks on original
### ---------------------------------------------------------------------------

i = 69
(images, names, encodings) = load_images_from_folder('C:\\NUS_ISS_MTech\\Year 2\\Semester 1\\1. Vision systems (VSE)\\CA 1\\dataset\\Crack\\\masks')
print(len(images))
print(len(names))
print(len(encodings)) 
print(images[i].shape, names[i], encodings[i]) # h,w,c


## --- show mask image ------------------
plt.figure() 
plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) / 255)
plt.axis('off')
plt.show()


## --- show original image ------------------
orig = cv2.imread('C:\\NUS_ISS_MTech\\Year 2\\Semester 1\\1. Vision systems (VSE)\\CA 1\\dataset\\Crack\\\images\\crack_'+str(i+1)+'.jpg')
print(orig.shape)
plt.figure() 
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

## --- build rle ------------------
rle_str = array_to_rle_str(encodings[i])

## --- build mask ------------------
mask = build_mask([rle_str], [3], [images[i].shape])
print(mask.shape)

## --- visualize mask ------------------
out = visualise_mask(orig, mask)
plt.figure() 
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

### ---------------------------------------------------------------------------


