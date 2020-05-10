# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:15:54 2020

@author: rajpaul
"""

import mrcnn.utils as utils
import cv2 as cv2
import os

folders = ['Blowhole', 'Crack']


for folder in folders:

    img_rd_fldr = os.path.join('dataset', folder, 'images') # "dataset/Blowhole/images" 
    mask_rd_fldr = os.path.join('dataset', folder, 'masks') # "dataset/Blowhole/masks"
    
    img_wrt_fldr = os.path.join('dataset\\resized', folder, 'images') # "dataset/resized/Blowhole/images"
    mask_wrt_fldr = os.path.join('dataset\\resized', folder, 'masks') # "dataset/resized/Blowhole/masks"
    
    print(img_rd_fldr)
    print(img_wrt_fldr)
    print(mask_rd_fldr)
    print(mask_wrt_fldr)

    for filename in os.listdir(img_rd_fldr):
            print(filename)
            ##---- Image resize ----##
            # Read image        
            img = cv2.imread(os.path.join(img_rd_fldr, filename))
            #print('img:', img.shape)
            # resize image
            (image, window, scale, padding, crop) = utils.resize_image(img, max_dim=512, mode="square")
            #print('rz img:', image.shape)
            # Write resized image
            cv2.imwrite(os.path.join(img_wrt_fldr, filename), image)
            
            
            ##---- Mask resize ----##
            # change name
            maskname = filename.split('.')[0] + '.png'
            # Read mask
            msk = cv2.imread(os.path.join(mask_rd_fldr, maskname))
            #print('mask:', msk.shape)
            # resize mask
            mask = utils.resize_mask(msk, scale, padding)
            #print('rz mask:', mask.shape)
            # Write resized mask
            cv2.imwrite(os.path.join(mask_wrt_fldr, maskname), mask)
            
            #break
    