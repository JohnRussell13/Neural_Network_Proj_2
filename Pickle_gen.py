import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

import imgaug
import imgaug.augmenters as iaa

#main directory
DATADIR = "C:/Users/Aleksandar/Desktop/Neural_Network_Proj_2/DATASET"
#subdirectories
CATEGORIES = ["X", "1", "2", "3", "4", "5", "6"]

#main training arrays
X = []
y = []

IMG_SIZE = 100 #image dimensions

#preparation for data augmentation
rotate = iaa.Affine(rotate=(-50, 30))
gaussian_noise = iaa.AdditiveGaussianNoise(10,20)
crop = iaa.Crop(percent=(0, 0.3))
flip_hr = iaa.Fliplr(p=1.0)
flip_vr = iaa.Flipud(p=1.0)
contrast = iaa.GammaContrast(gamma=2.0)

for category in CATEGORIES: #loop subdirectories
    path = os.path.join(DATADIR, category) #create path
    class_num = CATEGORIES.index(category) #turn types(subdirectiories) into numbers

    for img in tqdm(os.listdir(path)): #loop images
        #read image
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) #convert to array
        image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize
        
        #add to main array
        X.append(image) 
        y.append(class_num)
        
        #increase dataset by data augmentation
        #rotation
        rot_img = rotate.augment_image(image)
        X.append(rot_img)
        y.append(class_num)

        #Gaussian noise
        noise_img = gaussian_noise.augment_image(image)
        X.append(noise_img)
        y.append(class_num)

        #crop
        crop_img = crop.augment_image(image)
        X.append(crop_img)
        y.append(class_num)

        #horizontal flip
        fliph_image = flip_hr.augment_image(image)
        X.append(fliph_image)
        y.append(class_num)

        #vertical flip
        flipv_image = flip_vr.augment_image(image)
        X.append(flipv_image)
        y.append(class_num)

        #contrast decrease
        contrast_img = contrast.augment_image(image)
        X.append(contrast_img)
        y.append(class_num)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #turn back to N (-1) IMG_SIZExIMG_SIZEx1 array -  2D picture where every pixel is separated
y = np.array(y)

#input scaling
X = X / 255.0

#pickle it
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()