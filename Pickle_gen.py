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

def col_conv(img):
  a, b, c = [1,1,1]
  new_img = np.zeros((IMG_SIZE,IMG_SIZE))
  for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
      a = int(img[i][j][0])
      b = int(img[i][j][1])
      c = int(img[i][j][2])
      a = a*a
      b = b*b
      c = c*c
      if a > 0 and b > 0 and c > 0:
        new_img[i][j] = int(np.sqrt(3/(1/a+1/b+1/c)))
  return new_img

for category in CATEGORIES: #loop subdirectories
    path = os.path.join(DATADIR, category) #create path
    class_num = CATEGORIES.index(category) #turn types(subdirectiories) into numbers

    for img in tqdm(os.listdir(path)): #loop images
        #read image
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR ) #convert to array

        image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize
        image = col_conv(image)
        image = np.ubyte(image)
        #add to main array
        X.append(image) 
        y.append(class_num)
#
#X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #turn back to N (-1) IMG_SIZExIMG_SIZEx1 array -  2D picture where every pixel is separated
#y = np.array(y)
#
#input scaling
#X = X / 255.0

#pickle it
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()