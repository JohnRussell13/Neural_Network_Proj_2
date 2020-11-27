import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

#main directory
DATADIR = "C:/Users/Aleksandar/Desktop/Neural_Network_Proj_2/DATASET"
#subdirectories
CATEGORIES = ["X", "1", "2", "3", "4", "5", "6"]

#main training arrays
X = []
y = []

IMG_SIZE = 100 #image dimensions

for category in CATEGORIES: #loop subdirectories
    path = os.path.join(DATADIR, category) #create path
    class_num = CATEGORIES.index(category) #turn types(subdirectiories) into numbers

    for img in tqdm(os.listdir(path)): #loop images
        try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) #convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize
            #add to main array
            X.append(new_array) 
            y.append(class_num)
        except Exception as e: #not really important
            pass
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #turn back to N (-1) IMG_SIZExIMG_SIZEx1 array -  2D picture where every pixel is separated

#pickle it
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()