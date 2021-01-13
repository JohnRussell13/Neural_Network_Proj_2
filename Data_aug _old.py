#prep data
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import cv2

IMG_SIZE = 100 #image dimensions
A = 11 #constant
B = 12 #radius size

#preparation for data augmentation
rotate90 = iaa.Affine(rotate=(0, 90))
rotate180 = iaa.Affine(rotate=(0, 180))
rotate270 = iaa.Affine(rotate=(0, 270))
gaussian_noise = iaa.AdditiveGaussianNoise(10,20)
flip_hr = iaa.Fliplr(p=1.0)
flip_vr = iaa.Flipud(p=1.0)
contrast = iaa.GammaContrast(gamma=2.0)

X = []
y = []
for i in np.arange(np.array(x).shape[0]):
  image = x[i]
  class_num = Y[i]

  print(i)

  img = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #increase dataset by data augmentation
  #Gaussian noise
  img = gaussian_noise.augment_image(image)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #rotation
  img = rotate90.augment_image(image)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  img = rotate180.augment_image(image)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  img = rotate270.augment_image(image)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #horizontal flip
  img = flip_hr.augment_image(image)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #vertical flip
  img = flip_vr.augment_image(image)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #contrast decrease
  img_c = contrast.augment_image(image)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)
  img = gaussian_noise.augment_image(img_c)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #rotation
  img = rotate90.augment_image(img_c)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  img = rotate180.augment_image(img_c)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  img = rotate270.augment_image(img_c)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #horizontal flip
  img = flip_hr.augment_image(img_c)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

  #vertical flip
  img = flip_vr.augment_image(img_c)
  img = cv2.blur(img, (4,4))
  img_r = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img_r)
  y.append(class_num)
  img = gaussian_noise.augment_image(img)
  img = cv2.blur(img, (4,4))
  img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  X.append(img)
  y.append(class_num)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #turn back to N (-1) IMG_SIZExIMG_SIZEx1 array -  2D picture where every pixel is separated
y = np.array(y)