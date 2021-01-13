#prep data
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import cv2

IMG_SIZE = 100 #image dimensions
A = 11 #constant
B = 100 #radius size for threshold
C = 6 #radius of patch finder

#isolate the dark patch
def dark_spot(image, radius):
  b, a = image.shape
  b = b - 2*radius
  a = a - 2*radius
  new_image = np.zeros((b,a))
  for i in range(b):
    for j in range(a):
      new_image[i][j] = np.sum(image[i-radius:i+radius, j-radius:j+radius] / ((2*radius+1)*(2*radius+1)))
      if new_image[i][j] > .6:
        new_image[i][j] = 1
      else:
        new_image[i][j] = 0
  return new_image

#find coordinates and radius of the dark patch
def spot_center(image, radius):
  b, a = image.shape
  avg_h = np.zeros(a)
  avg_v = np.zeros(b)
  avg_h_n = []
  avg_v_n = []

  rad1 = 0
  k = np.zeros(a)
  temp = np.array([0,0])
  for i in np.arange(b)[radius:]:
    for j in np.arange(a)[radius:]:
      avg_h[i-radius] = avg_h[i-radius] + (1-image[i][j])*j
      k[i-radius] = k[i-radius] + (1-image[i][j])
    if k[i-radius] != 0:
      avg_h_n.append(avg_h[i-radius] / k[i-radius])
      rad1 = rad1+1

  if rad1 != 0:
    res1 = int(np.sum(np.array(avg_h_n))/rad1)
  else:
    res1 = -1

  rad2 = 0
  k = np.zeros(b)
  temp = np.array([0,0])
  for i in np.arange(a)[radius:]:
    for j in np.arange(b)[radius:]:
      avg_v[i-radius] = avg_v[i-radius] + (1-image[j][i])*j
      k[i-radius] = k[i-radius] + (1-image[j][i])
    if k[i-radius] != 0:
      avg_v_n.append(avg_v[i-radius] / k[i-radius])
      rad2 = rad2+1;

  if rad2 != 0:
    res2 = int(np.sum(np.array(avg_v_n))/rad2)
  else:
    res2 = -1
  
  if rad1 > rad2:
    rad = rad1
  else:
    rad = rad2

  return [res1, res2, rad]

def prep_finish(class_num, img_array, params):
  avg_x = params[0]
  avg_y = params[1]
  rad = params[2]
  if avg_x != -1 and avg_y != -1:
    if avg_x-rad < 0 or avg_y-rad < 0 or avg_x+rad > IMG_SIZE or avg_y+rad > IMG_SIZE:
      crop_img = img_array
    else:
      crop_img = img_array[avg_x-rad:avg_x+rad, avg_y-rad:avg_y+rad]
      crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
    final = cv2.adaptiveThreshold(crop_img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, A, B)
  else:
    final = np.zeros(img_array.shape)
  X.append(final)
  y.append(class_num)
  return final

def full_pack(class_num, img, C):
  img_ds = dark_spot(img, C)
  params = spot_center(img_ds, C)
  final1 = prep_finish(class_num, img, params)
  img_2 = gaussian_noise.augment_image(img)
  img_ds = dark_spot(img_2, C)
  params = spot_center(img_ds, C)
  final2 = prep_finish(class_num, img_2, params)
  return [final1, final2]

def half_pack(img):
  X.append(img[0])
  y.append(class_num)
  X.append(img[1])
  y.append(class_num)


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
img = []
for i in np.arange(np.array(x).shape[0]):
  image = x[i]
  class_num = Y[i]
  print(i)
  print(np.array(X).shape)

  #increase dataset by data augmentation
  image_fp = full_pack(class_num, image, C)

  #rotation
  img.append(rotate90.augment_image(image_fp[0]))
  img.append(rotate90.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  img.append(rotate180.augment_image(image_fp[0]))
  img.append(rotate180.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  img.append(rotate270.augment_image(image_fp[0]))
  img.append(rotate270.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  #horizontal flip
  img.append(flip_hr.augment_image(image_fp[0]))
  img.append(flip_hr.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  #vertical flip
  img.append(flip_vr.augment_image(image_fp[0]))
  img.append(flip_vr.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  #contrast decrease
  image = contrast.augment_image(image)
  image_fp = full_pack(class_num, image, C)

  #rotation
  img.append(rotate90.augment_image(image_fp[0]))
  img.append(rotate90.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  img.append(rotate180.augment_image(image_fp[0]))
  img.append(rotate180.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  img.append(rotate270.augment_image(image_fp[0]))
  img.append(rotate270.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  #horizontal flip
  img.append(flip_hr.augment_image(image_fp[0]))
  img.append(flip_hr.augment_image(image_fp[1]))
  half_pack(img)
  img = []

  #vertical flip
  img.append(flip_vr.augment_image(image_fp[0]))
  img.append(flip_vr.augment_image(image_fp[1]))
  half_pack(img)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #turn back to N (-1) IMG_SIZExIMG_SIZEx1 array -  2D picture where every pixel is separated
y = np.array(y)