import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 50

img_array = cv2.imread('/content/drive/My Drive/DATA/9.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img_array, cmap='gray')
plt.show()

img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
new_array = cv2.blur(img_array, (4,4))
new_array = cv2.adaptiveThreshold(new_array, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 12)
plt.imshow(new_array, cmap='gray')
plt.show()

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
      rad1 = rad1+1;

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

img = dark_spot(new_array, 6)
avg_x, avg_y, rad = spot_center(img, 6)
if avg_x != -1 and avg_y != -1:
  if avg_x-rad < 0 or avg_y-rad < 0 or avg_x+rad > IMG_SIZE or avg_y+rad > IMG_SIZE:
    crop_img = img_array
  else:
    crop_img = img_array[avg_x-rad:avg_x+rad, avg_y-rad:avg_y+rad]
  final = cv2.adaptiveThreshold(crop_img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 100)
  print(avg_x, avg_y, rad)
  plt.imshow(final, cmap='gray')
  plt.show()