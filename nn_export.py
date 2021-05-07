from tensorflow.keras.models import Sequential
#1
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

IMG_SIZE = 28
C_SIZE = 32

X_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = np.arange(0,IMG_SIZE*IMG_SIZE).reshape(1,IMG_SIZE,IMG_SIZE,1)

model = Sequential() #sequential network - one input and one output

model.add(Conv2D(C_SIZE, (3, 3), use_bias=False, input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(10, use_bias = False, activation = 'softmax')) #output layer #not sigmoid output activation -- relu is better for FPGA!

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, #training
          batch_size = 32, #NB number of used training images 32
          epochs  = 5, #NB number of cycles
          shuffle = True, #shuffle data
          validation_split = 0.2) #NB fraction of batch data note used for training - only for validation - 0.1 seems optimal

#2
i = 0
weights = []
for layer in model.layers:
  if (i == 0 or i == 4): #layers with weights
    weights.append(layer.get_weights()[0])
  i = i + 1

#3
def conv(size, n, m, p):
  for i in range(size - 2):
    for j in range(size - 2):
      for k in range(3):
        for l in range(3):
          p[i][j] =  p[i][j] + n[i + k][j + l] * m[k][l]
	
def pooling(size, n, p):
  for i in range(size):
    for j in range(size):
      if n[2*i][2*j] > n[2*i][2*j+1]:
        temp = n[2*i][2*j]
      else:
        temp = n[2*i][2*j+1]
      if n[2*i+1][2*j] > n[2*i+1][2*j+1]:
        if temp < n[2*i+1][2*j]:
          temp = n[2*i+1][2*j]
      else:
        if temp < n[2*i+1][2*j+1]:
          temp = n[2*i+1][2*j+1]
      p[i][j] = p[i][j] + temp

def relum(size, n):
  for i in range(size):
    for j in range(size):
      if n[i][j] < 0:
        n[i][j] = 0

def flatten(size, nums, n, f):
  for i in range(size):
    for j in range(size):
      for k in range(nums):
        f.append(n[k][i][j])

def dense(size, f, size2, y, w):
  for i in range(size2):
    for j in range(size):
      y[i] = y[i] + f[j]*w[j][i]

#4
#CONV WEIGHTS
CONV_W = []
for k in range(1):
  for l in range(C_SIZE):#32
    for i in range(3):
      for j in range(3):
        CONV_W.append(weights[0][i][j][k][l])
CONV_W = np.array(CONV_W).reshape(C_SIZE, 3, 3) #3 x 3 kernel

#DENSE WEIGHTS
DENSE_W = weights[1]

#START TEST
MISS = 0
for r in range(10):
  X = x_test[r].reshape(IMG_SIZE,IMG_SIZE)

  #INIT CONV RESULT
  CONV_RES = []
  for i in range(C_SIZE*(IMG_SIZE-2)*(IMG_SIZE-2)):
    CONV_RES.append(0.0)
  CONV_RES = np.array(CONV_RES).reshape(C_SIZE, IMG_SIZE-2, IMG_SIZE-2)

  #INIT POOL RESULT
  POOL_RES = []
  for i in range(C_SIZE*int((IMG_SIZE-2)/2)*int((IMG_SIZE-2)/2)):
    POOL_RES.append(0.0)
  POOL_RES = np.array(POOL_RES).reshape(C_SIZE, int((IMG_SIZE-2)/2), int((IMG_SIZE-2)/2))

  #INIT FLATTENING RESULT
  FLAT = []

  #INIT DENSE RESULT
  FINAL = []
  for i in range(10):
    FINAL.append(0.0)

  #2D LAYERS
  for i in range(C_SIZE):
    conv(IMG_SIZE, X, CONV_W[i], CONV_RES[i])
    pooling(int((IMG_SIZE-2)/2), CONV_RES[i], POOL_RES[i])
    relum(int((IMG_SIZE-2)/2), POOL_RES[i])

  #FLAT LAYERS
  flatten(int((IMG_SIZE-2)/2), C_SIZE, POOL_RES, FLAT)
  dense(int((IMG_SIZE-2)/2)*int((IMG_SIZE-2)/2)*C_SIZE, FLAT, 10, FINAL, DENSE_W)

  #CHECK
  if np.argmax(FINAL) != np.argmax(model.predict(X.reshape(1,IMG_SIZE,IMG_SIZE,1))):
    #print(FINAL)
    #print(model.predict(X.reshape(1,IMG_SIZE,IMG_SIZE,1)))
    MISS = MISS + 1
  
  #PRINT
  print(r-MISS+1, '/', r+1)