#1
def convert(x,whole,fract):
  if x >= 0:
    x = x % (2 ** (whole - 1))
  else:
    x = - ((-x) % (2 ** (whole - 1)))
  return int(x * (2 ** fract))

def aconvert(x,whole,fract):
  return x / 2 ** fract

def trans(x,whole,fract):
  return aconvert(convert(x,whole,fract),whole,fract)

#2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 256.0
x_test = x_test / 256.0

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
          epochs  = 1, #NB number of cycles
          shuffle = True, #shuffle data
          validation_split = 0.2) #NB fraction of batch data note used for training - only for validation - 0.1 seems optimal

#3
i = 0
weights = []
for layer in model.layers:
  if (i == 0 or i == 4): #layers with weights
    weights.append(layer.get_weights()[0])
  i = i + 1

print(np.shape(weights[0]))
print(np.shape(weights[1]))

#4
def conv(size, n, m, p):
  for i in range(size - 2):
    for j in range(size - 2):
      for k in range(3):
        for l in range(3):
          p[i][j] =  p[i][j] + n[i + k][j + l] * m[k][l]
          
def conv_Q(size, n, m, p, WHOLE, FRACT):
  for i in range(size - 2):
    for j in range(size - 2):
      for k in range(3):
        for l in range(3):
          p[i][j] = trans(p[i][j] + trans(n[i + k][j + l] * m[k][l],WHOLE,FRACT),WHOLE,FRACT)
        
      if p[i][j] < 0:
        p[i][j] = 0
	
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

def dense_Q(size, f, size2, y, w, WHOLE, FRACT):
  for i in range(size2):
    for j in range(size):
      y[i] = trans(y[i] + trans(f[j] * w[j][i],WHOLE,FRACT),WHOLE,FRACT)

#5
WHOLE = 1+4
FRAC = 7

#CONV WEIGHTS
CONV_W = []
CONV_W_Q = []
for k in range(1):
  for l in range(C_SIZE):#32
    for i in range(3):
      for j in range(3):
        CONV_W.append(weights[0][i][j][k][l])
        CONV_W_Q.append(0.0)
CONV_W = np.array(CONV_W).reshape(C_SIZE, 3, 3) #3 x 3 kernel
CONV_W_Q = np.array(CONV_W_Q).reshape(C_SIZE, 3, 3) #3 x 3 kernel

for i in range(C_SIZE):
  for j in range(3):
    for k in range(3):
      CONV_W_Q[i][j][k] = trans(CONV_W[i][j][k], WHOLE, FRAC)

#DENSE WEIGHTS
DENSE_W = weights[1]
DENSE_W_Q = weights[1]
for i in range(C_SIZE*int((IMG_SIZE-2)/2)*int((IMG_SIZE-2)/2)):
  for j in range(10):
    DENSE_W_Q[i][j] = trans(DENSE_W[i][j], WHOLE, FRAC)

#START TEST
MISS = 0
for r in range(100):
  X = x_test[r].reshape(IMG_SIZE,IMG_SIZE)
  X_Q = []
  for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
      X_Q.append(trans(X[i][j],WHOLE,FRAC))
  X_Q = np.array(X_Q).reshape(IMG_SIZE, IMG_SIZE)


  #INIT CONV RESULT
  CONV_RES = []
  CONV_RES_Q = []
  for i in range(C_SIZE*(IMG_SIZE-2)*(IMG_SIZE-2)):
    CONV_RES.append(0.0)
    CONV_RES_Q.append(0.0)
  CONV_RES = np.array(CONV_RES).reshape(C_SIZE, IMG_SIZE-2, IMG_SIZE-2)
  CONV_RES_Q = np.array(CONV_RES_Q).reshape(C_SIZE, IMG_SIZE-2, IMG_SIZE-2)

  #INIT POOL RESULT
  POOL_RES = []
  POOL_RES_Q = []
  for i in range(C_SIZE*int((IMG_SIZE-2)/2)*int((IMG_SIZE-2)/2)):
    POOL_RES.append(0.0)
    POOL_RES_Q.append(0.0)
  POOL_RES = np.array(POOL_RES).reshape(C_SIZE, int((IMG_SIZE-2)/2), int((IMG_SIZE-2)/2))
  POOL_RES_Q = np.array(POOL_RES_Q).reshape(C_SIZE, int((IMG_SIZE-2)/2), int((IMG_SIZE-2)/2))

  #INIT FLATTENING RESULT
  FLAT = []
  FLAT_Q = []

  #INIT DENSE RESULT
  FINAL = []
  FINAL_Q = []
  for i in range(10):
    FINAL.append(0.0)
    FINAL_Q.append(0.0)

  #2D LAYERS
  for i in range(C_SIZE):
    conv_Q(IMG_SIZE, X, CONV_W_Q[i], CONV_RES_Q[i], WHOLE, FRAC)
    pooling(int((IMG_SIZE-2)/2), CONV_RES_Q[i], POOL_RES_Q[i])

  flatten(int((IMG_SIZE-2)/2), C_SIZE, POOL_RES_Q, FLAT_Q)
  dense_Q(int((IMG_SIZE-2)/2)*int((IMG_SIZE-2)/2)*C_SIZE, FLAT_Q, 10, FINAL_Q, DENSE_W_Q, WHOLE, FRAC)

  print(r-MISS+1, "/", r+1)

  #CHECK
  if np.argmax(FINAL_Q) != np.argmax(model.predict(X.reshape(1,IMG_SIZE,IMG_SIZE,1))):
    #print(FINAL)
    #print(FINAL_Q)
    #print(model.predict(X.reshape(1,IMG_SIZE,IMG_SIZE,1)))
    MISS = MISS + 1
  
  #PRINT
  #print(np.argmax(FINAL))
  #print(FINAL)
  #print(FINAL_Q)
  #print(model.predict(X.reshape(1,IMG_SIZE,IMG_SIZE,1)))

#PRINT
#print(MISS)