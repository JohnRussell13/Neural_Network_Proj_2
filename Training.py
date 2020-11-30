#NB == place for improvment
#NW == not working right now
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import pickle
#NW import time

CATEGORIES = ["X", "1", "2", "3", "4", "5", "6"]
CLASS_SIZE = np.array(CATEGORIES).shape[0]

#read pickle
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

#manual data shuffle - X and y must preserve equal order
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

#network parameters
dense_layers = [1]
layer_sizes = [64]
conv_layers = [2]

#loop layers
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            #NW NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time())) #create unique name

            model = Sequential() #sequential network - one input and one output

            #first layer needs input_shape - IMG_SIZExIMG_SIZEx1 (just like in Pickle_gen.py)
            model.add(Conv2D(layer_size, (3, 3), input_shape = X.shape[1:], use_bias = False, activation = 'relu')) #NB convolution #NB relu activation function; ReLU is very good for FPGA; can be added in line above? (not important)
            model.add(MaxPooling2D(pool_size = (2, 2))) #NB pooling
            model.add(BatchNormalization()) #NB batchnorm (goes after activation) - reduce epoch number \ dropout - dropping nodes

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3), use_bias = False, activation = 'relu'))
                model.add(MaxPooling2D(pool_size = (2, 2)))
                model.add(BatchNormalization())

            model.add(Flatten()) #flattening input - turning 2D image into 1D array

            for _ in range(dense_layer):
                model.add(Dense(layer_size, use_bias = False, activation = 'relu')) #adding neuron layer use_bias?
                model.add(BatchNormalization())

            model.add(Dense(CLASS_SIZE, use_bias = False, activation = 'softmax')) #output layer #not sigmoid output activation -- relu is better for FPGA!

            #NW tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) #log files generator

            model.compile(loss='sparse_categorical_crossentropy', #NB loss calculation
                          optimizer = 'adam', #NB standard optimizer
                          metrics = ['accuracy']) #NB what to optimize

            model.fit(X, y, #training
                      batch_size = 32, #NB number of used training images
                      epochs  = 3, #NB number of cycles
                      shuffle = True, #shuffle data
                      validation_split = 0.1) #NB fraction of batch data note used for training - only for validation
                      #NW callbacks = [tensorboard])  #NB log files

model.save('Neural_Network_Proj_2.model')