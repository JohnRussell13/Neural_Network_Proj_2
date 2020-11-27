#NB == place for improvment

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

#read pickle
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

#rescale data
X = X/255.0

#network parameters
dense_layers = [1]
layer_sizes = [64]
conv_layers = [2]

#loop layers
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time())) #create unique name

            model = Sequential() #sequential network - one input and one output

            #first layer needs input_shape - IMG_SIZExIMG_SIZEx1 (just like in Pickle_gen.py)
            model.add(Conv2D(layer_size, (3, 3), input_shape = X.shape[1:])) #NB convolution use_bias?
            model.add(Activation('relu')) #NB relu activation function; can be added in line above?
            model.add(MaxPooling2D(pool_size = (2, 2))) #NB pooling

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size = (2, 2)))

            model.add(Flatten()) #flattening input - turning 2D image into 1D array

            for _ in range(dense_layer):
                model.add(Dense(layer_size)) #adding neuron layer use_bias?
                model.add(Activation('relu')) #NB relu activation function; can be added in line above?

            model.add(Dense(1)) #output layer
            model.add(Activation('sigmoid')) #NB sigmoid output activation

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) #log files generator

            model.compile(loss='binary_crossentropy', #NB loss calculation
                          optimizer = 'adam', #NB standard optimizer
                          metrics = ['accuracy']) #NB what to optimize

            model.fit(X, y, #training
                      batch_size = 32, #NB number of used training images
                      epochs  = 3, #NB number of cycles
                      validation_split = 0.3) #NB fraction of batch data note used for training - only for validation
                      #callbacks = [tensorboard])  #NB log files not working right now