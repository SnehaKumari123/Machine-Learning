# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QXivR9io57532DtjX6W-DuZo-zh-oEao
"""

# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z7_SoNg1sgarRNQ1XqWB4aZoUU3F0Q20
"""

#importing required libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import  Dense, Dropout, Flatten
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
import tensorflow   as tf
import matplotlib.pyplot as plt # use for Crate the Plot
import os
import numpy as np

#creating a class for keras callback
class PlotLosses(keras.callbacks.Callback):

    # Flag to enable or disable the Display the Graph
    display_graph = False
    
    #Calls the methods of its callbacks.(logs: dict, currently no data is passed to this argument
    #for this method but that may change in the future.)
    def on_train_begin(self, logs={}):

        self.epoch = []
        self.accuracy = []
        self.val_accuracy = []
        self.val_loss = []
        self.losses = []
        self.fig = plt.figure(figsize=(9, 6), facecolor='w', edgecolor='k')
        self.logs = []
    #Called at the end of an epoch.(Subclasses should override for any actions to run.
    #This function should only be called during train mode.)
    
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.epoch.append(epoch+1)
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))

        print ("epoch %s logs %s "%(epoch+1,logs))    
        
    #Plot the graph with training error and test error

    def display(self,title,TR_Accuracy,TE_Accuracy,TR_Loss,TE_Loss):

        ax = plt.subplot (111)
        plt.text(1, 0.90,'Training Accuracy  = %8.4f '%(TR_Accuracy), horizontalalignment='right', verticalalignment='top',transform = ax.transAxes)
        plt.text(1, 0.87,'Test Accuracy      = %8.4f '%(TE_Accuracy)    , horizontalalignment='right', verticalalignment='top',transform = ax.transAxes)
        plt.text(1, 0.84,'Training Loss      = %8.4f '%(TR_Loss), horizontalalignment='right', verticalalignment='top',transform = ax.transAxes)
        plt.text(1, 0.81,'Test Loss          = %8.4f '%(TE_Loss)    , horizontalalignment='right', verticalalignment='top',transform = ax.transAxes)
    
        plt.title (title)
        plt.suptitle ("Accuracy and Loss")
        plt.xlabel ('Epoch')
        plt.ylabel ('Accuracy / Loss')
        plt.plot(self.epoch, self.accuracy      , label="Training Accuracy")
        plt.plot(self.epoch, self.val_accuracy  , label="Test Accuracy")
        plt.plot(self.epoch, self.losses        , label="Training Loss")
        plt.plot(self.epoch, self.val_loss      , label="Test Loss")
        plt.legend()

        # if display graph true then display otherwise save to the file located at the output path
        if (self.display_graph): plt.show ()
        else:
          if not os.path.exists('images'): os.makedirs('images')      
          FileRootImg = os.path.abspath('images') + '/' + title.strip() 
          print (FileRootImg)   
          plt.savefig (FileRootImg)

# creating object of the class
plot_losses = PlotLosses()
# Assigning values for batch size, epochs and image rows and columns
batch_size=128 
num_classes=10
epochs=12
img_rows, img_cols = 28, 28


# split the the data between train and test sets

#DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
#path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Default image data format convention ('channels_first' or 'channels_last')
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#Reshaping and Normalizing the MNIST Images
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#Printing the shape of train and test data samples
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Coding for baseline

print('Baseline')
model_Bl = Sequential()

#The first hidden layer is a convolutional layer, with 6 feature maps. The convolution kernels are of 3x3 in size. Use stride 1 for convolution.

model_Bl.add(Conv2D(6, kernel_size=(3, 3),activation='relu',input_shape=input_shape,strides=(1,1)))

#The convolutional layer is followed by a max pooling layer. The pooling is 2x2 with stride 1.

model_Bl.add (MaxPooling2D (pool_size=(2,2),strides=(1,1)))

#After max pooling, the layer is connected to the next convolutional layer, with 16 feature maps. 
#The convolution kernels are of 3x3 in size. Use stride 1 for convolution.

model_Bl.add(Conv2D(16, (3, 3), activation='relu',strides=(1,1)))

#The second convolutional layer is followed by a max pooling layer. The pooling is 2x2 with stride 1.
model_Bl.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))

#After max pooling, the layer is fully connected to the next hidden layer with 120 nodes and relu as the activation function.
model_Bl.add(Flatten())
model_Bl.add(Dense (120,activation="relu"))

#The fully connected layer is followed by another fully connected layer with 84 nodes and relu as the activation function, 
#then connected to a softmax layer with 10 output nodes (corresponding to the 10 classes).

model_Bl.add(Dense (84,activation="relu"))
model_Bl.add(Dense (num_classes,activation="softmax"))

#We will train such a network with the training set and then test it on the testing set.

model_Bl.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

#The history object is returned from calls to the fit() function used to train the model.
#Metrics are stored in a dictionary in the history member of the object returned.

history = model_Bl.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,callbacks=[plot_losses],validation_data=(x_test, y_test))
score = model_Bl.evaluate(x_test, y_test, verbose=0)

#Getting all the average values for Traing accuracy, test accuracy, training loss nd test loss  with baseline.

TR_Accuracy = np.mean (history.history ['acc'])
TE_Accuracy = np.mean (history.history ['val_acc'])
TR_Loss     = np.mean (history.history ['loss'])
TE_Loss     = np.mean (history.history ['val_loss'])

#Print all parameters

print ('Configuration Parameters')
print ('Batch Size       : %s '%(batch_size))
print ('Number of Class  : %s '%(num_classes))
print ('Epochs           : %s '%(epochs))
print ("Training Accuracy     = %s "%(TR_Accuracy))
print ("Test     Accuracy     = %s "%(TE_Accuracy))
print ("Training Loss         = %s "%(TR_Loss))
print ("Test Loss             = %s "%(TE_Loss))

#Plot the training error and the testing error as a function of the learning epochs. 
plot_losses.display('Baseline',TR_Accuracy,TE_Accuracy,TR_Loss,TE_Loss)

#Coding for Experiment 1
print ("-------------------------------")
print('Experiment 1')


#Change kernel size to 5*5, redo the experiment, plot the learning errors along with the epoch. 

model_E1 = Sequential()
model_E1.add(Conv2D(6, kernel_size=(5, 5),activation='relu',input_shape=input_shape,strides=(1,1)))
model_E1.add (MaxPooling2D (pool_size=(2,2),strides=(1,1)))
model_E1.add(Conv2D(16, (5, 5), activation='relu',strides=(1,1)))
model_E1.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))
model_E1.add(Flatten())
#Change the number of the feature maps in the first and second convolutional layers, 
#plot the learning errors along with the epoch, and report the testing error and accuracy on the test set.

model_E1.add(Dense (120,activation="relu"))
model_E1.add(Dense (84,activation="relu"))
model_E1.add(Dense (num_classes,activation="softmax"))
model_E1.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
history = model_E1.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,callbacks=[plot_losses],validation_data=(x_test, y_test))
score = model_E1.evaluate(x_test, y_test, verbose=0)

##Getting all the average values for Traing accuracy, test accuracy, training loss nd test loss with experiment 1.

TR_Accuracy = np.mean (history.history ['acc'])
TE_Accuracy = np.mean (history.history ['val_acc'])
TR_Loss     = np.mean (history.history ['loss'])
TE_Loss     = np.mean (history.history ['val_loss'])

##Print all parameters

print ('Configuration Parameters')
print ('Batch Size       : %s '%(batch_size))
print ('Number of Class  : %s '%(num_classes))
print ('Epochs           : %s '%(epochs))
print ("Training Accuracy     = %s "%(TR_Accuracy))
print ("Test     Accuracy     = %s "%(TE_Accuracy))
print ("Training Loss         = %s "%(TR_Loss))
print ("Test Loss             = %s "%(TE_Loss))

#Plot the training error and the testing error as a function of the learning epochs with experiment 1.
plot_losses.display('Experiment1',TR_Accuracy,TE_Accuracy,TR_Loss,TE_Loss)

print('Experiment 2')

#Change the number of the feature maps in the first and second convolutional layers with values 32 and 64, with kernel(5,5)
#plot the learning errors along with the epoch, and report the testing error and accuracy on the test set.

model_E2 = Sequential()
model_E2.add(Conv2D(32, kernel_size=(5,5),activation='relu',input_shape=input_shape,strides=(1,1)))
model_E2.add (MaxPooling2D (pool_size=(2,2),strides=(1,1)))
model_E2.add(Conv2D(64, (5, 5), activation='relu',strides=(1,1)))
model_E2.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))
model_E2.add(Flatten())
model_E2.add(Dense (120,activation="relu"))
model_E2.add(Dense (84,activation="relu"))
model_E2.add(Dense (num_classes,activation="softmax"))
model_E2.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model_E2.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,callbacks=[plot_losses],validation_data=(x_test, y_test))
score = model_E2.evaluate(x_test, y_test, verbose=0)

print ('Configuration Parameters')
print ('Batch Size       : %s '%(batch_size))
print ('Number of Class  : %s '%(num_classes))
print ('Epochs           : %s '%(epochs))
print ("Training Accuracy     = %s "%(TR_Accuracy))
print ("Test     Accuracy     = %s "%(TE_Accuracy))
print ("Training Loss         = %s "%(TR_Loss))
print ("Test Loss             = %s "%(TE_Loss))
plot_losses.display('Experiment2',TR_Accuracy,TE_Accuracy,TR_Loss,TE_Loss)

print ("-------------------------------")


print('Experiment 3')

#Change the number of the feature maps in the first and second convolutional layers with values 32 and 64, with kernel(3,3)
#plot the learning errors along with the epoch, and report the testing error and accuracy on the test set.
#Additional part.
model_E3 = Sequential()
model_E3.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape,strides=(1,1)))
model_E3.add (MaxPooling2D (pool_size=(2,2),strides=(1,1)))
model_E3.add(Conv2D(64, (3, 3), activation='relu',strides=(1,1)))
model_E3.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))
model_E3.add(Flatten())
model_E3.add(Dense (120,activation="relu"))
model_E3.add(Dense (84,activation="relu"))
model_E3.add(Dense (num_classes,activation="softmax"))
model_E3.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model_E3.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,callbacks=[plot_losses],validation_data=(x_test, y_test))
score = model_E3.evaluate(x_test, y_test, verbose=0)

print ('Configuration Parameters')
print ('Batch Size       : %s '%(batch_size))
print ('Number of Class  : %s '%(num_classes))
print ('Epochs           : %s '%(epochs))
print ("Training Accuracy     = %s "%(TR_Accuracy))
print ("Test     Accuracy     = %s "%(TE_Accuracy))
print ("Training Loss         = %s "%(TR_Loss))
print ("Test Loss             = %s "%(TE_Loss))
plot_losses.display('Experiment3',TR_Accuracy,TE_Accuracy,TR_Loss,TE_Loss)

print ("-------------------------------")