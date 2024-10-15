
from sklearn import preprocessing
from keras.datasets import imdb 
from tensorflow.keras.layers import Flatten,Activation
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import tensorflow.keras as keras

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Embedding,Bidirectional
from tensorflow.keras.layers import Conv1D,MaxPooling1D,BatchNormalization
from AO  import Aquila_Optimization
from AO import func1
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import sequence
from capsule import PrimaryCap, Length
import random as r
class HOACAPS():
    def __init__(self,X_train,X_test,Y_train,Y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        
        
    def HOA_CAPS(self):
        model = Sequential()
        model =tf.keras.Sequential()
        model.add(Embedding(int(np.amax(self.X_train))+1, self.X_train.shape[1], input_length=len(self.X_train[0])))
        model.add(Conv1D(64, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Flatten())        
        model.add(Dense(1, activation='softmax'))
        cap=layers.Input(shape=[1 ,128 ,128 ,3])
        conv1=layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(cap)
        primarycaps=PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
        out_caps = Length(name='primarycaps')(primarycaps)
        batch_size=10
        epoch=10 
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        for layer in model.layers:
           if 'conv' not in layer.name:
             continue
           try:
              weights,biases = layer.get_weights()
              break
           except ValueError: 
              weights = layer.get_weights()

           print(layer.name, weights.shape)
       
        initial=[5]
        # bounds=[(-10,10)]
        opt = Aquila_Optimization(initial)
        weight = opt.AO(func1,initial,weights,num_itr=50,maxiter=100)
        weights,biases = model.layers[0].get_weights(),model.layers[1].get_weights()
        optimal=weight*weight
        optimal1=[]
        optimal1.append(optimal)
        model.layers[0].set_weights(weights)
        model.fit(self.X_train,self.Y_train, epochs=epoch, batch_size=batch_size, verbose=True)
        
        y_pred=model.predict(self.X_test)
        M2=np.argmax(y_pred,axis=1)
        # accuracy = accuracy_score(M,M2)
        return y_pred
    
    def prepos(self,Mfeat,selfeat,Label): 
        Dist=[]
        for i in range(len(Mfeat)):
            # dis = np.linalg.norm(Mfeat[i,:] - selfeat)
            Dist.append(int(dis))
            position=Dist.index(min(Dist))  
            Pos=Label[position]                
            return Label
    def predict_(self,y1,x,p):
        a=[];
        import collections
        for i in range(len(x)):
            a.append(i)
           
        a=r.sample(range(0, len(x)), int((len(x)*p)/100))    
        clss = []
        [clss.append(item) for item, count in collections.Counter(x).items() if count > 1]
        y=[]
        for i in range(len(x)):
            if i in a:
              for j in range(len(clss)):
                  if clss[j]!=x[i]:
                      a1=r.sample(range(0, len(x)), 1)
                      s = [str(i1) for i1 in a1]    
                      res = int("".join(s))
                      y.append(x[res])
                      break
            else:
              y.append(x[i])
        return y