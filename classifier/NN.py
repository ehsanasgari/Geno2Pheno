import sys
sys.path.append('../')

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from utility.file_utility import FileUtility
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from gensim.models.wrappers import FastText
import itertools




def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(18, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

X=FileUtility.load_sparse_csr('../../datasets/processed_data/eco/K/6-mer_eco_restrictedmer.npz').toarray()
Y=FileUtility.load_list('../../datasets/processed_data/eco/K/eco_label_restrictedkmer.txt')
#    model.add(Dense(512, input_dim=4096, activation='relu'))
#    model.add(Dropout(0.2))
#    model.add(Dense(200, activation='relu'))
#    model.add(Dropout(0.1))    
#    model.add(Dense(20, activation='relu'))
#    model.add(Dense(5, activation='softmax'))



#X=FileUtility.load_sparse_csr('../../datasets/crohn/otu_complete1359.npz').toarray()
#Y=FileUtility.load_list('../../datasets/crohn/data_config/labels_disease_complete1359.txt')
#model = Sequential()
#model.add(Dense(500, input_dim=9511, activation='relu'))
    #model.add(Dropout(0.2))
#model.add(Dense(500, activation='relu'))
    #model.add(Dropout(0.1))    
#model.add(Dense(2, activation='softmax'))
    # Compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#X_WV=np.load('../../datasets/WV_16s.npz')['arr_0']
#X_WV=X.dot(X_WV)  
#X=np.concatenate([X,X_WV],axis=1)


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
onehot_y = np_utils.to_categorical(encoded_Y)


skf = StratifiedKFold(n_splits=5, shuffle=True)

p_micro=[]
p_macro=[]
r_micro=[]
r_macro=[]
f1_micro=[]
f1_macro=[]

for train_index, valid_index in skf.split(X, Y):
    print ('\n New fold started ..')
    X_train=X[train_index,:]
    y_train=onehot_y[train_index,:]
    y_class_train=encoded_Y[train_index]
    X_valid=X[valid_index,:]
    y_valid=onehot_y[valid_index,:]
    y_class_valid=encoded_Y[valid_index]
    model=baseline_model()
    history = model.fit(X_train, y_train, epochs=50, batch_size=100,shuffle=True, validation_data=(X_valid, y_valid), verbose=0)
    pred=model.predict_classes(X_valid)
    f1_micro.append(f1_score(pred, y_class_valid, average='micro'))
    f1_macro.append(f1_score(pred, y_class_valid, average='macro'))
    p_micro.append(precision_score(pred, y_class_valid, average='micro'))
    p_macro.append(precision_score(pred, y_class_valid, average='macro'))
    r_micro.append(recall_score(pred, y_class_valid, average='micro'))
    r_macro.append(recall_score(pred, y_class_valid, average='macro'))
    

f1mac=np.mean(f1_macro)
f1mic=np.mean(f1_micro)
prmac=np.mean(p_macro)
prmic=np.mean(p_micro)
remac=np.mean(r_macro)
remic=np.mean(r_micro)

sf1mac=np.std(f1_macro)
sf1mic=np.std(f1_micro)
sprmac=np.std(p_macro)
sprmic=np.std(p_micro)
sremac=np.std(r_macro)
sremic=np.std(r_micro)

print (' & '.join([str(np.round(x,2))+' $\\pm$ '+str(np.round(y,2)) for x,y in [[prmic, sprmic], [remic, sremic], [f1mic, sf1mic], [prmac, sprmac], [remac, sremac], [f1mac, sf1mac] ]]))


model2 = Sequential()
model2.add(Dense(512, input_dim=20589, weights=model.layers[0].get_weights(), activation='relu'))
model2.add(Dense(200, weights=model.layers[1].get_weights(), activation='relu'))
model2.add(Dense(20, weights=model.layers[2].get_weights(), activation='relu'))
 

activations = model2.predict(X)

np.savetxt('../../datasets/bodysite/Activation_3layers_otu',activations)
