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
    embedding=np.load('../../datasets/WV_16s.npz')['arr_0']
    model = Sequential()
    model.add(Dense(1000, input_dim=4096,  weights = [embedding, np.array([0]*1000)], trainable=False , activation='linear'))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))    
    model.add(Dense(2, activation='softmax'))
    # Compile model
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

X=FileUtility.load_sparse_csr('../../datasets/crohn/6-mers_rate_complete1359_seq_5000.npz').toarray()
Y=FileUtility.load_list('../../datasets/crohn/data_config/labels_disease_complete1359.txt')

#X=FileUtility.load_sparse_csr('../../datasets/bodysite/6-mers_rate_5000.npz').toarray()
#Y=FileUtility.load_list('../../datasets/bodysite/data_config/labels_phen4_class.txt')
#    embedding=np.load('../../datasets/WV_16s.npz')['arr_0']
#    model = Sequential()
#    model.add(Dense(1000, input_dim=4096,  weights = [embedding, np.array([0]*1000)], trainable=False , activation='linear'))
#    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.2))
#    model.add(Dense(100, activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(20, activation='relu'))
#    model.add(Dense(4, activation='softmax'))
    # Compile model

#X=FileUtility.load_sparse_csr('../../datasets/crohn/6-mers_rate_complete1359_seq_-1.npz').toarray()
#Y=FileUtility.load_list('../../datasets/crohn/data_config/labels_disease_complete1359.txt')
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


#model2 = Sequential()
#model2.add(Dense(1024, input_dim=4096, weights=model.layers[0].get_weights(), activation='relu'))
#model2.add(Dense(256, input_dim=1024, weights=model.layers[1].get_weights(), activation='relu'))
#model2.add(Dense(256, input_dim=256, weights=model.layers[2].get_weights(), activation='relu'))
#model2.add(Dense(128, input_dim=256, weights=model.layers[3].get_weights(), activation='relu'))
#model2.add(Dense(64, input_dim=128, weights=model.layers[4].get_weights(), activation='relu'))
 

#activations = model2.predict(X)

#np.savetxt('../../datasets/crohn/Activation_CD_complete1359',activations)
