import pandas as pd
import numpy as np
import random
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

# Packages for explanation
from keras import backend as K
from keras.models import Model
from deepexplain.tensorflow import DeepExplain
from IPython.display import display, HTML
import matplotlib.pyplot as plt

import numpy as np, warnings
warnings.filterwarnings("ignore") #Ignore warnings
datatrain = pd.read_csv('data/yeast/yeast-train.csv')
datatest = pd.read_csv('data/yeast/yeast-test.csv')

datatrain.loc[datatrain['Class1'] =="b'0'", 'Class1'] = 0
datatrain.loc[datatrain['Class1'] =="b'1'", 'Class1'] = 1
datatest.loc[datatest['Class1'] =="b'0'", 'Class1'] = 0
datatest.loc[datatest['Class1'] =="b'1'", 'Class1'] = 1

dataset = pd.concat([datatrain, datatest], ignore_index=True)
classNum=1
y_class = 102 + classNum
X = dataset.iloc[:,:103].values
y = dataset.iloc[:,y_class:y_class+1].values
random.seed(0)
X_train,X_test, y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

sgdOptimizer = 'adam'
lossFun='categorical_crossentropy'
finalLayerActivation = 'softmax'
batchSize=25
numEpochs = 500
nb_classes = 2

# Define training data
features = ['Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7', 'Att8', 'Att9', 
            'Att10', 'Att11', 'Att12', 'Att13', 'Att14', 'Att15', 'Att16', 'Att17', 
            'Att18', 'Att19', 'Att20', 'Att21', 'Att22', 'Att23', 'Att24', 'Att25', 
            'Att26', 'Att27', 'Att28', 'Att29', 'Att30', 'Att31', 'Att32', 'Att33', 
            'Att34', 'Att35', 'Att36', 'Att37', 'Att38', 'Att39', 'Att40', 'Att41', 
            'Att42', 'Att43', 'Att44', 'Att45', 'Att46', 'Att47', 'Att48', 'Att49', 
            'Att50', 'Att51', 'Att52', 'Att53', 'Att54', 'Att55', 'Att56', 'Att57', 
            'Att58', 'Att59', 'Att60', 'Att61', 'Att62', 'Att63', 'Att64', 'Att65', 
            'Att66', 'Att67', 'Att68', 'Att69', 'Att70', 'Att71', 'Att72', 'Att73', 
            'Att74', 'Att75', 'Att76', 'Att77', 'Att78', 'Att79', 'Att80', 'Att81', 
            'Att82', 'Att83', 'Att84', 'Att85', 'Att86', 'Att87', 'Att88', 'Att89', 
            'Att90', 'Att91', 'Att92', 'Att93', 'Att94', 'Att95', 'Att96', 'Att97', 
            'Att98', 'Att99', 'Att100', 'Att101', 'Att102', 'Att103']
x_train = dataset[features]
inputDim = len(features)
trainX = x_train

# y_train = dataset['Class1']
# trainY = np_utils.to_categorical(y_train, num_classes = nb_classes)
trainY = np_utils.to_categorical(y_train, num_classes = nb_classes)
testY = np_utils.to_categorical(y_test, num_classes = nb_classes)
# Define model
model = Sequential()
model.add(BatchNormalization(input_shape=(inputDim,)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(nb_classes, activation=finalLayerActivation))
model.compile(loss=lossFun, optimizer=sgdOptimizer, metrics=["accuracy"])
co=1
for i in range(10):
	start = time.time()
	print('Training iteration: {}'.format(co)) 
	model.fit(X_train,trainY,validation_data=(X_test,testY) ,batch_size=batchSize, epochs=numEpochs, verbose=0)
	score = model.evaluate(X_train,trainY, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	score = model.evaluate(X_test,testY, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
	    
		'''
		Need to reconstruct the graph in DeepExplain context, using the same weights.
		1. Get the input tensor
		2. Get embedding tensor
		3. Target the output of the last dense layer (pre-softmax)
		'''

		inputTensor = model.layers[0].input
		fModel = Model(inputs=inputTensor, outputs = model.layers[-1].output)
		targetTensor = fModel(inputTensor)

		# Sample Data for attribution
		#     sampleX = trainX[num:num_next]    
		#     ys = trainY[num:num_next]

		# Sample Data for attribution
		sampleX = X_train
		ys = trainY
		relevances = de.explain('elrp', targetTensor * ys, inputTensor, sampleX)
		print(relevances.shape)

		relFeatures =  ['Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7', 'Att8', 'Att9', 
			    'Att10', 'Att11', 'Att12', 'Att13', 'Att14', 'Att15', 'Att16', 'Att17', 
			    'Att18', 'Att19', 'Att20', 'Att21', 'Att22', 'Att23', 'Att24', 'Att25', 
			    'Att26', 'Att27', 'Att28', 'Att29', 'Att30', 'Att31', 'Att32', 'Att33', 
			    'Att34', 'Att35', 'Att36', 'Att37', 'Att38', 'Att39', 'Att40', 'Att41', 
			    'Att42', 'Att43', 'Att44', 'Att45', 'Att46', 'Att47', 'Att48', 'Att49', 
			    'Att50', 'Att51', 'Att52', 'Att53', 'Att54', 'Att55', 'Att56', 'Att57', 
			    'Att58', 'Att59', 'Att60', 'Att61', 'Att62', 'Att63', 'Att64', 'Att65', 
			    'Att66', 'Att67', 'Att68', 'Att69', 'Att70', 'Att71', 'Att72', 'Att73', 
			    'Att74', 'Att75', 'Att76', 'Att77', 'Att78', 'Att79', 'Att80', 'Att81', 
			    'Att82', 'Att83', 'Att84', 'Att85', 'Att86', 'Att87', 'Att88', 'Att89', 
			    'Att90', 'Att91', 'Att92', 'Att93', 'Att94', 'Att95', 'Att96', 'Att97', 
			    'Att98', 'Att99', 'Att100', 'Att101', 'Att102', 'Att103']
		for i in range(len(relFeatures)):
			word = str(relFeatures[i])
			originalRelevance = "{:8.2f}".format(relevances[0][i])
			#         print ("\t\t\t" + str(originalRelevance) + "\t" + word)

	relDataFrame = pd.DataFrame.from_records(relevances, columns=relFeatures)

	df = pd.Series(relDataFrame.mean())

	new_col = 'expr_'+str(co)
	d = pd.read_csv('relevance.csv')
	d[new_col] = df.values
	d.to_csv('relevance.csv', index=False)
	co=co+1
	print('Time taken: {} seconds'.format(time.time()-start))








