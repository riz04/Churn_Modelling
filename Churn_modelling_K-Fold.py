Created on Sun May 19 16:34:29 2019

@author: riz04

# building ANN with K-Fold cross validation

# the problem we are dealing here is classification problem
# ANN can do a great job in this
# we are trying to predict a binary variable

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data 

# Country 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# categorical variables are not ordinal in the country column
# there is no relational order between the categories
# e.g France is not higher than germany
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# we will remove one dummy variable
# so that we don't fall in dummy variable trap
# we drop our first column(0th index)
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Let's make ANN
import keras

# sequential module is required to initialize our neural network
# dense module required to build layers of our ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# defining the object
# initializing ANN classifier
classifier = Sequential()


# improving the ANN
# dropout regularization to reduce overfitting
# overfitting - when our model was trained too much
# and we can fig this out, when we have major diff in accuracy
# of test and train set, we get very high accuracy on train test
# than test set
# we also get high variance, while applying k-fold cross validation
# at each iteration of training - some neurons of your artificial neuron network 
# are randomly disabled
# to prevent them to from being too dependant on each other
# when they learn the correlations
# therefore by over writing these neurons
# ANN Learns several independent correlations in the data
# neurons work more independently, and don't learn too much
# and therefore that prevents overfitting


# adding the input layer and first hidden layer with dropout
classifier.add(Dense(activation="relu", input_dim=11, units=6, 
                     kernel_initializer="uniform"))
# when we have overfitting, we can start with p=0.1
classifier.add(Dropout(p = 0.1))


# adding second hidden layer
# we only need to specify input_dim in the first layer, so that it has some 
# idea, what is going to come
# but, in second hidden layer we don't need to specify input_dim
# because it knows what to expect
classifier.add(Dense(activation="relu", units=6, 
                     kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1))

# adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, 
                     kernel_initializer="uniform"))

# compilation of neural network
# optimizer - algorithm, you want to use, to find optimize set of weights
# loss function is part of the optimizer
# which we have to optimize to find optimize weights
# since we have sigmoid function in our output class, we are going to
# use logarithm loss function
# metrics - criterian you choose to evaluate your model
classifier.compile(optimizer = "adam" , loss = "binary_crossentropy", 
                   metrics = ["accuracy"])

# fitting the ANN to the training set
# batch_size - number of observations after which we want to update our weights
# epoch - when whole dataset is passed through ANN, how many times we want it 
# to repeat
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# predicting the Test set results
y_pred = classifier.predict(X_test)

# confusion matrix will not take probabilities
# so we need to covert them into predicted results in the form of T/F
# for that we choose a threshold
# following statement will convert the results in T/F format
y_pred = (y_pred>0.5)

# predicting a single new observation
# to store the information horizontally
# we use double brackets
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,
                                        2,1,1,50000]])))

new_prediction = (new_prediction>0.5)


# making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculating accuracy by dividing correct prediction with total no of
# prediction
# we achieve accuracy rate of 85%
# (1522 + 167)/2000
# Out[31]: 0.8445


# Evaluating the ANN
# implementing K-fold cross validation
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# following function bulilts architecture of ANN
# i.e classifier of ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, 
                     kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, 
                     kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, 
                     kernel_initializer="uniform"))
    classifier.compile(optimizer = "adam" , loss = "binary_crossentropy", 
                   metrics = ["accuracy"])
    return classifier
        

classifier = KerasClassifier(build_fn = build_classifier,
                                 batch_size = 10, epochs = 100)
    
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                                 cv = 10, pre_dispatch= -1) 
mean = accuracies.mean()
variance = accuracies.std()


# parameter tuning
# we have several fixed parameters like, epochs, neurons etc, they are called
# hyper parameters
# parameter tuning is finding best values for hyperparameters
# we are gonna use an approach called gridsearch
# using several combination of these values 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# following function bulilts architecture of ANN
# i.e classifier of ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, 
                     kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, 
                     kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, 
                     kernel_initializer="uniform"))
    classifier.compile(optimizer = optimizer , loss = "binary_crossentropy", 
                   metrics = ["accuracy"])
    return classifier
        
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {"batch_size" : [25,32] , "epochs" : [100,500], "optimizer" : 
    ["adam" , "rmsprop"]}
    
grid_search = GridSearchCV(estimator = classifier, param_grid=parameters,
                         scoring="accuracy", cv = 10)
grid_search = grid_search.fit(X_train,y_train)

# best selection of parameters
# best accuracy

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_













