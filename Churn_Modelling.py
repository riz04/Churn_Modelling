Created on Sun May 19 12:22:11 2019

@author: riz04

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

# defining the object
# initializing ANN classifier
classifier = Sequential()

# Adding the input layer and first hidden layer
# first step is to initializ the weights, that will be done using dense function
# We have 11 IV's, so in our input layer, we will have 11 input nodes
# followed by forward propagation
# from left to right neurons are activated by activation functions
# such as higher the value of the activation function is
# more impact the neural will have in the network
# more it will pass signals from left node to right nodes
# we can use sigmoid fun in the output layer, and
# we get probabilities of customers likely to leave bank or not
# and we will use rectifier function for hidden layers
# after comparing the acctual y with predicted y
# the error we receive is back propagated in the neural networ
# and then weights get updated

# output_dim = no of nodes we wnat to add in hidden layer
# init = initialize the weight randomly with a uniform function
# input_dim = no of nodes in the input layer (numbers of IV)
classifier.add(Dense(activation="relu", input_dim=11, units=6, 
                     kernel_initializer="uniform"))

# adding second hidden layer
# we only need to specify input_dim in the first layer, so that it has some 
# idea, what is going to come
# but, in second hidden layer we don't need to specify input_dim
# because it knows what to expect
classifier.add(Dense(activation="relu", units=6, 
                     kernel_initializer="uniform"))

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