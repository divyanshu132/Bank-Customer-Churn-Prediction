import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:, 3:13].values 
y=dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:, 1]=labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2=LabelEncoder()
x[:, 2]=labelencoder_x_2.fit_transform(x[:, 2])

Onehotencoder=OneHotEncoder(categorical_features=[1])
x=Onehotencoder.fit_transform(x).toarray()
x=x[:, 1:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=10, epochs=100)

y_pred=classifier.predict(x_test)
y_pred=y_pred>0.5

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

accuracy=(cm[0][0]+cm[1][1])/len(y_test)
print("accuracy of model: ",accuracy)

#individual prediction

#Change the parameters of .predict below according to the input data columns for individual prediction
individual_predict=classifier.predict(sc.transform(np.array([[0.0, 1, 600, 1, 25, 2, 100000, 2, 1, 1, 50000]])))

individual_predict=individual_predict>0.5
print("Output of individual prediction: ",individual_predict)
print("\n")


#Tunning the model
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

def __classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=__classifier, batch_size=10, epochs=100)

accuracies=cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=1)

print("accuracy using 10-fold cross validation: ",accuracies)
mean=accuracies.mean()
variance=accuracies.std()
print("mean of 10-fold cross validation: ", mean)
print("variance of 10-fold cross validation: ", variance)

def __classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=__classifier)

from sklearn.model_selection import GridSearchCV
parameters={'batch_size':[25, 32], 
            'nb_epoch':[100, 500],
            'optimizer':['adam', 'rmsprop']}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)

grid_search=grid_search.fit(x_train, y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_

print("Best parameter chosen by GridSearchCV from given parameters is: ",best_parameters)
print("Final improved accuracy: ", best_accuracy)

