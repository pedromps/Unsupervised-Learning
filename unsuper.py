# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

x_train1 = np.load("dataset1_xtrain.npy")
y_train1 = np.load("dataset1_ytrain.npy")
x_test1 = np.load("dataset1_xtest.npy")
y_test1 = np.load("dataset1_ytest.npy")

x_train2 = np.load("dataset2_xtrain.npy")
y_train2 = np.load("dataset2_ytrain.npy")
x_test2 = np.load("dataset2_xtest.npy")
y_test2 = np.load("dataset2_ytest.npy")


print("MLP para o primeiro dataset:\n\n")
model = Sequential()
model.add(Dense(922, input_shape=(17,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='softmax'))
#model.summary()

x_train, x_validation, y_train, y_validation = train_test_split(x_train1, y_train1, test_size=0.30)
es = EarlyStopping(patience = 15, restore_best_weights=True)
model.compile(optimizer='Adam',loss='mean_squared_error', metrics=['accuracy'])

history_es1 = model.fit(x=x_train, y=y_train, epochs=200, batch_size=921, validation_data=(x_validation, y_validation), callbacks=[es])
plt.figure(1)
plt.plot(history_es1.history['loss'])
plt.title('Perdas no modelo com Early Stopping')
plt.ylabel('Perdas')
plt.xlabel('Época')
plt.legend('Treino', loc='upper right')
plt.show()
print("\nConfusion Matrix = \n", confusion_matrix(y_test1, model.predict(x_test1)))

print("MLP para o segundo dataset:\n\n")
model = Sequential()
model.add(Dense(231, input_shape=(10,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='softmax'))
#model.summary()

x_train, x_validation, y_train, y_validation = train_test_split(x_train2, y_train2, test_size=0.30)
es = EarlyStopping(patience = 15, restore_best_weights=True)
model.compile(optimizer='Adam',loss='mean_squared_error', metrics=['accuracy'])

history_es2 = model.fit(x=x_train, y=y_train, epochs=200, batch_size=231, validation_data=(x_validation, y_validation), callbacks=[es])
plt.figure(1)
plt.plot(history_es1.history['loss'])
plt.title('Perdas no modelo com Early Stopping')
plt.ylabel('Perdas')
plt.xlabel('Época')
plt.legend('Treino', loc='upper right')
plt.show()
print("\nConfusion Matrix = \n", confusion_matrix(y_test2, model.predict(x_test2)))





#Testado com kernel rbf (Gaussiano)
print("SVM com Kernel Gaussiano para o primeiro dataset:\n\n")
clf = SVC(kernel = 'rbf', max_iter = 100000, gamma = 'auto')
clf.fit(x_train1, y_train1.ravel())

print("\nAccuracy do modelo treinado = ", clf.score(x_train1, y_train1))
print("\nConfusion Matrix = ", confusion_matrix(y_train1, clf.predict(x_train1)))

print("\nAccuracy do modelo com dataset de teste = ", clf.score(x_test1, y_test1))
print("\nConfusion Matrix = \n", confusion_matrix(y_test1, clf.predict(x_test1)))


print("SVM com Kernel Gaussiano para o segundo dataset:\n\n")
clf = SVC(kernel = 'rbf', max_iter = 100000, gamma = 'auto')
clf.fit(x_train2, y_train2.ravel())

print("\nAccuracy do modelo treinado = ", balanced_accuracy_score(y_train2, clf.predict(x_train2)))
print("\nConfusion Matrix = ", confusion_matrix(y_train2, clf.predict(x_train2)))

print("\nAccuracy do modelo com dataset de teste = ", balanced_accuracy_score(y_test2, clf.predict(x_test2)))
print("\nConfusion Matrix = \n", confusion_matrix(y_test2, clf.predict(x_test2)))





print("Decision Tree para o primeiro dataset:\n\n")
clf = DecisionTreeClassifier()
clf.fit(x_train1, y_train1)

print("\nAccuracy do modelo treinado = ", clf.score(x_train1, y_train1))
print("\nConfusion Matrix = \n", confusion_matrix(y_train1, clf.predict(x_train1)))

print("\nAccuracy do modelo com dataset de teste = ", clf.score(x_test1, y_test1))
print("\nConfusion Matrix = \n", confusion_matrix(y_test1, clf.predict(x_test1)))


print("Decision Tree para o segundo dataset:\n\n")
clf = DecisionTreeClassifier()
clf.fit(x_train2, y_train2)

print("\nAccuracy do modelo treinado = ", clf.score(x_train2, y_train2))
print("\nConfusion Matrix = \n", confusion_matrix(y_train2, clf.predict(x_train2)))

print("\nAccuracy do modelo com dataset de teste = ", balanced_accuracy_score(y_test2, clf.predict(x_test2)))
print("\nConfusion Matrix = \n", confusion_matrix(y_test2, clf.predict(x_test2)))