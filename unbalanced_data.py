# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression

x_train2 = np.load("dataset2_xtrain.npy")
y_train2 = np.load("dataset2_ytrain.npy")
x_test2 = np.load("dataset2_xtest.npy")
y_test2 = np.load("dataset2_ytest.npy")

x_train2 = normalize(x_train2, axis = 1)
x_test2 = normalize(x_test2, axis = 1)

# There's no hints about the datasets. It is just known that it is a classification problem!
# this dataset is UNBALANCED
weights = compute_class_weight('balanced', np.unique(y_train2), y_train2)


# Neural Network (a simple MLP)
ES = EarlyStopping(patience = 50, restore_best_weights=True)
nn_clf = Sequential()
nn_clf.add(Dense(128, activation = 'relu', input_shape = (x_train2.shape[1], )))
nn_clf.add(Dense(32, activation = 'relu'))
nn_clf.add(Dense(1, activation = 'sigmoid'))
nn_clf.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["acc"])
nn_clf.summary()
history = nn_clf.fit(x_train2, y_train2, batch_size = 128, epochs = 2000, validation_split = 0.2, callbacks = ES, verbose = 0)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.show()

print("\n\n\n\nMLP:")
print("Training Accuracy of the model = {:.2f}".format(balanced_accuracy_score(y_train2, nn_clf.predict_classes(x_train2))))
print("Confusion Matrix = \n", confusion_matrix(y_train2, nn_clf.predict_classes(x_train2)))
print("Testing Accuracy of the model = {:.2f}".format(balanced_accuracy_score(y_test2, nn_clf.predict_classes(x_test2))))
print("Confusion Matrix = \n", confusion_matrix(y_test2, nn_clf.predict_classes(x_test2)))

# SVM with linear kernel and a harder margin formulation (best performance so far)
svm_clf = SVC(kernel = 'linear', C = 100, gamma = 'auto')
svm_clf.fit(x_train2, y_train2.ravel())
print("\n\n\n\nSVM:")
print("Training Accuracy of the model = {:.2f}".format(svm_clf.score(x_train2, y_train2)))
print("Confusion Matrix = \n", confusion_matrix(y_train2, svm_clf.predict(x_train2)))
print("Testing Accuracy of the model = {:.2f}".format(svm_clf.score(x_test2, y_test2)))
print("Confusion Matrix = \n", confusion_matrix(y_test2, svm_clf.predict(x_test2)))

# Decision tree, with pruning to try and generalise best (no node can have less than 10 instances)
tree_clf = DecisionTreeClassifier(random_state = 0, min_samples_leaf = 10)
tree_clf.fit(x_train2, y_train2)
print("\n\n\n\nDecision Tree:")
print("Training Accuracy of the model = {:.2f}".format(tree_clf.score(x_train2, y_train2)))
print("Confusion Matrix = \n", confusion_matrix(y_train2, tree_clf.predict(x_train2)))
print("Testing Accuracy of the model = {:.2f}".format(tree_clf.score(x_test2, y_test2)))
print("Confusion Matrix = \n", confusion_matrix(y_test2, tree_clf.predict(x_test2)))

# Regressions
reg_clf = LogisticRegression(penalty = 'l2', C = 10)
reg_clf.fit(x_train2, y_train2.ravel())
print("\n\n\n\nLogistic Regression:")
print("Training Accuracy of the model = {:.2f}".format(reg_clf.score(x_train2, y_train2)))
print("Confusion Matrix = \n", confusion_matrix(y_train2, reg_clf.predict(x_train2)))
print("Testing Accuracy of the model = {:.2f}".format(reg_clf.score(x_test2, y_test2)))
print("Confusion Matrix = \n", confusion_matrix(y_test2, reg_clf.predict(x_test2)))



# now for the unbalanced dataset