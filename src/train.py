import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import logistic_regressor as lr #custom model

split_data = "../data/small_test_set/processed"

X_train = np.load(split_data + "/X_train.npy")
y_train = np.load(split_data + "/y_train.npy")
X_val = np.load(split_data + "/X_val.npy")
y_val = np.load(split_data + "/y_val.npy")
X_test = np.load(split_data + "/X_test.npy")
y_test = np.load(split_data + "/y_test.npy")

pca = PCA(n_components=200, whiten=True) # Radical dimensionality reduction from 196,608 -> 200
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

kpca = KernelPCA(n_components=100, kernel='rbf', gamma=0.01) # Further reduce using nonlinear version of PCA
X_train_kpca = kpca.fit_transform(X_train_pca)
X_val_kpca   = kpca.transform(X_val_pca)
X_test_kpca   = kpca.transform(X_test_pca)

model = LogisticRegression(max_iter=500, solver="lbfgs")

model.fit(X_train_pca, np.argmax(y_train, axis=1)) # train and evaluate PCA
print("PCA Validation accuracy:", model.score(X_val_pca, np.argmax(y_val, axis=1)))
print("PCA Test accuracy:", model.score(X_test_pca, np.argmax(y_test, axis=1)))

model.fit(X_train_kpca, np.argmax(y_train, axis=1)) # train and evaluate KPCA, +4 points typically.
print("KPCA Validation accuracy:", model.score(X_val_kpca, np.argmax(y_val, axis=1)))
print("KPCA Test accuracy:", model.score(X_test_kpca, np.argmax(y_test, axis=1)))

# (SN) Can't use custom LR model as it has trouble handling the dimensional differences PCA/LDA produces. It's also much slower
# (SN) getting about 30% accuracy score with the KPCA. Not great but about what we should expect using LR for this.


#_______________________________________ From-Scratch LR Model _________________________________________
# print(X_train.shape)
# print(y_train.shape)

#model = LogisticRegression(penalty=None, max_iter=1000).fit(X_train, y_train.flatten())
#hyperparameters
# learning_rate = 5e-5
# number_epochs = 100
# lam = 1.0
# reg = 2

# w, b, t_costs, v_costs = lr.train(X_train, y_train, learning_rate, number_epochs, lam, reg, X_val, y_val)


# y_train_labels = np.argmax(y_train, axis=1)
# y_val_labels   = np.argmax(y_val, axis=1)

# y_pred_train = np.argmax(lr.predict_proba(X_train, w, b), axis=1)
# y_pred_val   = np.argmax(lr.predict_proba(X_val_pca, w, b), axis=1)

# print(y_train.shape, y_pred_train.shape)
# print(y_train[:10])
# print(y_pred_train[:10])

# print("Training accuracy:", accuracy_score(y_train_labels, y_pred_train))
# print("Validation accuracy:", accuracy_score(y_val_labels, y_pred_val))


# plt.plot(t_costs, label='Training Loss')
# plt.plot(v_costs, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Logistic Regression Training Progress')
# plt.show()