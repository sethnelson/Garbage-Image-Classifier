# pair programming between Seth Nelson and Braeden Watkins

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

split_data = "../data/processed"
pca_cache = "../data/pca_cache"

# load in train and test sets
X_train = np.load(split_data + "/X_train.npy")
y_train = np.load(split_data + "/y_train.npy")
X_test = np.load(split_data + "/X_test.npy")
y_test = np.load(split_data + "/y_test.npy")

# use incrementalPCA instead of PCA to avoid integer overflow
pca = IncrementalPCA(
    n_components=155, # 90% explainability after graphing explained variance ratio #155
    batch_size=1000,
    whiten=True)   # appreciably boosts performance
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

# plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()

# save reduced dimension train and test set to use in train.py
np.save(os.path.join(pca_cache, "X_train_pca.npy"), X_train_pca)
np.save(os.path.join(pca_cache, "X_test_pca.npy"),  X_test_pca)