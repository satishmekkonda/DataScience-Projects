#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:31:12 2020

@author: narsingrao
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:04:02 2020

@author: narsingrao
"""


import os
os.chdir('/Users/narsingrao/Documents/Satish_ML/Machine Learning A-Z (Codes and Datasets)/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Python')

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




from sklearn.decomposition import KernelPCA
#pca = PCA(n_components = None)
#X_train = pca.fit_transform(X_train)
#X_test = pca.fit_transform(X_test)
#explained_variance = pca.explained_variance_ratio_

kpca = KernelPCA(n_components = 2, kernel='rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.fit_transform(X_test)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)