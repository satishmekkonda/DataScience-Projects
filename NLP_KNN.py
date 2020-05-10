#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:57:58 2020

@author: narsingrao
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:51:41 2020

@author: narsingrao
"""



import os
os.chdir('/Users/narsingrao/Documents/Satish_ML/Machine Learning A-Z (Codes and Datasets)/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Python')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
corpus = []
for i in range(0, 1000):
        
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review= review.lower()
    review = review.split()
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p =2)
classifier.fit(X_train, y_train)


#Predicting Results TEST

y_pred = classifier.predict(X_test)

#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0] + cm[0][1]) / (cm[1][0]+cm[1][1])