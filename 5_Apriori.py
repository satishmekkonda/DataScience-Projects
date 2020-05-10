#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:41:56 2020

@author: narsingrao
"""


import os
os.chdir('/Users/narsingrao/Documents/Satish_ML/Machine Learning A-Z (Codes and Datasets)/Part 5 - Association Rule Learning/Section 28 - Apriori/Python')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Datasets = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j] for j in range(0, 20))])
    

from apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
    
results = list(rules)

    
    