# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:07:51 2020

@author: Akshansh
"""

import numpy as np
import pickle
import pandas as pd
with open("MODS_all_data_bw_224_224_0.pkl", 'rb') as f:
  l = pickle.load(f, encoding='bytes')
  f.close()
  
training_data = np.array(l[0][0])
training_data_out = np.array(l[0][1])
test_data = np.array(l[1][0])
test_data_out = np.array(l[1][1])

pos = []
neg = []
for i in range(len(training_data_out)):
    if training_data_out[i] == 1:
        pos.append(np.append(training_data[i],1))
    else:
        neg.append(np.append(training_data[i],0))
for i in range(len(test_data_out)):
    if test_data_out[i] == 1:
        pos.append(np.append(test_data[i],1))
    else:
        neg.append(np.append(test_data[i],0))

pos_df = pd.DataFrame(pos)
neg_df = pd.DataFrame(neg)   
pos_df.to_csv('positives.csv')
neg_df.to_csv('negatives.csv')     