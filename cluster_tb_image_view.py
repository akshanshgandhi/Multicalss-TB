# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:11:58 2020

@author: Akshansh
"""
import pandas as pd
import numpy as np
dataset = pd.read_csv('total.csv')
X1 = dataset1.iloc[:,1:50177].values





from PIL import Image
for i in range(len(y_pred)):
    if d[i] == 0 and y_pred[i] == True:
        x = c[i,:]
        x = x.flatten()
        x = x.reshape(1,224,224)
        new_im = Image.frombytes('L',(224,224),x)
        new_im.save(str(i)+'.jpg')
        print(i)
        