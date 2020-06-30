# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:12:26 2020

@author: aksha
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the dataset
# clustering without feature extraction
dataset1 = pd.read_csv('positives.csv')
X1 = dataset1.iloc[:,1:50177].values

dataset2 = pd.read_csv('negatives.csv')
X2 = dataset2.iloc[:,1:50177].values

# Using the elbow method to find the optimal number of clusters
def plot(X,n_clusters):
    wcss1 = []
    for i in range(1, n_clusters):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss1.append(kmeans.inertia_)
        print(i)
    plt.plot(range(1, n_clusters), wcss1,markersize = 1,linewidth=1.0)
    plt.title('The Elbow Method(positives)')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

plot(X1,21)
plot(X2,21)
   
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans_pos = kmeans.fit_predict(X1)        

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans_neg = kmeans.fit_predict(X2) 

d = {}
for i in range(len(y_kmeans_pos)):
    if y_kmeans_pos[i] == 0:
        d[i] = 1
    else:
        d[i] = 2
        print(i)

e = pd.DataFrame.from_dict(d,orient = 'index')
X1 = dataset1.iloc[:,1:50177]
result_pos = pd.concat([X1,e],axis = 1,sort = False)

f = {}
for i in range(len(y_kmeans_neg)):
    if y_kmeans_neg[i] == 0:
        f[i] = 3
    elif y_kmeans_neg[i] == 1:
        f[i] = 4
    elif y_kmeans_neg[i] == 2:
        f[i] = 5
    else:
        f[i] = 6
        print(i)

g = pd.DataFrame.from_dict(f,orient = 'index')
X2 = dataset2.iloc[:,1:50177]
result_neg = pd.concat([X2,g],axis = 1,sort = False)

result_pos.to_csv('result_positives.csv')
result_neg.to_csv('result_negatives.csv')
a,b,c,d = [],[],[],[]
for i in range(len(result_neg.iloc[:,-1])):
    if result_neg.iloc[i,-1] == 3:
        a.append(i)
    elif result_neg.iloc[i,-1] == 4:
        b.append(i)
    elif result_neg.iloc[i,-1] == 5:
        c.append(i)
    else:
        d.append(i)
        
import random
from PIL import Image

def img_grid(l):
    for i in range(9):
        rand = random.choice(l) 
        x = result_neg.iloc[rand,:50176].values
        x = x.reshape(224,224)
        plt.subplot(3,3,i+1)
        plt.imshow(x,'gray')
        plt.xticks([]),plt.yticks([])
    plt.show()

img_grid(a)
img_grid(b)
img_grid(c)
img_grid(d)



        
# using feature extraction to find optimal number of clusters
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\positives"
targetdir = "C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos"

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

plot(featurelist,21)
optimum_no= 6
kmeans = KMeans(n_clusters=optimum_no, random_state=0).fit(np.array(featurelist))
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")

imdir = "C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\negatives"
targetdir = "C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg"

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())
    
plot(featurelist,21)
optimum_no = 6
kmeans = KMeans(n_clusters=optimum_no, random_state=0).fit(np.array(featurelist))
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")



