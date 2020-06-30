# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:24:12 2020

@author: aksha
"""


import os, shutil, glob, os.path
from PIL import Image
from PIL import ImageFilter
import random

def img_grid(l):
    for i in range(9):
        rand = random.choice(l) 
        x = result_neg.iloc[rand,:50176].values
        x = x.reshape(224,224)
        plt.subplot(3,3,i+1)
        plt.imshow(x,'gray')
        plt.xticks([]),plt.yticks([])
    return plt.show()

pos0 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\0', '*.jpg'))
pos1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1', '*.jpg'))
pos2 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\2', '*.jpg'))
pos3 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3', '*.jpg'))
pos4 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\4', '*.jpg'))
pos5 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\5', '*.jpg'))

neg0 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\0', '*.jpg'))
neg1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1', '*.jpg'))
neg2 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\2', '*.jpg'))
neg3 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\3', '*.jpg'))
neg4 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\4', '*.jpg'))
neg5 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\5', '*.jpg'))

for i in range(len(pos1)):
    im =Image.open(pos1[i])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\unsh_'+str(i)+'.jpg')
    
for i in range(300):
    m = random.randrange(0,len(pos3))
    im =Image.open(pos3[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3\\unsh_'+str(i)+'.jpg')
    print(i)

for i in range(400):
    m = random.randrange(0,len(pos4))
    im =Image.open(pos4[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\4\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\4\\unsh_'+str(i)+'.jpg')
    print(i)
    
for i in range(275):
    m = random.randrange(0,len(pos5))
    im =Image.open(pos5[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\5\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\5\\unsh_'+str(i)+'.jpg')
    print(i)

for i in range(300):
    m = random.randrange(0,len(pos3))
    im =Image.open(pos3[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3\\unsh_'+str(i)+'.jpg')
    print(i)

for i in range(400):
    m = random.randrange(0,len(neg0))
    im =Image.open(neg0[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\0\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\0\\unsh_'+str(i)+'.jpg')
    print(i)

for i in range(400):
    m = random.randrange(0,len(neg0))
    im =Image.open(neg0[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\0\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\0\\unsh_'+str(i)+'.jpg')
    print(i)

for i in range(len(neg1)):
    im =Image.open(neg1[i])
#    im_blur = im.filter(ImageFilter.GaussianBlur)
#    im_unsharp = im.filter(ImageFilter.UnsharpMask)
#    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\bl_'+str(i)+'.jpg')
#    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\unsh_'+str(i)+'.jpg')
    out1 = im.transpose(Image.FLIP_LEFT_RIGHT)
    out1.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\flip_LR_'+str(i)+'.jpg')
    print(i)

neg1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1', '*.jpg'))

for i in range(len(neg1)):
    im =Image.open(neg1[i])
#    im_blur = im.filter(ImageFilter.GaussianBlur)
#    im_unsharp = im.filter(ImageFilter.UnsharpMask)
#    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\bl_'+str(i)+'.jpg')
#    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\unsh_'+str(i)+'.jpg')
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out1.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\flip_TB_'+str(i)+'.jpg')
    print(i)

neg1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1', '*.jpg'))

for i in range(len(neg1)):
    im =Image.open(neg1[i])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_edgeenh = im.filter(ImageFilter.EDGE_ENHANCE)
#    im_emboss = im.filter(ImageFilter.EMBOSS)
    im_sharpen = im.filter(ImageFilter.SHARPEN)
    im_smooth = im.filter(ImageFilter.SMOOTH)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\unsh_'+str(i)+'.jpg')
    im_edgeenh.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\EE_'+str(i)+'.jpg')
#    im_emboss.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\Emb_'+str(i)+'.jpg')
    im_sharpen.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\Sharp_'+str(i)+'.jpg')
    im_smooth.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\1\\Smo_'+str(i)+'.jpg')
    print(i)
    
for i in range(250):
    m = random.randrange(0,len(neg2))
    im =Image.open(neg2[m])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\2\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\2\\unsh_'+str(i)+'.jpg')
    print(i)
    
for i in range(len(neg4)):
#    m = random.randrange(0,len(neg0))
    im =Image.open(neg4[i])
    im_blur = im.filter(ImageFilter.GaussianBlur)
    im_unsharp = im.filter(ImageFilter.UnsharpMask)
    im_sharpen = im.filter(ImageFilter.SHARPEN)
    im_smooth = im.filter(ImageFilter.SMOOTH)
    im_blur.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\4\\bl_'+str(i)+'.jpg')
    im_unsharp.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\4\\unsh_'+str(i)+'.jpg')
    im_sharpen.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\4\\Sharp_'+str(i)+'.jpg')
    im_smooth.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\4\\Smo_'+str(i)+'.jpg')
    print(i)

pos0 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\0', '*.jpg'))
pos1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1', '*.jpg'))
pos2 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\2', '*.jpg'))
pos3 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3', '*.jpg'))
pos4 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\4', '*.jpg'))
pos5 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\5', '*.jpg'))

for i in range(800):
    m = random.randrange(0,len(pos0))
    im =Image.open(pos0[m])
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out2 = out1.transpose(Image.FLIP_LEFT_RIGHT)
    out2.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\0\\flip_TBLR_'+str(i)+'.jpg')
    print(i)

for i in range(400):
    m = random.randrange(0,len(pos1))
    im =Image.open(pos1[m])
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out2 = out1.transpose(Image.FLIP_LEFT_RIGHT)
    out2.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1\\flip_TBLR_'+str(i)+'.jpg')
    print(i)
    
for i in range(800):
    m = random.randrange(0,len(pos2))
    im =Image.open(pos2[m])
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out2 = out1.transpose(Image.FLIP_LEFT_RIGHT)
    out2.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\2\\flip_TBLR_'+str(i)+'.jpg')
    print(i)

for i in range(900):
    m = random.randrange(0,len(pos3))
    im =Image.open(pos3[m])
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out2 = out1.transpose(Image.FLIP_LEFT_RIGHT)
    out2.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3\\flip_TBLR_'+str(i)+'.jpg')
    print(i)

for i in range(800):
    m = random.randrange(0,len(pos4))
    im =Image.open(pos4[m])
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out2 = out1.transpose(Image.FLIP_LEFT_RIGHT)
    out2.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\4\\flip_TBLR_'+str(i)+'.jpg')
    print(i)

for i in range(800):
    m = random.randrange(0,len(pos5))
    im =Image.open(pos5[m])
    out1 = im.transpose(Image.FLIP_TOP_BOTTOM)
    out2 = out1.transpose(Image.FLIP_LEFT_RIGHT)
    out2.save('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\5\\flip_TBLR_'+str(i)+'.jpg')
    print(i)

#%%
#converting it into a data frame

pos0 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\0', '*.jpg'))
pos1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\1', '*.jpg'))
pos2 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\2', '*.jpg'))
pos3 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\3', '*.jpg'))
pos4 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\4', '*.jpg'))
pos5 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster pos\\5', '*.jpg'))

neg0 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\6', '*.jpg'))
neg1 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\7', '*.jpg'))
neg2 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\8', '*.jpg'))
neg3 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\9', '*.jpg'))
neg4 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\10', '*.jpg'))
neg5 = glob.glob(os.path.join('C:\\Users\\Akshansh\\Desktop\\New folder\\Project tubercolosis\\New folder\\cluster neg\\11', '*.jpg'))

def toframe(di,cl):
    a = []
    for i in di:
        c = cv.imread(i,0)
        c.flatten()
        na = np.append(c,cl)
        a.append(na)
    return pd.DataFrame(a,index = None)

pos0df = toframe(pos0,0)
pos1df = toframe(pos1,1)
pos2df = toframe(pos2,2)
pos3df = toframe(pos3,3)
pos4df = toframe(pos4,4)
pos5df = toframe(pos5,5)
neg6df = toframe(neg0,6)
neg7df = toframe(neg1,7)
neg8df = toframe(neg2,8)
neg9df = toframe(neg3,9)
neg10df = toframe(neg4,10)
neg11df = toframe(neg5,11)

result_positives_new = pd.concat([pos0df,pos1df,pos2df,pos3df,pos4df,pos5df],axis = 0)
result_negatives_new = pd.concat([neg6df,neg7df,neg8df,neg9df,neg10df,neg11df],axis = 0)

result_positives_new.to_csv('result_pos_new.csv')
result_negatives_new.to_csv('result_neg_new.csv')

x = pd.read_csv('result_pos_new.csv')
y = pd.read_csv('result_neg_new.csv')
overall = pd.concat([x,y],axis = 0)
X = overall.iloc[:,1:50177].values
Y = overall.iloc[:,50177].values

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

count={}
for i in Y_train:
    if i in count.keys():
        count[i] += 1
    else:
        count[i] = 1
        
count2={}
for i in Y_test:
    if i in count2.keys():
        count2[i] += 1
    else:
        count2[i] = 1
    

X_train1,X_train2,X_train3,X_train4 = np.split(X_train,4) 
Y_train1,Y_train2,Y_train3,Y_train4 = np.split(Y_train,4)

X_train1 = pd.DataFrame(X_train1,index = None)
X_train2 = pd.DataFrame(X_train2,index = None)
X_train3 = pd.DataFrame(X_train3,index = None)
X_train4 = pd.DataFrame(X_train4,index = None)
X_test = pd.DataFrame(X_test,index = None)

X_train1.to_csv('X_train1.csv')
X_train2.to_csv('X_train2.csv')
X_train3.to_csv('X_train3.csv')
X_train4.to_csv('X_train4.csv')
X_test.to_csv('X_test_all.csv')

Y_train1 = pd.DataFrame(Y_train1,index = None)
Y_train2 = pd.DataFrame(Y_train2,index = None)
Y_train3 = pd.DataFrame(Y_train3,index = None)
Y_train4 = pd.DataFrame(Y_train4,index = None)
Y_test = pd.DataFrame(Y_test,index= None)

Y_train1.to_csv('Y_train1.csv')
Y_train2.to_csv('Y_train2.csv')
Y_train3.to_csv('Y_train3.csv')
Y_train4.to_csv('Y_train4.csv')
Y_test.to_csv('Y_test_all.csv')

#%%
import random
from matplotlib import pyplot as plt
import cv2 
def img_grid(l,tit):
    for i in range(9):
        rand = random.choice(l) 
        x = cv2.imread(rand,0)
        plt.subplot(3,3,i+1)
        plt.imshow(x,'gray')
        if i == 1:
            plt.title(tit)
        plt.xticks([]),plt.yticks([])
    return plt.show()
img_grid(neg0,'negative 6')
img_grid(neg1,'negative 7')
img_grid(neg2,'negative 8')
img_grid(neg3,'negative 9')
img_grid(neg4,'negative 10')
img_grid(neg5,'negative 11')
img_grid(pos0,'positive 0')
img_grid(pos1,'positive 1')
img_grid(pos2,'positive 2')
img_grid(pos3,'positive 3')
img_grid(pos4,'positive 4')
img_grid(pos5,'positive 5')

#%%
import pandas as pd
import numpy as np
Y_train1 = pd.read_csv('Y_train1.csv')
Y_train2 = pd.read_csv('Y_train2.csv')
Y_train3 = pd.read_csv('Y_train3.csv')
Y_train4 = pd.read_csv('Y_train4.csv')
Y_test_all = pd.read_csv('Y_test_all.csv')
Y_train1 = Y_train1.iloc[:,1:]
Y_train2 = Y_train2.iloc[:,1:]
Y_train3 = Y_train3.iloc[:,1:]
Y_train4 = Y_train4.iloc[:,1:]
Y_test_all = Y_test_all.iloc[:,1:]

X_train1 = pd.read_csv('X_train1.csv')
X_train2 = pd.read_csv('X_train2.csv')
X_train3 = pd.read_csv('X_train3.csv')
X_train4 = pd.read_csv('X_train4.csv')
X_test_all = pd.read_csv('X_test_all.csv')

X_train1 = X_train1.iloc[:,1:]
X_train2 = X_train2.iloc[:,1:]
X_train3 = X_train3.iloc[:,1:]
X_train4 = X_train4.iloc[:,1:]
X_test_all = X_test_all.iloc[:,1:]

total = pd.concat([X_train1,X_train2,X_train3,X_train4],axis = 0).values
total_Y = pd.concat([Y_train1,Y_train2,Y_train3,Y_train4],axis = 0).values
test = X_test_all.values
test_Y = Y_test_all.values
total_1 = total[0:10176,:]
total_2 = total[10176:20352,:]

import pickle as pkl
pkl.dump( total_Y, open( "total_Y.p", "wb" ) )
pkl.dump( total_1, open( "total_1.p", "wb" ) )
pkl.dump( total_2, open( "total_2.p", "wb" ) )
pkl.dump( test, open( "test.p", "wb" ) )
pkl.dump( test_Y, open( "test_Y.p", "wb" ) )
favorite_color = pkl.load( open( "total_Y.p", "rb" ) )

def count(x):
    c = {}
    for i in x.iloc[:,-1]:
        if str(i) in c.keys():
            c[str(i)] += 1
        else:
            c[str(i)] = 1
    return c
            
print(count(Y_train1))
print(count(Y_train2))
print(count(Y_train3))
print(count(Y_train4))
print(count(Y_test_all))  