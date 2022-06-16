# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:07:59 2018

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% create dataset

# class1
x1 = np.random.normal(25,5,1000) # normal demek gaussion demek, 1000 tane değerim olsun ortalaması
# ve %66 sı 20 ile 30 arasında olsun. gaussian ile alakalı
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)
# 3 farklı class ı yaratmış oldum. Bunu ben biliyorum ama cluster  bunu bilmeyecek. 
x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

dic = {"x":x, "y":y}
data = pd.DataFrame(dic) #DataFrame ile dataframe oluşturduk. 
# 2 freature ı olan 3000 tana sample olan dataFrame 
data.describe()

# plt.scatter(x1,y1,color="black")
# plt.scatter(x2,y2,color="black")
# plt.scatter(x3,y3,color="black")
# plt.show() # görünüyor ku bunlar epey birbirinden ayrılmış class lar.Ama simsiyah hepsi çünkü sınıflarını bilmiyoruz. 
# algoritma sadece dataları görecek ve bunları sınıflandırmaya çalışcak.
#%% K-means algorithm
from sklearn.cluster import KMeans
wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) # her bir key değeri ve fitleme ardından inertia yani wcss değerini bul diyorum
    # burada optimum k değeri bulunacak bu verimin kaç class a ayrılacağını söyler
plt.plot(range(1,15),wcss)
plt.xlabel("number of K Value")
plt.ylabel("wcss")
plt.show() 
# elbow yaptığı yer o eklem bölgesi k nın olması gereken değer.Cluster sayım 3 olsun diyorum yani K sayım
# %% k=3 için Modelim
kmean2 = KMeans(n_clusters=3)
clusters = kmean2.fit_predict(data)  # fit ve predict et yani bana sınıfılarımı söyle
data["label"] = clusters
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue")
plt.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1],color="yellow")
plt.show()

























