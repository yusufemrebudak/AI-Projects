# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 23:53:33 2018

@author: user
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


#%%
iris = load_iris()

x = iris.data
y = iris.target

# %% normalization , knn algoritmasında normalization şart, featureların birbirini domine etmemesi için
x = (x-np.min(x))/(np.max(x)-np.min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
#%% knn modelini kullacağız
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
#%% k fold K=10 seçicem, x_train datasını 10 a böl 1 ini sadece valiadation olarak kullan
# ve bunu 10 farklı şekilde seç bana 10 tane acc ver 10 a böl ortalamasını bana ver
from sklearn.model_selection import cross_val_score
# cross validation yaparken knn algoritmasını kullan demek
accuracies = cross_val_score(estimator = knn, X=x_train, y=y_train, cv=10)
print("average acc : ",np.mean(accuracies)) # 0.9818181818181818 
print("average std  : ",np.std(accuracies)) # 0.036363636363636376 

# şuan karar verdik k=3 iyi değer artık ilk başta hiç kullanmadığımız test verileriyle test edeceğiz.
knn.fit(x_train,y_train)
print("test accuracy", knn.score(x_test,y_test))
      
#%% grid search cross validation 
# hem knn algoritmasındaki k yi farklı değerler için deniyoruz, hem de her bir farklı değer için 
#cross validation uygularken cross validation yapıyoruz.

from sklearn.model_selection import GridSearchCV

grid={"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10) # GridSearchCV, cv=10 olsun ,
# 1 den 50 ye kadar k değerleri için validation 10 değeri için yap. grid gibi bişey olacak
knn_cv.fit(x,y)

#%% print hyperparameter KNN algoritmasındaki K değeri
print("tuned hyperparameter K: ", knn_cv.best_params_)
print("tuned parametreye göre en iyi acc (best score): ", knn_cv.best_score_)
# tuned hyperparameter K:  {'n_neighbors': 13}
# tuned parametreye göre en iyi acc (best score):  0.9800000000000001
# %98 acc veren değerim knn algoritmasında k nın 14 olduğu hal imiş. 
########## GRİD SEARCH VALİDATION  ile knn algoritmasını birleştirdik #############
# k sayısını elden kafadan seçmek yerine en iyi k değerini bize veriypr zaten gridsearch validation
#%% grid search validation with logisticRegression
x = x[:100,:]
y = y[:100]
from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7)} # bu c parametresi logistic regressionda regularization parametresidir.
# eğer c büyükse overfit(ezberleme ) olur, c küçükse de underfit olur yani hiçbir şekilde datayı öğrenememe..
# penaly ise loss değerleri l1 = losso ve l2=ridge parametreleri

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(x,y)
print("tuned hyperparameter :(best parameteres): ", logreg_cv.best_params_)
print("en iyi acc (best score): ", logreg_cv.best_score_)












































