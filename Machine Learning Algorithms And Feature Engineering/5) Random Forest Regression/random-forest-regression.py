# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:09:38 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random-forest-regression-dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
# random foresst ensemble learning in machine learning algorithmalarından bir tanesi idi. 
rf = RandomForestRegressor(n_estimators = 100,random_state = 42)
# n_estimators 100 demek bu rnadom forest metodunun içinde 100 tane tree kullanmak istiyorum diyorum
# random_state demek : datamdan bir subsdata yaratıyorum n adet sample seçmem lazım data setimden, bu n adet sample random 
# seçilir genelde. ama 42 yazarsak random seç ama 2.kez run ettiğimde yine datayı aynı şekilde böl bana aynı sonucu ver.
# bize er çalışmada aynı sonucu vermesi için 42 yazıyorum
rf.fit(x,y)
print("7.5 seviyesinde fiyat ne kadar: ",rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribün level")
plt.ylabel("ucret")
plt.show()

# bunun farkı 1 tane decision tree kullanmak yerine 100 tane kullanmıştır.