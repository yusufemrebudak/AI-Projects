import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
data = pd.read_csv("column_2C_weka.csv")
data.rename(columns = {'class':'class_'}, inplace = True) 
data.class_ = [1 if each == "Normal" else 0 for each in data.class_]

x = data.loc[:,data.columns != 'class_']
y = data.loc[:,'class_']
# %%
from sklearn.model_selection import train_test_split
# x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)


# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
#%%
score_list = []
for each in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,30),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

# code have shown that the best Key value is 30