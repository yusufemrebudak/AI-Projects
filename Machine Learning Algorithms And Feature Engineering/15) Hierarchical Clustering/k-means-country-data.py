# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import AgglomerativeClustering



df = pd.read_csv('countryData.csv')

data=df.drop(['country'],axis=1)

scaling = StandardScaler()
scaled=scaling.fit_transform(data)
# Our dataset is not scaled some values are much bigger than others,
#if we will not scale our data our model will not going to perform well.
#So now we are are going to scale our data for this we are going to use a StandardScaler library
# StandardScaler transform the data such the the mean will be 0 and

scaled_df = pd.DataFrame(scaled,columns=data.columns)
# princt scaled dataset
scaled_df.head()


a=[]
K=range(1,10)
for i in K:
    kmean=KMeans(n_clusters=i)
    kmean.fit(data)
    a.append(kmean.inertia_)
    
plt.plot(K,a,marker='o')
plt.title('Elbow Method',fontsize=15)
plt.xlabel('Number of clusters',fontsize=15)
plt.ylabel('Sum of Squared distance',fontsize=15)
plt.show()


kmeans = KMeans(n_clusters = 3,random_state = 111)
kmeans.fit(scaled_df)
cluster_labels = kmeans.fit_predict(scaled_df)
preds = kmeans.labels_
kmeans_df = pd.DataFrame(df)
kmeans_df['KMeans_Clusters'] = preds

kmeans_df.to_csv('kmeans_result.csv',index=False)

# sns.scatterplot(kmeans_df['child_mort'],kmeans_df['gdpp'],hue='KMeans_Clusters',data=kmeans_df) 
# plt.title("Child Mortality vs gdpp", fontsize=15)
# plt.xlabel("Child Mortality", fontsize=12)
# plt.ylabel("gdpp", fontsize=12)
# plt.show()

sns.scatterplot(kmeans_df['inflation'],kmeans_df['gdpp'],hue='KMeans_Clusters',data=kmeans_df) 
plt.title("inflation vs gdpp", fontsize=15)
plt.xlabel("inflation", fontsize=12)
plt.ylabel("gdpp", fontsize=12)
plt.show()











