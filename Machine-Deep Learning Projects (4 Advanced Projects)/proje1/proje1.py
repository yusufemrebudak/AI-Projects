import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler # for standardization
from sklearn.model_selection import train_test_split, GridSearchCV 
# GridSearchCV - will find best parameters for KNN
from sklearn.metrics import accuracy_score, confusion_matrix
# confusion_matrix -> TP-TN-FN-FP 
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# warning library
import warnings
warnings.filterwarnings("ignore")
#%% dataset
data = pd.read_csv("cancer.csv")
data.drop(["Unnamed: 32","id"],inplace=True,axis= 1)
data = data.rename(columns={"diagnosis":"target"})
sns.countplot(data["target"])

#%% 
data["target"] = [1 if i.strip() == "M" else 0  for i in data.target]
#%%
print("Length of Data: ",len(data))
print("Data Shape: ",data.shape)
data.info() # quick review to dataset
describe = data.describe() # There is huge scale difference between datas, we need to standardize  
#%% EDA

# correlation
# eğer categorical yani string değerlerde olsaydı corr() otomatikman onları ignore edecekti.
corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Correlation between features")

threshold = 0.75
filtre = np.abs(corr_matrix["target"])>threshold # abs absolute value mutlak değer
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot=True,fmt=".2f")
plt.title("Correlation between features w Corr threshold")
"""
there are some correlated features,
bu featureları farklı veri setlerinde falan bunları ortadan kaldırabiliriz yada regularization
"""

# box plot
data_melted = pd.melt(data, id_vars = "target"
                      ,var_name="features",
                      value_name="value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue="target", data=data_melted)
plt.xticks(rotation=90)
plt.show()
"""
standardization-normalization
"""
# pair plot
sns.pairplot(data[corr_features], diag_kind="kde",markers="+",hue="target")
plt.show()
#%% outlier detection
y = data.target
x = data.drop(["target"],axis=1)
columns = x.columns.tolist()
clf = LocalOutlierFactor()# bu yöntemi kullanırken kaç tane neighbor seçeceğimiz önemli idi.
# n_neighbor parametresi önemli default olarak 20 dir.
y_pred = clf.fit_predict(x) # 1 ve -1 değerleri döndürür. 


X_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = X_score
#treshold
threshold =  -2.0
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist() 



plt.figure()
plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color="blue",s=50,label="Outliers")

plt.scatter(x.iloc[:,0],x.iloc[:,1],color="k",s=3,label="Data Points")

radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())  # normalization
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none",label="outlier scores")
plt.legend()# labellar görünsün diye
plt.show()

#%%drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values
#train_test_split
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=test_size, random_state=42) # shuffle=True -> default
# random_State her zaman aynı shuffle ı yapmamızı sağlayan parametre 
# datanın split edilme şeklini sabitliyorum ve datanın farklı shuffle edilmesinden dolayı oluşabilecek acc değişimlerini ignore edebiliyorum.
#%% standartization 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_train e göre eğitilmil scaler ımı X_test için uyguluyorum ve X_test i scale etmiş oluyorum.
# verimin mean(u) sunu 0 a std sini 1 çektim. Kaymalarında çok bii değişiklik olmadı. 
X_train_df = pd.DataFrame(X_train,columns=columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train
#%% boxplot
data_melted = pd.melt(X_train_df, id_vars = "target"
                      ,var_name="features",
                      value_name="value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue="target", data=data_melted)
plt.xticks(rotation=90)
plt.show()
# plottan çıkarılan yorum;
    # bazı featurelar class ları çok iyi ayırmış bunlar benim için en anlamlı classlar,
    # bazıları iyi ayrılmamış , data pointler iç içe yani 
    # bazı featurlar için çok yüksek outlier değerleri bulunabilir, bunlar çıkarılmalıdır.

#%%
sns.pairplot(X_train_df[corr_features], diag_kind="kde",markers="+",hue="target")
plt.show()
#%% Basic KNN method

knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
acc = accuracy_score(Y_test,y_pred)
score = knn.score(X_test,Y_test)
print("Score: ",score)
print("Basic KNN Score: ",score)
print("CM: ",cm)
"""
CM:  [[107   0]
      [  9  54] ,  107 tanesini iyi huylu olarak tahminim var 0 tanesini yanlış tahmin etmişim yani iyi huyluların hepsini doğru tahmin etmişim.
                   63 tane kötü huylu verim  54 tanesine kötü huylu 9 tanesine iyi huylu demişim yani 9 tanesini yanlış tahmin etmişim.   
"""

# overfitting veriyi ezberleme yeni veri geldiğinde doğru tahmin edemez, high variance
# underfitting hiç öğrenememesi, high bias ortaya çıkar,
# good balance, low bias, low variance
#%% choose knn best parameters
def KNN_best_params(x_train, x_test, y_train, y_test):
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    # gridSearch için gerekli parametreleri bir dict e koymam gerekiyor
    param_grid=dict(n_neighbors = k_range, weights = weight_options)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,param_grid, cv = 10, scoring="accuracy")
    # machine learning modeli olarak gridsearch yaparken knn kullan parametrelerinide param_grid den al.
    grid.fit(x_train,y_train)
    print("Best training score: {} with parameters: {}".format(grid.best_score_ ,grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)
    
    y_pred_test = knn.predict(x_test)    
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test) 
    cm_train = confusion_matrix(y_train, y_pred_train)    

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print("Test score: {}, Train score: {}".format(acc_test,acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)

    return grid


grid = KNN_best_params(X_train, X_test, Y_train, Y_test)
# Test score: 0.9470588235294117, Train score: 1.0 burada train skoru testten yüksek çıkmış
# burada bir overfit söz konusu %6 lık, test veri setinde az da olsa azalma vardır ve ezberleme vardır train datası üzeirnde

#%%
x = [2.4, 0.6, 2.1, 2, 3, 2.5, 1.9, 1.1, 1.5, 1.2]
y = [2.5, 0.7,2.9,2.2,3.0,2.3,2.0,1.1,1.6,0.8]
x = np.array(x)
y = np.array(y)
# plt.scatter(x,y)
x_mean = np.mean(x)
y_mean = np.mean(y)
x = x-x_mean
y = y-y_mean
plt.scatter(x,y) # plotu 0 merkezlerine çektim
# kovaryans matrisi bulmak için
cov = np.cov(x,y)
print("con(x,y): ",cov)
"""
con(x,y):  [[0.53344444 0.56411111]
 [0.56411111 0.68988889]]
"""
from numpy import linalg as LA
w , v = LA.eig(cov)
# w eigen value ları içinde barındırır
# v eigen vektörleri içinde barındır.
print("w : ",w) 
# w :  [0.04215805 1.18117528]
print("v : ",v)
# v :  [[-0.75410555 -0.65675324]
#  [ 0.65675324 -0.75410555]]
# v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
p1 = v[:,1]
p2 = v[:,0]
plt.plot([0,p1[0]], [0,p1[1]])
#%% PCA 
# veriyi standardize etmeliyiz, pca unsupervised algoritmadır, herhangi bir class label ına ihtiyaç yoktur
# bundan dolayı train ve test split diye ayırmaya gerek kalmadan elimdeki tüm veriyi kullanabilirim.
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components = 2) # feature sayısını 2 ye düşürmek istedim 
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns=["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x="p1",y = "p2", hue="target", data=pca_data)
plt.title("PCA: p1 vs p2 ")


X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca,y,test_size=test_size, random_state=42) # shuffle=True -> default
grid_pca = KNN_best_params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)


#%% visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = X_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))
#%% NCA
nca = NeighborhoodComponentsAnalysis(n_components=2,random_state = 2)# supervised dır.
nca.fit(x_scaled,y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca , columns=["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x="p1", y="p2", hue="target", data=nca_data)
plt.title("NCA:p1  vs p2")

# pca e göre daha iyi bir ayrım söz konusu


X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca,y,test_size=test_size, random_state=42) # shuffle=True -> default
grid_nca = KNN_best_params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)


#visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))
# %% find wrong decision
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)

diff = np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)





















