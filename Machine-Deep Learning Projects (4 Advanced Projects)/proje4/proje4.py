import pandas as pd
import seaborn as sns #  visualization library
import numpy as np
import matplotlib.pyplot as plt # other visualization library

from scipy import stats # 
from scipy.stats import norm, skew 

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # linear models 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin ,clone # tüm modelleri birleştirirken kullanılacak.

# XGBoost
import xgboost as xgb

# warning 
import warnings
warnings.filterwarnings('ignore')

column_name = ["MPG", "Cylinders", "Displacement","Horsepower","Weight","Acceleration","Model Year", "Origin"]
data = pd.read_csv("auto-mpg.data", names = column_name, na_values = "?", comment = "\t",sep = " ", skipinitialspace = True)
# içinde nan value varsa bunları ? olarak al
# commentleri tabla 
# veri boşluklarla ayrıldığı için sep=" ",
# skipinitialspace = true, initial space var boşluk bunu true diyip atlayacağız.

data = data.rename(columns={"MPG":"target"})
#%%
print(data.head())
print("Data Shape: ", data.shape)
data.info()

# 0   target        398 non-null    float64
# 1   Cylinders     398 non-null    int64  
# 2   Displacement  398 non-null    float64
# 3   Horsepower    392 non-null    float64 #horsepower da missing value var, nan olarak
# 4   Weight        398 non-null    float64
# 5   Acceleration  398 non-null    float64
# 6   Model Year    398 non-null    int64  
# 7   Origin        398 non-null    int64 
# there is no categorical feature in here except origin, 
# in those cateogrial features there is no any mathematical meaning ,
describe = data.describe()
# my data generally consist of numerical features
# in some of my dat, there is skewnes, we need to deal with
#%% handling missing values
# we will not remove the missing value, because we have data loss, we dont want that
# we have to handle these missing values,ok?
print(data.isna().sum())
data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())
print(data.isna().sum())

sns.displot(data.Horsepower)
#%% EDA
corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Correlation Between Features")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist() # threshold u 0.75 den büyük olan featureları aldım.
sns.clustermap(data[corr_features].corr(), annot=True,fmt=".2f")
plt.title("Correlation Between Features")
plt.show()

# featureların birbirleriyleyüksek koralasyonda olması çok iyi değildir. Bunun anlamı 1feature  mesela 5 featurlaa uğraşıyoruz demektir.

sns.pairplot(data, diag_kind="kde", markers="+")
plt.show()
"""
cylinders and origin can be categorical feature (feature engineering)
"""
plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

# box
for c in data.columns:
    plt.figure()
    sns.boxplot(x=c, data=data, orient="v")

"""
box plottan baktığımızda, çubukların altında ve üstünde kalan değerler outlier olarak adlandırılır.
outlier: Horspower, acceleration
"""
#%% outlier: veri içindeki featurlardaki aykırı değerler
thr = 2
horsepower_desc = describe["Horsepower"] # ı am getting the infrmation about Horsepower
q3_hp = horsepower_desc[6] 
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr * IQR_hp
bottom_limit_hp = q1_hp - thr * IQR_hp
filter_hp_bottom = bottom_limit_hp<data["Horsepower"]
filter_hp_top = data["Horsepower"]<top_limit_hp
# i want to detect outliers
filter_hp = filter_hp_bottom & filter_hp_top 
data = data[filter_hp]
# there is only a outlier, :)
    

acceleration_desc = describe["Acceleration"]
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc # q3 - q1
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top= data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc] # remove Horsepower outliers
# there is only two outlier, :), 
# those outliers ruin my data, that's why i must remove these data that have outlier valuesfrom my dataset  
print("Data Shape: ", data.shape) # Data Shape:  (395, 8)
#%% feature engineering
# skewness

# target dependent variable
sns.distplot(data.target,fit=norm)# right skewness vardır
(mu,sigma) = norm.fit(data["target"])
print("mu: {}, sigma: {}".format(mu,sigma)) # mu: 23.472405063291134, sigma: 7.756119546409932

# qq plot
fig = plt.figure()
stats.probplot(data["target"],plot = plt)
plt.show()
# şuan normal dağılıma sahip değil, grafikler tam oturmuyor

# skewneslığı azaltmak için log transformu gerçekleştireceğiz.
data["target"] = np.log1p(data["target"])
plt.figure()
sns.distplot(data.target, fit = norm)# right skewness vardır
(mu,sigma) = norm.fit(data["target"])
print("mu: {}, sigma: {}".format(mu,sigma)) # mu: 3.146474056830183, sigma: 0.3227569103044823

#qq plot
fig = plt.figure()
stats.probplot(data["target"],plot = plt)
plt.show()

#%%feature- independent variable
skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False) 
# herhangi bir nan değer varsa bunları düşür öyle skew metoduna sok diyorum
skewness = pd.DataFrame(skewed_feats, columns=["skewed"])
# 1 den büyükse pozitif skewness var demiştik, -1 den küçükse negatif skew var
"""
Box Cox Transformation
"""
# %% one hot encoder
data["Cylinders"] = data["Cylinders"].astype(str) # categorical veriler stringlerden oluşur ondan dolayı astype(str) yazıyoruz.
data["Origin"] = data["Origin"].astype(str)
data = pd.get_dummies(data)


















