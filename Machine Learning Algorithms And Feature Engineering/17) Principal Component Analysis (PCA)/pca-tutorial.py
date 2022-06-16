

from sklearn.datasets import load_iris # sklearn in data set ini kullanıyorun
import pandas as pd

# %% data frame oluşturma iris kütüphanesinden
iris = load_iris()
data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns=feature_names)
df["sinif"] = y

x = data
#%%
from sklearn.decomposition import PCA
# amaç 4 feature ı 2 boyuta düşürmek
pca = PCA(n_components=2, whiten=True) # whiten=true demek normalize etmek demektir. bir fetaruren diğerini domine etmemesi için
pca.fit(x) # şuan pca modelini oluşturdum 
# ve mu modeli x datam üzerinde uyguluyorum ve feature sayısını indiriyorum.
x_pca = pca.transform(x) # şuan 4 feature ı 2 ye indirdim, biri princibal compononet biri second componont
# hangisi hangisi anlamak için
print("variance ratio: ",pca.explained_variance_ratio_)
# [0.92461872 0.05306648] büyük olan benim princibal componentim
# 4 den 2 ye çektim ama bakalım data mın bana sağladığı bilgileri ne kadar korunmuşum 
print("sum: ",sum(pca.explained_variance_ratio_)) # sum:  0.977685206318795 çok az bir veri kaybı yaşamışım sıkıntı yok
# 4 den 2 ye düşdüm ama neredeyse hiçbir şey kaybetmedim yani varyansımı korumuş oluyorum
#%%
df["p1"] = x_pca[:,0] # p1 kolonuna princibal componont imi atıyorum
df["p2"] = x_pca[:,1] # p2 kolonuna second componont imi atıyorum
color=["red","green","blue"]
import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif==each],df.p2[df.sinif==each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
    
    
    
    
    
    
    
    
    
    
    

























