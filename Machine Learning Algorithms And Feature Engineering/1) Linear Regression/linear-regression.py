import pandas as pd
import matplotlib.pyplot as plt
#import data
df = pd.read_csv("linear-regression-dataset.csv",sep=";")
#plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()
#%% sklearn- scikit-learn (bunun içinde machine learning algoritmaları ve bir sürü modeller var)
import numpy as np
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()
# x = df.deneyim.values # .values kısmı pandas serilerini numpy a çevirmek için 
# y = df.maas.values
# y.shape (14,) sklearn bunu anlamaz (14,1) görmek ister
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y) # artık benim noktalarıma bir line fit edildi.
## MSE ü en düşük yapacak şekilde b0 ve b1 değeri üretiliyor.

b0 = linear_reg.predict([[0]])
print("b0: ",b0)

b0_ = linear_reg.intercept_ # b0 ı bulmak için 
print("b0_: ",b0_)

# b1 i bulacağız bu coeff idi katsayı idi yani "eğim"
b1 = linear_reg.coef_
print("b1: ",b1)

# maas = 1663 + 1138*deneyim
maas_yeni = 1663 + 1138*11
print("maas_yeni: ",maas_yeni) # 14181

maas_yeni_predict = linear_reg.predict([[11]])

print("maas_yeni_predict: ",maas_yeni_predict) # 14185, MANTIKLI BİR SONUÇ GELDİ

#predict edilen line a bakalım nasıl kesişiyor bizimkiyle
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # (16,1)
plt.scatter(x,y)
plt.show()


y_head = linear_reg.predict(array)

plt.plot(array,y_head,color="red")









