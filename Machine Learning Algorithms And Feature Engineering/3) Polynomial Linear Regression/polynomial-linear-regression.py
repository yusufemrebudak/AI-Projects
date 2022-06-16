# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:11:43 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial-regression.csv",sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

# linear regression =  y = b0 + b1*x
# multiple linear regression   y = b0 + b1*x1 + b2*x2

#  linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)


y_head = lr.predict(x)

plt.plot(x,y_head,color="red",label ="linear")
plt.show()

print("10 milyon tl lik araba hizi tahmini: ",lr.predict([[10000]]))

## YUKARIDAKİ LİNEAR REGERRESSON A GORE ÇALIŞMADI, POLİNOMİAL OLMALI

# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression  = PolynomialFeatures(degree=2) # x kare ye kadar gidiyor b0+b1*x'1+b2*x'2


x_polynomial = polynomial_regression.fit_transform(x)

linearRegression2 = LinearRegression()
linearRegression2.fit(x_polynomial, y)

y_head2 = linearRegression2.predict(x_polynomial)
plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()




























