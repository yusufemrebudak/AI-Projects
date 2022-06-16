import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#import data
df = pd.read_csv("multiple-linear-regression-dataset.csv",sep=";")
# benim train etmem gereke featurlar deneyim ve yaş bunlar maaşı etkileyecek,
x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

multipleLinearReg = LinearRegression()
multipleLinearReg.fit(x,y)

print("b0: ",multipleLinearReg.intercept_)
print("b1:, b2: ",multipleLinearReg.coef_)

multipleLinearReg.predict(np.array([[10,35],[5,35]])) # birden fazla feature maaşı etkiliyor
# bize 2 tane bu featurlar için coeef verir yani parametre bunları vererek predict e bize bir değer predict 
#etmesini söylüyoruz.