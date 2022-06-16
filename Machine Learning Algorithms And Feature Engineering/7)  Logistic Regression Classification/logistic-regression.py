
#  libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  read csv
data = pd.read_csv("data.csv")

print(data.info())

data.drop(["Unnamed: 32","id"],axis=1,inplace = True) # tüm bir column ı drop ediyoruz.
# inplace = True drop et ve veriye kaydet demek 

data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis]
# eğer M ise 1 olcak değilse 0 olacak, tüm veriyi gez 0 ı veya 1 i set et. 
print(data.info())


y = data.diagnosis.values # .values bunları numoy arraye çevirir
x_data = data.drop(["diagnosis"],axis=1) # bunu da düşürdükten sonra artık
# ben sadece data nın içinde featurlarımı bırakmış oldum 

# normalization, tüm featurları 0 ve 1 arasında normalize etmem lazım
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


















































